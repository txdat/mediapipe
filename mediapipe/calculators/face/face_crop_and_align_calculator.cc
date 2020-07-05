#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/face_data.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#define RADIAN_TO_DEGREE(r) (r) * 180 / M_PI

namespace mediapipe {

// STREAM_NAME
constexpr char INPUT_IMAGE[] = "IMAGE"; // ImageFrame
constexpr char INPUT_RECTS[] =
    "NORM_RECTS"; // std::vector<NormalizedRect>, in [0, 1]
constexpr char INPUT_LANDMARKS[] =
    "LANDMARKS"; // std::vector<NormalizedLandmarkList>, in [0, 1]
constexpr char INPUT_TARGET_SIZE[] = "TARGET_SIZE"; // int (side_packet)
constexpr char OUTPUT[] = "FACE_LANDMARKS"; // std::vector<FaceLandmarks>

//----------------------------------------
// ROTATED RECTANGLE CROPPING WITH POINTS
//----------------------------------------

inline std::vector<cv::Point3f>
convertNormalizedLandmarksToPoints(const NormalizedLandmarkList &landmark_list,
                                   float image_width, float image_height) {
  std::vector<cv::Point3f> landmark_points;
  for (int i = 0; i < landmark_list.landmark_size(); i++) { // skip z-axis
    const NormalizedLandmark &landmark = landmark_list.landmark(i);
    landmark_points.push_back(cv::Point3f(
        landmark.x() * image_width, landmark.y() * image_height, landmark.z()));
  }

  return landmark_points;
}

inline NormalizedLandmarkList convertPointsToNormalizedLandmarks(
    const std::vector<cv::Point3f> &landmark_points,
    const NormalizedLandmarkList &landmark_list, float image_width,
    float image_height) {
  NormalizedLandmarkList landmark_list_;
  for (int i = 0; i < landmark_points.size(); i++) { // skip z-axis
    NormalizedLandmark landmark;
    landmark.set_x(landmark_points[i].x / image_width);
    landmark.set_y(landmark_points[i].y / image_height);
    landmark.set_z(landmark_points[i].z);
    landmark.set_visibility(landmark_list.landmark(i).visibility());

    landmark_list_.add_landmark(landmark);
  }

  return landmark_list_;
}

inline cv::Rect computeRectBbox(const NormalizedRect &rect, float image_width,
                                float image_height, float size_) {
  cv::RotatedRect rotated_rect = cv::RotatedRect(
      cv::Point2f(rect.x_center() * image_width,
                  rect.y_center() * image_height),
      cv::Size2f(size_, size_), RADIAN_TO_DEGREE(rect.rotation()));

  return rotated_rect.boundingRect();
}

inline cv::Mat cropImageWithPoints(const cv::Mat &image, const cv::Rect &rect,
                                   std::vector<cv::Point3f> &points) {
  for (cv::Point3f &point : points) { // skip z-axis
    point.x -= rect.x;
    point.y -= rect.y;
  }

  return image(rect);
}

inline cv::Mat rotateImageWithPoints(const cv::Mat &image,
                                     std::vector<cv::Point3f> &points,
                                     double rotation) {
  cv::Mat rotation_mat = cv::getRotationMatrix2D(
      cv::Point2f(float(image.cols) / 2.0, float(image.rows) / 2.0), rotation,
      /*scale*/ 1.0);

  double abs_cos = std::abs(rotation_mat.at<double>(0, 0));
  double abs_sin = std::abs(rotation_mat.at<double>(0, 1));
  double bound_w = abs_sin * image.rows + abs_cos * image.cols;
  double bound_h = abs_cos * image.rows + abs_sin * image.cols;

  rotation_mat.at<double>(0, 2) += bound_w / 2 - double(image.cols) / 2;
  rotation_mat.at<double>(1, 2) += bound_h / 2 - double(image.rows) / 2;

  // rotate image
  cv::Mat rotated_image;
  cv::warpAffine(image, rotated_image, rotation_mat, cv::Size(bound_w, bound_h),
                 cv::INTER_CUBIC);

  // rotate points
  cv::Mat point_mat = cv::Mat(points).reshape(1);
  point_mat = (rotation_mat * point_mat.t()).t();

  for (int i = 0; i < points.size(); i++) { // skip z-axis
    points[i].x = point_mat.at<double>(i, 0);
    points[i].y = point_mat.at<double>(i, 1);
  }

  return rotated_image;
}

inline CvMat serializeCvMat(const cv::Mat &mat) { // encode mat to bytes
  CvMat encoded_mat;
  encoded_mat.set_rows(mat.rows);
  encoded_mat.set_cols(mat.cols);
  encoded_mat.set_elt_type(mat.type());
  encoded_mat.set_elt_size(int(mat.elemSize()));
  encoded_mat.set_data(mat.data, mat.rows * mat.cols * mat.elemSize());

  return encoded_mat;
}

class FaceCropAndAlignCalculator : public CalculatorBase {
private:
  cv::Size target_size = cv::Size(224, 224); // face's size after rescaling

public:
  static ::mediapipe::Status GetContract(CalculatorContract *cc) {
    // check stream_names' tags (required)
    RET_CHECK(cc->Inputs().HasTag(INPUT_IMAGE));
    RET_CHECK(cc->Inputs().HasTag(INPUT_RECTS));
    RET_CHECK(cc->Inputs().HasTag(INPUT_LANDMARKS));
    RET_CHECK(cc->Outputs().HasTag(OUTPUT));

    // set stream_names' tags
    cc->Inputs().Tag(INPUT_IMAGE).Set<InputFrame>();
    cc->Inputs().Tag(INPUT_RECTS).Set<std::vector<NormalizedRect>>();
    cc->Inputs()
        .Tag(INPUT_LANDMARKS)
        .Set<std::vector<NormalizedLandmarkList>>();
    cc->Outputs().Tag(OUTPUT).Set<std::vector<FaceLandmarks>>();

    // check side_packet's tag (optional)
    if (cc->InputSidePackets().HasTag(INPUT_TARGET_SIZE)) {
      cc->InputSidePackets().Tag(INPUT_TARGET_SIZE).Set<int>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext *cc) {
    cc->SetOffset(TimestampDiff(0));

    if (cc->InputSidePackets().HasTag(
            INPUT_TARGET_SIZE)) { // get new target_size if available
      int size_ = cc->InputSidePackets().Tag(INPUT_TARGET_SIZE).Get<int>();
      target_size = cv::Size(size_, size_);
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext *cc) {
    // GET INPUTS FROM STREAMS...
    const ImageFrame &input_frame =
        cc->Inputs().Tag(INPUT_IMAGE).Get<ImageFrame>();
    cv::Mat image = formats::MatView(&input_frame);
    float image_width = image.cols, image_height = image.rows;

    const std::vector<NormalizedRect> &rects =
        cc->Inputs().Tag(INPUT_RECTS).Get<std::vector<NormalizedRect>>();

    const std::vector<NormalizedLandmarkList> &multi_face_landmarks =
        cc->Inputs()
            .Tag(INPUT_LANDMARKS)
            .Get<std::vector<NormalizedLandmarkList>>();

    // COMPUTE FACE'S IMAGE AND LANDMARKS...
    std::vector<FaceLandmarks> multi_faces_with_landmarks;
    for (int i = 0; i < rects.size(); i++) {
      float size_ = std::max(rect.width(), rect.height());
      cv::Rect bbox =
          computeRectBbox(rects[i], image_width, image_height,
                          size_); // return squared bbox (size_Xsize_)
      std::vector<cv::Point3f> landmark_points =
          convertNormalizedLandmarksToPoints(
              multi_face_landmarks[i], image_width,
              image_height); // unnormalize landmarks

      cv::Mat roi = cropImageWithPoints(image, bbox, landmark_points);
      roi = rotateImageWithPoints(roi, landmark_points,
                                  RADIAN_TO_DEGREE(rects[i].rotation()));
      roi = cropImageWithPoints(roi,
                                cv::Rect(float(roi.cols) / 2 - size_ / 2,
                                         float(roi.rows) / 2 - size_ / 2, size_,
                                         size_),
                                landmark_points);
      cv::resize(roi, roi, target_size, 0, 0, cv::INTER_CUBIC);

      FaceLandmarks face_with_landmarks;
      face_with_landmarks.set_data(serializeCvMat(roi));
      face_with_landmarks.set_landmarks(convertPointsToNormalizedLandmarks(
          landmark_points, multi_face_landmarks[i], size_,
          size_)); // not affected by rescaling
      multi_face_with_landmarks.push_back(face_with_landmarks);
    }

    // GENERATE OUTPUT...
    cc->Outputs().Tag(OUTPUT).Add(multi_face_with_landmarks,
                                  cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(FaceCropAndAlignCalculator);

} // namespace mediapipe
