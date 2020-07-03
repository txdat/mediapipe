#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/face_data.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

#include <cmath>
#include <memory>
#include <vector>

namespace mediapipe {

// STREAM_NAME
constexpr char INPUT_IMAGE[] = "IMAGE"; // ImageFrame
constexpr char INPUT_RECTS[] =
    "NORM_RECTS"; // std::vector<NormalizedRect>, in [0, 1]
constexpr char INPUT_LANDMARKS[] =
    "LANDMARKS"; // std::vector<NormalizedLandmarkList>, in [0, 1]
constexpr char INPUT_TARGET_SIZE[] = "TARGET_SIZE"; // int (side_packet)
constexpr char OUTPUT[] = "FACE_LANDMARKS"; // std::vector<FaceLandmarks>

//--------------------------------
// Rotated Rectangle Cropping
//--------------------------------

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
  int target_size = 224; // face's size after rescaling

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
      target_size = cc->InputSidePackets().Tag(INPUT_TARGET_SIZE).Get<int>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext *cc) {
    // GET INPUTS FROM STREAMS...
    const ImageFrame &input_frame =
        cc->Inputs().Tag(INPUT_IMAGE).Get<ImageFrame>();
    cv::Mat image = formats::MatView(&input_frame);

    const std::vector<NormalizedRect> &rects =
        cc->Inputs().Tag(INPUT_RECTS).Get<std::vector<NormalizedRect>>();

    const std::vector<NormalizedLandmarkList> &multi_face_landmarks =
        cc->Inputs()
            .Tag(INPUT_LANDMARKS)
            .Get<std::vector<NormalizedLandmarkList>>();

    int num_faces = rects.size();

    // COMPUTE FACE'S IMAGE AND LANDMARKS...
    for (int i = 0; i < num_faces; i++) {
      // rotate and crop rect
    }

    // GENERATE OUTPUT...

    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(FaceCropAndAlignCalculator);

} // namespace mediapipe
