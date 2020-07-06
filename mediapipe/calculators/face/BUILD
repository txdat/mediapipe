load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "face_crop_and_align_calculator",
    srcs = ["face_crop_and_align_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:face_data_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//third_party:glog",
    ],
    alwayslink = 1,
)
