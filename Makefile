aar:
	bazel build -c opt --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_example:mp_face_mesh_aar
graph:
	bazel build -c opt mediapipe/mediapipe/graphs/face_mesh:face_mesh_mobile_gpu_binary_graph
all:
	make aar && make graph
