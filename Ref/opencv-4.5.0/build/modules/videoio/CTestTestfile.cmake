# CMake generated Testfile for 
# Source directory: /home/bill/下载/opencv-4.5.0/modules/videoio
# Build directory: /home/bill/下载/opencv-4.5.0/build/modules/videoio
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_videoio "/home/bill/下载/opencv-4.5.0/build/bin/opencv_test_videoio" "--gtest_output=xml:opencv_test_videoio.xml")
set_tests_properties(opencv_test_videoio PROPERTIES  LABELS "Main;opencv_videoio;Accuracy" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;199;ocv_add_accuracy_tests;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;0;")
add_test(opencv_perf_videoio "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml")
set_tests_properties(opencv_perf_videoio PROPERTIES  LABELS "Main;opencv_videoio;Performance" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/performance" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1212;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;200;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;0;")
add_test(opencv_sanity_videoio "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_videoio PROPERTIES  LABELS "Main;opencv_videoio;Sanity" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1213;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;200;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/videoio/CMakeLists.txt;0;")
