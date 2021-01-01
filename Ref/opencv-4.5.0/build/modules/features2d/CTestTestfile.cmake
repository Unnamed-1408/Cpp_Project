# CMake generated Testfile for 
# Source directory: /home/bill/下载/opencv-4.5.0/modules/features2d
# Build directory: /home/bill/下载/opencv-4.5.0/build/modules/features2d
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_features2d "/home/bill/下载/opencv-4.5.0/build/bin/opencv_test_features2d" "--gtest_output=xml:opencv_test_features2d.xml")
set_tests_properties(opencv_test_features2d PROPERTIES  LABELS "Main;opencv_features2d;Accuracy" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1074;ocv_add_accuracy_tests;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;9;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;0;")
add_test(opencv_perf_features2d "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_features2d" "--gtest_output=xml:opencv_perf_features2d.xml")
set_tests_properties(opencv_perf_features2d PROPERTIES  LABELS "Main;opencv_features2d;Performance" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/performance" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1212;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1075;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;9;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;0;")
add_test(opencv_sanity_features2d "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_features2d" "--gtest_output=xml:opencv_perf_features2d.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_features2d PROPERTIES  LABELS "Main;opencv_features2d;Sanity" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1213;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1075;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;9;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/features2d/CMakeLists.txt;0;")
