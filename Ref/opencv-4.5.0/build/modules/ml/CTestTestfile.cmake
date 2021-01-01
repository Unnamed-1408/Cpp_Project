# CMake generated Testfile for 
# Source directory: /home/bill/下载/opencv-4.5.0/modules/ml
# Build directory: /home/bill/下载/opencv-4.5.0/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "/home/bill/下载/opencv-4.5.0/build/bin/opencv_test_ml" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1074;ocv_add_accuracy_tests;/home/bill/下载/opencv-4.5.0/modules/ml/CMakeLists.txt;2;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/ml/CMakeLists.txt;0;")
