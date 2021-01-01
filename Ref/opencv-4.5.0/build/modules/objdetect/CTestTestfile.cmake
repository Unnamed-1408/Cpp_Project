# CMake generated Testfile for 
# Source directory: /home/bill/下载/opencv-4.5.0/modules/objdetect
# Build directory: /home/bill/下载/opencv-4.5.0/build/modules/objdetect
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_objdetect "/home/bill/下载/opencv-4.5.0/build/bin/opencv_test_objdetect" "--gtest_output=xml:opencv_test_objdetect.xml")
set_tests_properties(opencv_test_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Accuracy" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1310;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1074;ocv_add_accuracy_tests;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;2;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;0;")
add_test(opencv_perf_objdetect "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_objdetect" "--gtest_output=xml:opencv_perf_objdetect.xml")
set_tests_properties(opencv_perf_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Performance" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/performance" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1212;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1075;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;2;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;0;")
add_test(opencv_sanity_objdetect "/home/bill/下载/opencv-4.5.0/build/bin/opencv_perf_objdetect" "--gtest_output=xml:opencv_perf_objdetect.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Sanity" WORKING_DIRECTORY "/home/bill/下载/opencv-4.5.0/build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/bill/下载/opencv-4.5.0/cmake/OpenCVUtils.cmake;1640;add_test;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1213;ocv_add_test_from_target;/home/bill/下载/opencv-4.5.0/cmake/OpenCVModule.cmake;1075;ocv_add_perf_tests;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;2;ocv_define_module;/home/bill/下载/opencv-4.5.0/modules/objdetect/CMakeLists.txt;0;")
