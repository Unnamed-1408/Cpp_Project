cmake_minimum_required(VERSION 3.5)

project(CNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CNN main.cpp CNN.cpp Test.cpp face_binary_cls.cpp Fast_CNN.cpp)
