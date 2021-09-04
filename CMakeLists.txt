cmake_minimum_required(VERSION 3.20)
project(SIFT)

set(CMAKE_CXX_STANDARD 23)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})
add_executable(SIFT main.cpp utils.cpp utils.h)