cmake_minimum_required(VERSION 3.20)
project(SIFT)

set(CMAKE_CXX_STANDARD 20)

add_library(SIFT SHARED src/utils.cpp src/utils.h src/lib.cpp src/lib.h)

find_package(OpenCV REQUIRED)
add_executable(test test/test.cpp)
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test ${CMAKE_SOURCE_DIR}/cmake-build-release/SIFT.lib)
