cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_STANDARD_REQUIRED ON)

project(OpenCVGpuExample)

find_package(OpenCV 3.2 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message( INCLUDE ${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
