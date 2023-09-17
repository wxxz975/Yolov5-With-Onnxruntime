cmake_minimum_required(VERSION 3.5) # CMake install : https://cmake.org/download/
project(OnnxDetector LANGUAGES CXX)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# onnxruntime path
#set(ONNXRUNTIME_HOME "/usr/local")
set(ONNXRUNTIME_HOME "/usr/local")
include_directories(${ONNXRUNTIME_HOME}/include)
link_directories(${ONNXRUNTIME_HOME}/lib)


# workspace
#include_directories(${CMAKE_SOURCE_DIR})
file(GLOB_RECURSE SRC_LIST "*.cpp")	#遍历获取source/路径下所有的*.cpp和*.c文件列表
file(GLOB_RECURSE HDR_LIST "*.h" "*hpp")

message("src List:${SRC_LIST}")


add_executable(${PROJECT_NAME}
    ${SRC_LIST}
    ${HDR_LIST} 
) 

target_link_libraries(
    ${PROJECT_NAME}
    onnxruntime
    ${OpenCV_LIBS}
)

