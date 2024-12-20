cmake_minimum_required(VERSION 3.1) # 更新为更现代的 CMake 版本
project(loop_closure)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17) # 使用 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求支持 C++17

# OpenCV
# 更改 OpenCV 版本要求
find_package(OpenCV REQUIRED) # 删除版本限制
include_directories(${OpenCV_INCLUDE_DIRS})

# DBoW3
find_package(DBoW3 REQUIRED) # 自动查找 DBoW3
include_directories(${DBoW3_INCLUDE_DIRS}) # 包含 DBoW3 的头文件路径
link_directories(${DBoW3_LIBRARY_DIRS}) # 包含 DBoW3 的库文件路径

# 如果 find_package 找不到，可以手动指定路径：
# set(DBoW3_INCLUDE_DIRS "/usr/local/include")
# set(DBoW3_LIBS "/usr/local/lib/libDBoW3.a")
# include_directories(${DBoW3_INCLUDE_DIRS})

# 添加可执行文件和链接库
add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBRARIES})

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBRARIES})

add_executable(gen_vocab gen_vocab_large.cpp)
target_link_libraries(gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBRARIES})
