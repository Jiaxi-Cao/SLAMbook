cmake_minimum_required(VERSION 3.1) # 将最低版本提高到支持 C++ 标准的现代版本
project(ch12)

# 设置全局编译选项
set(CMAKE_CXX_STANDARD 17) # 使用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -march=native")

# 添加子目录
add_subdirectory(dense_mono)
add_subdirectory(dense_RGBD)
