cmake_minimum_required(VERSION 3.10)  # 升级 CMake 版本
project(useEigen)

# 设置编译类型和 C++ 标准
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 Eigen 头文件路径
find_package(Eigen3 REQUIRED)  # 自动查找 Eigen3
include_directories(${EIGEN3_INCLUDE_DIR})  # 使用找到的路径

# 添加可执行文件
add_executable(eigenMatrix eigenMatrix.cpp)

# 链接 Eigen 库（如需要）
target_link_libraries(eigenMatrix Eigen3::Eigen)