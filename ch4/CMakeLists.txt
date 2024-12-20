cmake_minimum_required(VERSION 3.10)  # 升级 CMake 最低版本要求
project(useSophus)

# 设置 C++17 标准，解决 std::optional 和其他 C++17 功能问题
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 为使用 Sophus，需要使用 find_package 命令找到它
find_package(Sophus REQUIRED)

# Eigen 头文件路径
include_directories("/usr/include/eigen3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加可执行文件
add_executable(useSophus useSophus.cpp)

# 链接 Sophus 库
target_link_libraries(useSophus Sophus::Sophus)

# 加入子目录（如有其他子项目）
add_subdirectory(example)



