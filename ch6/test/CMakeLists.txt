cmake_minimum_required(VERSION 3.10)

# 设置项目和语言支持
project(TestAbseil CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Threads
find_package(Threads REQUIRED)

# 查找 Abseil
find_package(absl REQUIRED)

# 输出信息
message(STATUS "Threads found: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Abseil version: ${absl_VERSION}")

