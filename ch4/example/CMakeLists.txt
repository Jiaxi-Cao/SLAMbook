cmake_minimum_required(VERSION 3.10)
project(trajectoryError)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加选项，适配 Ubuntu 20.04
option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" OFF)

# 查找 Pangolin 库
find_package(Pangolin REQUIRED)

# 查找 fmt 库（仅在 Ubuntu 20.04 上启用）
if(USE_UBUNTU_20)
    message(STATUS "You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)  # 检查你的 fmt 名称是否正确，如 fmt::fmt-header-only
else()
    set(FMT_LIBRARIES "")  # 如果不使用 fmt，留空
endif()

# 添加可执行文件
add_executable(trajectoryError trajectoryError.cpp)

# 链接库文件
target_include_directories(trajectoryError PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(trajectoryError PRIVATE ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})
