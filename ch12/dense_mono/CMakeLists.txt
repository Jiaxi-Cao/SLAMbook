cmake_minimum_required(VERSION 3.1)
project(dense_monocular)

# 设置编译类型和编译选项
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17) # 使用C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 第三方库
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})

# 添加可执行文件并链接库
add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${THIRD_PARTY_LIBS})
