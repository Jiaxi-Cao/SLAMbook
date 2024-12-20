cmake_minimum_required(VERSION 3.1)

# 项目名称
project(MappingProject)

# 编译类型与选项
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)  # 使用 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -march=native")  # 优化选项

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OctoMap
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

# Boost (确保 Boost 可用)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# 添加可执行文件及其依赖库
add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(octomap_mapping octomap_mapping.cpp)
target_link_libraries(octomap_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(surfel_mapping surfel_mapping.cpp)
target_link_libraries(surfel_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})
