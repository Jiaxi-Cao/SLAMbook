cmake_minimum_required(VERSION 3.1)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

# 添加 cmake 文件搜索路径
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# 查找依赖库
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)

# 设置 G2O 库
set(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

# 添加头文件路径
include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

# 添加 bal_common 库
add_library(bal_common common.cpp)

# 添加可执行文件 bundle_adjustment_g2o
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)

# 添加可执行文件 bundle_adjustment_ceres
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)

# 链接库
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common)
target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS} bal_common)
