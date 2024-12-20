cmake_minimum_required(VERSION 3.0)
project(vo1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# G2O 路径设置
set(G2O_DIR "/usr/local/lib/cmake/g2o")
find_package(G2O REQUIRED)

# OpenCV 版本设置（支持 OpenCV 4 或 3）
find_package(OpenCV REQUIRED)

# Sophus 路径设置（如果必要）
find_package(Sophus REQUIRED)

# 添加 Sophus 的路径
include_directories(/usr/local/include/sophus)
include_directories(/usr/local/include/eigen3)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    /usr/local/include/eigen3
    /usr/local/include/sophus
)

# 可执行文件与目标链接库
add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
    g2o_core g2o_stuff
    ${OpenCV_LIBS}
)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d
    g2o_core g2o_stuff
    ${OpenCV_LIBS}
)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -msse4.2 -mpopcnt")


