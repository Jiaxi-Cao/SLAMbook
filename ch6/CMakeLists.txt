cmake_minimum_required(VERSION 3.10)
project(ch6)

# 设置编译选项
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

# 添加路径到 CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/cmake/absl" "/usr/local/lib/cmake/Ceres")

# 设置可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ===================== 依赖库查找与配置 =====================

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Ceres Solver
set(Ceres_DIR "/usr/local/lib/cmake/Ceres")  # 替换为实际路径
find_package(Ceres QUIET) # 禁用版本检查
if(Ceres_FOUND)
    include_directories(${CERES_INCLUDE_DIRS})
    message(STATUS "Ceres include dirs: ${CERES_INCLUDE_DIRS}")
    message(STATUS "Ceres libraries: ${CERES_LIBRARIES}")
else()
    message(WARNING "Ceres Solver not found or version mismatch! Please check your installation.")
endif()

# Abseil
set(absl_DIR "/usr/local/lib/cmake/absl")  # 替换为实际路径
find_package(absl QUIET)  # 禁用版本检查
if(absl_FOUND)
    message(STATUS "Abseil found at: ${absl_DIR}")
else()
    message(WARNING "Abseil not found or version mismatch! Please check your installation.")
endif()

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen include dirs: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# g2o
# 设置 g2o 的头文件路径和库文件路径
set(G2O_INCLUDE_DIRS "/usr/local/include")  # g2o 的头文件路径
set(G2O_LIBRARIES 
    "/usr/local/lib/libg2o_core.so"
    "/usr/local/lib/libg2o_stuff.so"
    "/usr/local/lib/libg2o_solver_csparse.so"
    "/usr/local/lib/libg2o_csparse_extension.so"
    "/usr/local/lib/libg2o_types_slam2d.so"
    "/usr/local/lib/libg2o_types_slam3d.so"
    "/usr/local/lib/libg2o_solver_dense.so"
    "/usr/local/lib/libg2o_solver_eigen.so"
)

# 确保路径存在并可用
if(EXISTS "${G2O_INCLUDE_DIRS}" AND EXISTS "${G2O_LIBRARIES}")
    include_directories(${G2O_INCLUDE_DIRS})
    message(STATUS "G2O include dirs: ${G2O_INCLUDE_DIRS}")
    message(STATUS "G2O libraries: ${G2O_LIBRARIES}")
else()
    message(FATAL_ERROR "g2o not found at specified locations. Please check your installation.")
endif()

# Threads
find_package(Threads REQUIRED)

# ===================== 可执行文件配置 =====================

# Gauss-Newton 代码
add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

# Ceres Solver 曲线拟合代码
if(Ceres_FOUND AND absl_FOUND)
    add_executable(ceresCurveFitting ceresCurveFitting.cpp)
    target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES} Threads::Threads)
else()
    message(WARNING "Skipping ceresCurveFitting target due to missing dependencies.")
endif()

# g2o 曲线拟合代码
add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_LIBRARIES})

# ===================== 输出配置 =====================
message(STATUS "Configuration completed successfully!")
