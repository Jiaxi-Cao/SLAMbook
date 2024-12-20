#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// Algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * Linear triangulation with SVD
 * @param poses     Poses of cameras in world coordinates
 * @param points    Points in normalized image plane
 * @param pt_world  Output: triangulated point in world coordinates
 * @return true if success
 */
inline bool triangulation(const std::vector<Sophus::SE3<double>>& poses,
                          const std::vector<Eigen::Vector3d>& points, // 使用 Eigen::Vector2d 统一点类型
                          Eigen::Vector3d& pt_world) { // 输出点改为 Eigen::Vector3d，避免混淆
    // Construct the A matrix for triangulation
    Eigen::MatrixXd A(2 * poses.size(), 4); // 动态大小矩阵，适配任意数量的姿态
    Eigen::VectorXd b(2 * poses.size());
    b.setZero();

    for (size_t i = 0; i < poses.size(); ++i) {
        // Convert SE3d to 3x4 projection matrix
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0); // 第一行约束
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1); // 第二行约束
    }

    // Perform SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector4d triangulated_point_h = svd.matrixV().col(3); // Homogeneous coordinates

    // Normalize homogeneous coordinates to get 3D point
    if (triangulated_point_h[3] != 0) { // 防止除零
        pt_world = triangulated_point_h.head<3>() / triangulated_point_h[3];
    } else {
        return false; // 返回失败
    }

    // Check the quality of the solution using singular values
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true; // Solution is reliable
    }
    return false; // Solution is not reliable
}

/**
 * Convert cv::Point2f to Eigen::Vector2d
 * @param p OpenCV point
 * @return Eigen vector
 */
inline Eigen::Vector2d toVec2(const cv::Point2f& p) {
    return Eigen::Vector2d(p.x, p.y);
}

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
