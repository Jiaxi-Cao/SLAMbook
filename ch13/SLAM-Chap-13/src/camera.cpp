#include "myslam/camera.h"
#include "myslam/common_include.h"


namespace myslam {

Camera::Camera() {
}

//世界坐标系的点转为相机坐标系
Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w) {
    return pose_ * T_c_w * p_w;//pose_是相机在世界坐标系中的变换矩阵，把点从世界坐标系换到相机坐标系
}

//相机坐标系的点转为世界坐标系
Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w) {
    return T_c_w.inverse() * pose_inv_ * p_c;//pose_inv是相机在世界坐标系中的变换矩阵的逆矩阵，T_c_w.inverse()同理
}

//相机坐标系的三维点转为像素坐标的二维点
Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c) {
    return Eigen::Vector2d(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,//fx_、fy_是相机的焦距
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_//cx_、cy_是相机的光心坐标，即像素坐标系的原点
    );
}

//像素坐标的二维点转为相机坐标系的三维点
Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth) {
    return Eigen::Vector3d(
            (p_p(0, 0) - cx_) * depth / fx_,//公式倒推即可
            (p_p(1, 0) - cy_) * depth / fy_,
            depth//深度即z
    );
}

//世界坐标系的三维点转为像素坐标系的二维点
Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w) {
    return camera2pixel(world2camera(p_w, T_c_w));//把上面两步走一遍就行
}

//像素坐标系的二维点转为世界坐标系的三维点
Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth) {
    return camera2world(pixel2camera(p_p, depth), T_c_w);//同理
}

}
