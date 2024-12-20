#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // 每一帧的ID
    unsigned long keyframe_id_ = 0;  // 关键帧的ID
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    Sophus::SE3d pose_;                       // Tcw 形式Pose
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat left_img_, right_img_;   // 表示左右视图的图像

    // 保存左图像中的特征点，每个特征点是一个 Feature 类型的智能指针
    std::vector<std::shared_ptr<Feature>> features_left_;
    // 保存右图像中的特征点，每个特征点是一个 Feature 类型的智能指针
    std::vector<std::shared_ptr<Feature>> features_right_;

   public:  // data members
    Frame() {}//构造函数，默认构造一个空的 Frame 对象

    Frame(long id, double time_stamp, const Sophus::SE3d &pose, const Mat &left,
          const Mat &right);

    // 获取当前帧的位姿, thread safe, pose可能会被前端和后端的线程同时访问
    Sophus::SE3d Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }
    
    //设置当前帧的位姿
    void SetPose(const Sophus::SE3d &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    /// 工厂构建模式（静态函数），分配id 
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
