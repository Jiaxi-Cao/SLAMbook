//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "visual_odometry.h"

// 使用实际路径
DEFINE_string(config_file, "/home/jiaxi/slambook/ch13/SLAM-Chap-13/config/default.yaml", "config file path");

int main(int argc, char **argv) {
    // 初始化 Google Flags
    google::ParseCommandLineFlags(&argc, &argv, true);

    // 输出加载的配置文件路径，便于调试
    std::cout << "Using configuration file: " << FLAGS_config_file << std::endl;

    // 创建视觉里程计对象并初始化
    myslam::VisualOdometry::Ptr vo = std::make_shared<myslam::VisualOdometry>(FLAGS_config_file);
    if (!vo->Init()) {
        std::cerr << "VisualOdometry initialization failed!" << std::endl;
        return -1;  // 返回错误码
    }

    // 启动视觉里程计
    vo->Run();

    return 0;
}