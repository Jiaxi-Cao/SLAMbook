#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"
#include <memory>
#include <opencv2/core/core.hpp>

namespace myslam {

/**
 * 配置类，使用 SetParameterFile 确定配置文件
 * 然后用 Get 得到对应值
 * 单例模式
 */
class Config {
   public:
    // 静态方法，用于设置配置文件路径
    static bool SetParameterFile(const std::string &filename);

    // 静态方法，获取配置参数的值
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }

    // 析构函数，关闭配置文件
    ~Config();

   private:
    // 私有构造函数，禁止外部实例化
    Config() = default;

    // 静态方法，用于生成实例
    static std::shared_ptr<Config> CreateInstance();

    static std::shared_ptr<Config> config_;  // 静态单例指针
    cv::FileStorage file_;                   // OpenCV 配置文件存储
};

}  // namespace myslam

#endif  // MYSLAM_CONFIG_H