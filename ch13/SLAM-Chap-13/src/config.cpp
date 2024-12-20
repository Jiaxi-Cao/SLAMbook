#include "myslam/config.h"
#include <glog/logging.h>

namespace myslam {

// 实现单列模式，初始化静态成员变量
std::shared_ptr<Config> Config::config_ = nullptr;//初始值为 nullptr，表示在程序运行之初，实例尚未创建

bool Config::SetParameterFile(const std::string &filename) {//根据文件路径尝试打开文件，返回bool值
    if (config_ == nullptr) {//如果不存在实例
        config_ = CreateInstance();  // 使用工厂方法创建实例
    }
    config_->file_.open(filename, cv::FileStorage::READ);//尝试以只读模式打开配置文件
    if (!config_->file_.isOpened()) {//如果打开失败，则打印错误日志
        LOG(ERROR) << "Cannot open config file: " << filename;
        config_->file_.release();
        return false;
    }
    return true;
}

// 析构函数，对象生命周期结束时才被调用，一种保险，析：意味着解构对象，将其内部的资源逐步释放
Config::~Config() {
    if (file_.isOpened()) {//未能正确关闭文件
        file_.release();//调用release关闭文件释放资源
    }
}

// 静态工厂方法，用于创建实例
std::shared_ptr<Config> Config::CreateInstance() {
    return std::shared_ptr<Config>(new Config());
}

}  // namespace myslam