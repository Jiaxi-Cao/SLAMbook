#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam {

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string& dataset_path);

    /// 初始化，返回是否成功
    bool Init();

    /// 读取数据集中的下一帧图像，并返回一个指向 Frame 对象的智能指针
    Frame::Ptr NextFrame();

    /// 通过传入相机ID返回对应的相机对象
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

   private:
    std::string dataset_path_;//用于存储数据集的路径
    int current_image_index_ = 0;//显示当前正在读取的数据集中的图像的索引位置

    std::vector<Camera::Ptr> cameras_;//cameras容器，用于存储多个相机的指针
};
}  // namespace myslam

#endif