#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

std::string current_sequence_name_; // 当前序列名

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    // 打开当前序列的 calib.txt 文件（拼接路径）
    //std::cout << dataset_path_ + "/" + current_sequence_name_ + "/calib.txt" << std::endl;
    //ifstream fin(dataset_path_ + "/" + current_sequence_name_ + "/calib.txt"); // 修改路径

     // 打开固定路径的 calib.txt 文件
    std::cout << dataset_path_ + "/calib.txt" << std::endl;
    ifstream fin(dataset_path_ + "/calib.txt");  // 使用固定路径
    if (!fin) { // 如果文件打不开，打印错误信息并返回 false
        LOG(ERROR) << "cannot find " << dataset_path_ + "/" + current_sequence_name_ + "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];//存储相机的名字（例如 P0、P1）
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];//读取投影矩阵projection_data的元素
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Mat33 K;//存储12个投影矩阵的元素
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];
        // 处理相机内外参数，内参K，外参t
        t = K.inverse() * t;//计算t值
        K = K * 0.5; // 缩小一半的比例
        
        // 读取相机参数后创建相机对象
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), Sophus::SE3d(Sophus::SO3d(), t)));
        cameras_.push_back(new_camera); // 把相机的内外参保存到cameras_中
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    current_image_index_ = 0;
    return true;
}

// 创建新图像帧NextFrame()，保存图像到里面，并更新 data index
Frame::Ptr Dataset::NextFrame() {
    boost::format fmt("%s/%s/image_%d/%06d.png"); // %s 是数据集路径，%s 是序列编号，%d 是相机编号，%06d 是图像编号
    cv::Mat image_left, image_right;
    
    // 读取左右相机的灰度图像
    image_left =
        cv::imread((fmt % dataset_path_ % current_sequence_name_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % current_sequence_name_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr; // 如果无法读取图像，打印警告并返回 nullptr
    }
    
    // 缩小图像为原图一半，为了提高处理速度，减少储存和内存占用
    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    
    // 创建 Frame 并保存缩小后的图像
    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;
    current_image_index_++; // 读取下一帧
    return new_frame;
}

}  // namespace myslam