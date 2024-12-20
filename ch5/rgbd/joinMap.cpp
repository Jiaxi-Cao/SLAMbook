
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>


using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 点云显示函数（showPointCloud)在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 用于存储5幅彩色图和深度图
    TrajectoryType poses;         // 相机位姿

    ifstream fin("./pose.txt");
    if (!fin) {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
    
    //读取彩色图、深度图、相机位姿
    for (int i = 0; i < 5; i++) {
        boost::format fmt("./%s/%d.%s"); //图像文件格式,%s表示一个字符串占位符,%d整数占位符
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));//将占位符换为color和png
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};//定义一个长度为7的 data 数组，用于存储一行位姿数据
        for (auto &d:data)//使用 fin（文件流）逐个读取 data 数组中的7个值tx ty tz qx qy qz qw
            fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),//构造表示旋转的四元数
                          Eigen::Vector3d(data[0], data[1], data[2]));//构造表示平移的向量，然后用SE3d构造完整的刚体变换
        poses.push_back(pose);//位姿 pose 存储到 poses 向量中
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 5; i++) {//遍历五幅图像
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];//从 colorImgs 和 depthImgs 分别读取当前图像的彩色图和深度图
        Sophus::SE3d T = poses[i];//从 poses 中读取对应的 相机位姿

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {//逐行逐列遍历当前图像的每个像素点
                unsigned int d = depth.ptr<unsigned short>(v)[u]; //访问深度图中第v行第u列的深度，保存为unsigned short16位无符号整数类型
                if (d == 0) continue; // 为0表示没有测量到，跳过该点
                Eigen::Vector3d point;//将深度图中的像素点根据相机内参投影到三维相机坐标系
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;//利用相机外参，即位姿T，将相机坐标系中的点转换到世界坐标系

                Vector6d p;
                p.head<3>() = pointWorld;//世界坐标系的XYZ坐标系作为前三个元素，后三个为颜色信息rgb
                p[5] = color.data[v * color.step + u * color.channels()];   // 蓝色通道值
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // 绿色通道值
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // 红色通道值。

                pointcloud.push_back(p);//将点和颜色添加到点云向量中
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);
    return 0;
}

//利用 Pangolin 进行三维点云的可视化展示
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;//如果为空，输出错误信息并返回，避免后续绘制过程出错
    }
    
    //创建窗口Point Cloud Viewer，分辨率为1024*768
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//启用深度测试，颜色混合功能

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//设置投影参数和观察视角
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));//绑定 3D交互控制器，允许用户拖拽、缩放视图

    while (pangolin::ShouldQuit() == false) {//进入窗口的绘制循环，检测窗口是否关闭
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);//激活相机视角，开始绘制
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);//背景设置为白色，(0.0f, 0.0f, 0.0f, 1.0f)则为背景黑，最后一个为不透明度

        glPointSize(2);//设置点的大小为2像素
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);//rgb
            glVertex3d(p[0], p[1], p[2]);//xyz
        }
        glEnd();
        pangolin::FinishFrame();//结束当前帧的绘制并刷新窗口
        usleep(5000);   // 延迟5毫秒，控制绘制速度，避免CPU占用过高
    }
    return;
}
