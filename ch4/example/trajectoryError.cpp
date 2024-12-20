#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "./example/groundtruth.txt";
string estimated_file = "./example/estimated.txt";//指定输入文件路径

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);//ReadTrajectory：读取轨迹文件，返回一个 SE3 向量
  TrajectoryType estimated = ReadTrajectory(estimated_file);//同样，读文件，返回向量
  assert(!groundtruth.empty() && !estimated.empty());//确保读取的轨迹不为空
  assert(groundtruth.size() == estimated.size());//确保两条轨迹的长度相等（保证误差计算的正确性）
 
  // 计算 RMSE (均方根误差) 
  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++) {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error = (p2.inverse() * p1).log().norm();
    //误差计算公式解读：p1是估计轨迹的位姿、p2是真值轨迹的位姿，p2^(-1)*p1是将p1映射到p2的坐标系中
    //.log()是将李群对数映射到李代数，。norm()是计算李代数的范数（即为误差大小），累积误差的大小则为均方根误差
    rmse += error * error;
  }
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  DrawTrajectory(groundtruth, estimated);//调用DrawTrajectory函数，在3D 可视化窗口中绘制两条轨迹
  return 0;
}

TrajectoryType ReadTrajectory(const string &path) {//打开文件路径 path
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;//逐行读取 time, tx, ty, tz, qx, qy, qz, qw
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));//使用 四元数 和 平移向量 构造 SE3 位姿 Sophus::SE3d
    trajectory.push_back(p1);//将位姿添加到 trajectory 容器中
  }
  return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // 初始化 Pangolin 窗口
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);//创建一个名为 Trajectory Viewer 的窗口，分辨率为 1024x768
  glEnable(GL_DEPTH_TEST);//启用深度测试，确保 3D 绘图时前后的物体正确显示
  glEnable(GL_BLEND);//启用颜色混合，使图像更加平滑
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),//定义相机投影矩阵，设置视角和观察范围
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()//创建一个渲染区域（显示窗口的大小）
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));//设置交互控制器，可以用鼠标拖拽视角


  while (pangolin::ShouldQuit() == false) {//判断窗口是否被关闭
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);//激活相机视角
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//清除颜色和深度缓冲区，准备开始新一轮绘制

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // 绘制真值轨迹（蓝色）
      glBegin(GL_LINES);//开始绘制线段
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);//p1是i时刻的3d点，用translation()获得轨迹点的平移部分
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);//p2是i+1时刻的3d点，与p1相邻；遍历轨迹中的每两个相邻点，用线段连接起来即绘制完成
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // 绘制估计轨迹（红色）
      glBegin(GL_LINES);//开始绘制线段
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // 延迟 5 毫秒
  }

}
