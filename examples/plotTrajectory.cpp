#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
#include <Eigen/Geometry>
// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// 指定轨迹文件的路径
string trajectory_file = "./examples/trajectory.txt";

//函数声明，负责可视化轨迹
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {

  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;//定义一个容器 poses，用于存储每个位姿
  ifstream fin(trajectory_file);//打开trajectory_file，如果文件不存在，输出错误信息并退出程序。
  if (!fin) {
    cout << "cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }
  
  //循环读取文件中的位姿数据
  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;//tx, ty, tz为平移向量、qx, qy, qz, qw为四元数（表示旋转）
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    poses.push_back(Twr);//将平移和旋转合并为Isometry3d并且存放在poses中
  }
  cout << "read total " << poses.size() << " pose entries" << endl;

  // 将解析好的poses数据给DrawTrajectory进行可视化绘制
  DrawTrajectory(poses);
  return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);//创建一个叫Trajectory Viewer的1024*768的窗口
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//启用 深度测试 和 透明度混合，保证 3D 图形的渲染效果

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),//设置相机的投影矩阵和视图矩阵
    //世界坐标系 -> 视图矩阵 -> 相机坐标系 -> 投影矩阵 -> 像素坐标系
    //1024, 768：窗口的宽度和高度；500, 500：相机的焦距；512, 389：光心坐标（图像中心）；0.1, 1000：近平面和远平面
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//定义了相机的位置、观察点和上方向
    //(0, -0.1, -1.8)：相机位置；(0, 0, 0)：相机观察的目标点（原点）；(0.0, -1.0, 0.0)：相机的上方向（Y 轴向下）
  );
  
  //显示轨迹的范围
  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  //每一帧循环清理屏幕，激活相机视角，并设置背景颜色和线宽
  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {

      // 画每个位姿的三个坐标轴
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);// X 轴为红色
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);// Y 轴为绿色
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);// Z 轴为蓝色
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);// 连线为黑色
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // 完成当前帧的渲染，延迟 5 毫秒，确保流畅的绘制效果
  }
}
