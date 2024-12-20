#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pangolin/pangolin.h>

using namespace std;
using namespace Eigen;

struct RotationMatrix {
  Matrix3d matrix = Matrix3d::Identity();//定义旋转矩阵结构，使用 Eigen::Matrix3d 初始化为单位矩阵
};

ostream &operator<<(ostream &out, const RotationMatrix &r) {
  out.setf(ios::fixed);
  Matrix3d matrix = r.matrix;
  out << "=[" << setprecision(2)
      << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
      << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
      << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";
  return out;//输出旋转矩阵 matrix，格式化输出到控制台
}

istream &operator>>(istream &in, RotationMatrix &r) {
  return in;
}

struct TranslationVector {
  Vector3d trans = Vector3d(0, 0, 0);//定义平移向量结构，使用 Eigen::Vector3d 初始化为 (0, 0, 0)
};

ostream &operator<<(ostream &out, const TranslationVector &t) {
  out << "=[" << t.trans(0) << "," << t.trans(1) << "," << t.trans(2) << "]";
  return out;//输出平移向量 trans
}

istream &operator>>(istream &in, TranslationVector &t) {
  return in;
}

struct QuaternionDraw {
  Quaterniond q;//定义四元数结构，使用 Eigen::Quaterniond
};

ostream &operator<<(ostream &out, const QuaternionDraw quat) {
  auto c = quat.q.coeffs();
  out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
  return out;//输出四元数的系数（qx, qy, qz, qw）
}

istream &operator>>(istream &in, const QuaternionDraw quat) {
  return in;
}

// 将 Pangolin 中的 OpenGlMatrix 转换为 Eigen 的 Matrix4d，实现两者的数据兼容。

Eigen::Matrix4d ToEigenMatrix4d(const pangolin::OpenGlMatrix &pangolin_matrix) {
  Eigen::Matrix4d eigen_matrix;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      eigen_matrix(i, j) = pangolin_matrix.m[j * 4 + i];
  return eigen_matrix;
}

int main(int argc, char **argv) {
  pangolin::CreateWindowAndBind("visualize geometry", 1000, 600);
  glEnable(GL_DEPTH_TEST);//创建一个 1000x600 的 OpenGL 窗口并启用深度测试
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
    pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
  );//设置相机的投影矩阵和模型视图矩阵

  const int UI_WIDTH = 500;

  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  // 在 UI 面板上添加旋转矩阵、平移向量、欧拉角和四元数的显示
  pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
  pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
  pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
  pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);//清理屏幕，激活相机，开始绘制新的一帧

    // 提取变换矩阵（即相机的位姿）并分解
    pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();
    Eigen::Matrix4d m = ToEigenMatrix4d(matrix);//从 OpenGL 获取变换矩阵

    // 提取旋转矩阵
    RotationMatrix R;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R.matrix(i, j) = m(j, i);
    rotation_matrix = R;

    // 提取平移向量
    TranslationVector t;
    t.trans = Vector3d(m(0, 3), m(1, 3), m(2, 3));
    t.trans = -R.matrix * t.trans;
    translation_vector = t;

    // 提取欧拉角
    TranslationVector euler;
    euler.trans = R.matrix.eulerAngles(2, 1, 0);
    euler_angles = euler;

    // 提取四元数
    QuaternionDraw quat;
    quat.q = Quaterniond(R.matrix);
    quaternion = quat;

    
    glColor3f(1.0, 1.0, 1.0);
    pangolin::glDrawColouredCube();//彩色立方体

    glLineWidth(3);
    glColor3f(0.8f, 0.f, 0.f);
    glBegin(GL_LINES);// 绘制坐标轴
    glVertex3f(0, 0, 0);
    glVertex3f(10, 0, 0);
    glColor3f(0.f, 0.8f, 0.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 10, 0);
    glColor3f(0.2f, 0.2f, 1.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 10);
    glEnd();

    pangolin::FinishFrame();
  }
}
