#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
  q1.normalize();
  q2.normalize();//定义两个四元数 q1 和 q2，表示坐标系1和坐标系2的旋转信息
  Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);//t1 和 t2：平移向量，分别表示坐标系1和坐标系2的平移
  Vector3d p1(0.5, 0, 0.2);//给定一个在坐标系1中的点P1(0.5，0，0.2）

  Isometry3d T1w(q1), T2w(q2);//表示刚体变换，包含旋转和平移
  T1w.pretranslate(t1);//从 世界坐标系 到 坐标系1 的变换
  T2w.pretranslate(t2);//从 世界坐标系 到 坐标系2 的变换

  Vector3d p2 = T2w * T1w.inverse() * p1;//P1先从坐标系1转到世界坐标系，然后从世界坐标系转到坐标系2
  cout << endl << p2.transpose() << endl;
  return 0;
}
