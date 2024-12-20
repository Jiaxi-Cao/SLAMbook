#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
//这段代码用于拟合二次指数函数y=ar * x * x + br * x + cr
int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值，将通过优化逐步逼近真实值
  int N = 100;                                 // 数据点个数
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;            // 噪声标准差，求其倒数，用于归一化
  cv::RNG rng;                                 // OpenCV随机数产生器，生成噪声
  
  //使用真实参数+随机数生成100个带噪声的模拟数据点
  vector<double> x_data, y_data;      // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  // 开始Gauss-Newton迭代
  int iterations = 100;    // 迭代次数100
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {

    Matrix3d H = Matrix3d::Zero();             // 初始化海森矩阵H = J^T * J in Gauss-Newton
    Vector3d b = Vector3d::Zero();             // 初始化梯度向量b
    cost = 0;

    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // 第i个数据点取x，y
      double error = yi - exp(ae * xi * xi + be * xi + ce);//计算当前点的误差
      Vector3d J; // 雅可比矩阵是一个3*1矩阵，是误差函数e（x）对a，b，c的一阶偏导数
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
      J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

      H += inv_sigma * inv_sigma * J * J.transpose();//更新近似的 Hessian 矩阵，3x3 矩阵
      b += -inv_sigma * inv_sigma * error * J;//更新梯度向量b

      cost += error * error;//累积误差平方和
    }

    // 求解线性方程 Hx=b得到dx，dx表示从当前位置如何更新以更接近最低点
    Vector3d dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {//如果当前代价 cost 大于等于上一次迭代的代价 lastCost，退出，因为已经无法继续降低代价函数
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }

    ae += dx[0];//更新此时的三个估计值
    be += dx[1];
    ce += dx[2];

    lastCost = cost;//输出此时的代价函数

    cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}
