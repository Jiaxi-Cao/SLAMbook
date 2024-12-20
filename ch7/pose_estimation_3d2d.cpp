#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>


#include <memory>

using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

int main(int argc, char **argv) {
  if (argc != 5) {//程序名、两张图像、两张深度图
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");//同理，确保加载图像成功

  vector<KeyPoint> keypoints_1, keypoints_2;//两个向量存储两幅图像中的特征点
  vector<DMatch> matches;//存储特征点之间的匹配关系
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);//使用 ORB 特征检测器提取特征点并进行匹配
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 基于匹配点和深度图建立 3D 点
  Mat d1 = imread(argv[3], IMREAD_UNCHANGED); // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//内参矩阵K，含焦距、主点等
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;//定义两个容器，分别存储从深度图中提取的两个图像中匹配点的3D和2D坐标，后续进行3D-2D位姿估计
  for (DMatch m:matches) {//遍历所有的匹配点 matches，根据图像1中的特征点坐标从深度图 d1 获取深度值 d
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth，跳过该点
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//将 2D 像素坐标转换为相机坐标系中的归一化坐标
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));//通过深度值将归一化坐标扩展为三维坐标，并将这个三维点添加到容器的末尾
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);//将该二维点的坐标添加到 pts_2d 容器的末尾
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;//输出成功计算出的 3D-2D 点对数量

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为旋转矩阵，r，t->R
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }//将 OpenCV 的 Point3f 和 Point2f 转换为 Eigen 库中的 Vector3d 和 Vector2d 类型，传递给后续的优化算法

  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;//Sophus::SE3d 类型的变量 pose_gn，用于存储相机位姿
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);//调用BA进行优化
  //pts_3d_eigen：3D点集合、pts_2d_eigen：2D投影点集合、K：相机内参、pose_gn：优化结果（相机位姿）
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);//同上
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
  return 0;
}

//常见的特征对应函数结构，记
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {//同前，将图像的像素坐标转换为相机归一化坐标
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;//定义 Vector6d：6维向量，用于表示优化变量（位姿的李代数表示）
  const int iterations = 10;// 迭代次数
  double cost = 0, lastCost = 0;// 当前误差和上一次误差
  double fx = K.at<double>(0, 0);// 相机内参：焦距 fx
  double fy = K.at<double>(1, 1);// 相机内参：焦距 fy
  double cx = K.at<double>(0, 2);// 相机内参：主点横坐标 cx
  double cy = K.at<double>(1, 2);// 相机内参：主点纵坐标 cy

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();//定义海森矩阵 H 和梯度向量 b，初始化为零。海森矩阵 H：表示误差二阶导数的近似，梯度向量 b：表示误差一阶导数

    cost = 0;//cost 用于记录当前迭代的总误差
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];//将3D点points_3d[i]投影到当前位姿下的相机坐标系，pose为位姿变换R，t
      double inv_z = 1.0 / pc[2];// 深度的倒数，2是数组下标，表示三维向量中的第3个元素，即点 pc 在 z 轴方向上的坐标值
      double inv_z2 = inv_z * inv_z; // 深度倒数的平方
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy); 
      //先归一化相机平面坐标（单位深度平面、焦平面），再投影到图像平面（像素坐标系），从三维变为二维

      Eigen::Vector2d e = points_2d[i] - proj;// 计算误差e（实际2D点与投影点之间的偏差）

      cost += e.squaredNorm();// 计算误差的平方和
      Eigen::Matrix<double, 2, 6> J; //雅可比矩阵是一个2x6矩阵，其中每一行对应一个坐标（x或y），每一列对应一个位姿参数（平移或旋转）
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;//J矩阵的元素表示误差对位姿参数的偏导数

      H += J.transpose() * J;// 更新海森矩阵
      b += -J.transpose() * e;// 更新梯度向量
    }

    Vector6d dx;//使用 H 和 b 构建H*dx=b线性方程。LLT分解求解线性方程，得到更新量 dx，表示位姿参数的增量
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {// 如果解中有 NaN（not a number)，则说明计算出现问题
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {// 如果误差没有减少，停止迭代
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // 更新位姿
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;// 更新上一轮的误差

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;//打印当前迭代的误差值
    if (dx.norm() < 1e-6) { // 如果更新量足够小，说明已经收敛
      // converge
      break;
    }
  }

  cout << "pose by g-n: \n" << pose.matrix() << endl;//输出最终的位姿矩阵
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {//优化变量是 6 维位姿（李代数表示）
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//Eigen 数据结构需要特定的内存对齐，这个宏用于保证对象的内存对齐

  virtual void setToOriginImpl() override {//实现将顶点的估计值重置为原点
    _estimate = Sophus::SE3d();//将位姿重置为单位变换（即无旋转、无平移）
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {//输入update是一个长度为6的数组，表示优化变量的增量，前三个旋转so3，后三个平移
    Eigen::Matrix<double, 6, 1> update_eigen;//update_eigen 是一个 6 维向量，用于存储输入的增量
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];//update转换为Eigen格式的6维向量方便后续计算
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;//将李代数的增量（6 维）转换为李群（SE(3)，4x4 的位姿矩阵）
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}//用于读写顶点的数据
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {//误差的维度为 2
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//保证 Eigen 数据结构的内存对齐

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}//3D 点的世界坐标

  virtual void computeError() override {//计算误差error
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();//返回当前顶点的估计位姿（Sophus::SE3d）
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);//使用位姿T将3D点_pos3d转换到相机坐标系再用内参K将相机坐标系中的点投影到像素平面
    pos_pixel /= pos_pixel[2];//将投影点归一化（除以 Z 坐标）
    _error = _measurement - pos_pixel.head<2>();//误差=实际的 2D 点（观测值）-计算得到的投影点（估计值）
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);//fx, fy, cx, cy 分别是焦距和主点坐标
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];//pos_cam 是 3D 点经过位姿变换后在相机坐标系中的位置
    double Z2 = Z * Z;
    _jacobianOplusXi//雅可比矩阵是误差对位姿的导数
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

private:
  Eigen::Vector3d _pos3d;// 3D 点的世界坐标
  Eigen::Matrix3d _K;// 相机内参
};

void bundleAdjustmentG2O(
   const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {

    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // 处理6维位姿（优化变量）和3维特征点（误差项）
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性方程组求解器

    // 使用裸指针分配LinearSolver
    auto *linearSolver = new LinearSolverType();

    // 使用裸指针分配BlockSolver
    auto *blockSolver = new BlockSolverType(linearSolver);

    // 使用高斯-牛顿法作为优化方法
    auto *solver = new g2o::OptimizationAlgorithmGaussNewton(blockSolver);

    // 定义图优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver); // 设置优化方法
    optimizer.setVerbose(true);     // 打开调试输出

  // 顶点添加
  VertexPose *vertex_pose = new VertexPose(); // 定义一个相机位姿顶点
  vertex_pose->setId(0);// 设置顶点的 ID
  vertex_pose->setEstimate(Sophus::SE3d());// 设置初始位姿（单位变换，无平移旋转）
  optimizer.addVertex(vertex_pose); // 将顶点加入优化器

  // 将 OpenCV 的 Mat 类型的相机内参矩阵 K 转换为 Eigen 的 Matrix3d，便于后续计算
  Eigen::Matrix3d K_eigen;//等价于Eigen::Matrix<double, 3, 3>，在eigen库中定义一个双精度浮点类型的3X3矩阵
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // 边的添加（顶点和边构成图模型，顶点表示优化变量，边表示约束关系、误差项P128）
  int index = 1;//用于给边分配ID
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];// 图像平面上的观测点（2D 像素坐标）
    auto p3d = points_3d[i];// 世界坐标系或相机坐标系中的三维点
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);//遍历每一对2D-3D数据点，为每对数据构建一个误差边
    edge->setId(index);// 设置边的 ID
    edge->setVertex(0, vertex_pose);// 连接相机位姿顶点
    edge->setMeasurement(p2d);// 设置实际观测值（2D 点）
    edge->setInformation(Eigen::Matrix2d::Identity());//信息矩阵表示权重，使用单位矩阵意味着所有误差的权重相同
    optimizer.addEdge(edge);// 将边加入优化器
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);  //启用优化器的详细输出模式如当前误差、增量
  optimizer.initializeOptimization();// 初始化优化
  optimizer.optimize(10); // 进行10次迭代
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;//从顶点 vertex_pose 中获取优化后的位姿
  pose = vertex_pose->estimate();
}
