#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>//用于矩阵运算
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>//g2o用于非线性优化（图优化）
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>//用于李群与李代数计算（SE3）

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

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;//pose->estimate()返回位姿的当前估计值，将 _point 通过当前位姿进行变换
    //计算边的误差值，误差定义为：e=z−h(x)，z为目标点实际测量值，h(x)为投影点=估计的位姿*固定点
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();//同理，当前位姿估计值
    Eigen::Vector3d xyz_trans = T * _point;//同理，xyz_trans 表示将 point通过T变换后的值
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();//平移部分的导数为负单位矩阵 
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);//hat 运算将向量转换为反对称矩阵
  }

  bool read(istream &in) {}

  bool write(ostream &out) const {}

protected:
  Eigen::Vector3d _point;//存储固定的 3D 点坐标，与误差和雅可比计算相关
};

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点，几个的这部分都大差不差，区别是2D、3D，记
  // 3D-3D 点对的位姿估计（如 SVD 或 ICP 方法），通过匹配的两组3D点，直接计算旋转矩阵R和平移向量t
  // 3D-2D 位姿估计（如 PnP 方法），计算相机的外参 [R,t]
  Mat depth1 = imread(argv[3], IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像（深度图）
  Mat depth2 = imread(argv[4], IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像，同上
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//同理，相机内参
  vector<Point3f> pts1, pts2;//定义两个容器，分别存储从深度图中提取的两个图像中匹配点的 3D 坐标，格式为 (x,y,z)

  for (DMatch m:matches) {//使用 ptr 方法访问深度图中的像素值（y，x）
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // 如果提取到的深度值为 0，则跳过该匹配点对
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//将特征点的像素坐标转换为相机归一化平面坐标
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;//把深度图的单位从毫米转化为米（1000或5000取决于相机设备）
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));//将计算得到的3D点分别存储到pts1和pts2中
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;//匹配 3D 点对数量
  Mat R, t;
  pose_estimation_3d3d(pts1, pts2, R, t);//调用pose_estimation_3d3d函数估计位姿（旋转矩阵R和平移向量t）
  cout << "ICP via SVD results: " << endl;//基于 SVD（奇异值分解），通过最小化两个点云之间的欧式距离计算旋转和平移
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;//输出旋转矩阵 R、平移向量 t
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;//输出旋转矩阵 R、平移向量 t的逆矩阵（验证与调试）

  cout << "calling bundle adjustment" << endl;
  
  //通过 BA 优化初始估计的R和t，最小化所有 3D 点对的重投影误差，进一步优化位姿
  bundleAdjustment(pts1, pts2, R, t);

  // 取前 5 对匹配点，对p1和p2的估计结果进行验证，查看是否满足p1=R*p2+t
  for (int i = 0; i < 5; i++) {
    cout << "p1 = " << pts1[i] << endl;
    cout << "p2 = " << pts2[i] << endl;
    cout << "(R*p2+t) = " <<
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
         << endl;
    cout << endl;
  }
}

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

Point2d pixel2cam(const Point2d &p, const Mat &K) {//同前，把像素坐标转化到归一画坐标
  return Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}

//基于SVD，估计R与t
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  Point3f p1, p2;     // 质心
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N); 
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }//计算去中心化后的点坐标，消除全局平移的影响，使点云只受旋转的影响，便于计算R

  // 构造协方差矩阵w=q1*q2^T求和
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }//q1和q2来自两组点云。协方差矩阵W表示两组点云之间的相关性，是后续 SVD 分解的基础
  cout << "W=" << W << endl;

  //对协方差矩阵进行SVD分解，W=U*对角矩阵*V^T
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());//计算旋转矩阵R=U*V^T，得R
  if (R_.determinant() < 0) {
    R_ = -R_;//如果R的行列式小于0则反转U的符号使得R行列式为1，避免点云发生反射变换
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
  //公式t=p1-R*p2，得t

  //将R和t结果转化为cv::Mat格式，便于与 OpenCV 相关函数兼容
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
  const vector<Point3f> &pts1,
const vector<Point3f> &pts2,
Mat &R, Mat &t) {
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // SE(3) 优化问题
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 使用稠密线性求解器

    // 使用裸指针初始化LinearSolver
    LinearSolverType *linearSolver = new LinearSolverType();

    // 使用裸指针初始化BlockSolver
    BlockSolverType *blockSolver = new BlockSolverType(linearSolver);

    // 使用Levenberg-Marquardt优化方法
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    // 图模型
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出


  // vertex
  VertexPose *pose = new VertexPose(); // camera pose
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  // edges
  for (size_t i = 0; i < pts1.size(); i++) {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    edge->setVertex(0, pose);
    edge->setMeasurement(Eigen::Vector3d(
      pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}
