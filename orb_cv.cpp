#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>//提供特征检测和描述子计算的相关类
#include <opencv2/highgui/highgui.hpp>//提供图像显示和 GUI 操作的相关类
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;//如不符合程序名 + 图像1路径 + 图像2路径的命令行参数则输出提示
  }
  //-- 读取图像
Mat img_1 = imread(argv[1], IMREAD_COLOR);//Mat为多维矩阵，存储像素点（BGR三通道）
Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);//不等于空指针即文件可读

  //-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;//创建矩阵存储图像1和图像2的关键点
  Mat descriptors_1, descriptors_2;//创建矩阵存储图像1和图像2的特征描述子
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();//分别创建一个ORB特征检测器、描述子
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//创建一个使用汉明距离的暴力匹配器

  //-- 第一步:检测 Oriented FAST 角点位置ORB
  //（Oriented FAST and Rotated BRIEF）：FAST（Features from Accelerated Segment Test）
  //BRIEF（Binary Robust Independent Elementary Features）
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//记录检测开始的当前时间
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);//使用 ORB 检测器在图像中提取关键点

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);//指针调用compute利用关键点计算特征子
  descriptor->compute(img_2, keypoints_2, descriptors_2);//根据检测到的关键点，计算特征子
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//再次获取当前时间
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//二者之差为时间间隔
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;//计算检测和描述子的时间，并输出

  Mat outimg1;
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);//用随机颜色绘制特征点图像
  imshow("ORB features", outimg1);//显示绘制图像

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);//指针调用匹配两图特征点
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//同理计算时间
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),//遍历的起始和结束迭代器，表示匹配点对的集合，计算两个最值
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;//指向最小值元素的迭代器，配合distance即求最小匹配距离
  double max_dist = min_max.second->distance;//指向最大值元素的迭代器，同理求最大匹配距离

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);//分别绘制所有匹配点和筛选后的匹配点

  return 0;
}
