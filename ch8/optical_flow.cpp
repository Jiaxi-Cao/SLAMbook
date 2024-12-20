//
// Created by Xiang on 2017/12/19.
//
//光流的目标是估计两帧图像之间每个像素点的运动矢量，描述物体或场景中像素的位移。

#include <opencv2/opencv.hpp>
#include <string>//用于字符串操作
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "./LK1.png";  
string file_2 = "./LK2.png";  // 定义了两张图片的路径

/// 光流跟踪器和接口
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,//输入两张图片
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,//两张图片的关键点
        vector<bool> &success_,//每个关键点是否跟踪成功的标记
        bool inverse_ = true, bool has_initial_ = false) ://inverse: 是否使用反向公式
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}//has_initial: 是否提供初始值

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    //边界检查
    if (x < 0) x = 0;
    if (y < 0) y = 0;//若 x 或 y 小于 0，修正为 0
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;//若 x 或 y 超出图像范围，修正为图像的最大有效坐标
    
    //计算相邻点坐标
    float xx = x - floor(x);
    float yy = y - floor(y);//floor是整数部分，xx和yy部分是小数部分，小数部分用于计算插值权重
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);//int(x)和int(y)是浮点坐标向下取整的结果，表示该点所在像素的左上角坐标
    //x_a1和y_a1是右下角的相邻像素坐标，用于插值。 std::min 确保右下角坐标不超过图像边界
    
    //双线性插值的核心思想是利用目标点周围 4 个像素的值，根据目标点与这 4 个点的距离，按比例计算加权平均值。
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)//目标点距离左上角点的权重（离得越近，权重值越大）
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)//目标点距离右上角点的权重
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)//目标点距离左下角点的权重
    + xx * yy * img.at<uchar>(y_a1, x_a1);//目标点距离右下角点的权重
    //img.at<uchar>(...) 获取相应像素点的灰度值，实现双线性插值
}

int main(int argc, char **argv) {

    // 读取两张灰度图
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // 使用 GFTT 算法提取图像 img1 的关键点
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    //使用单层光流跟踪，得到第二张图像中的关键点 kp2_single
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // 测试多层光流跟踪，通过图像金字塔进行多层光流计算
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // 使用 OpenCV 的 calcOpticalFlowPyrLK 函数进行光流计算，作为对比
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // 转换图像为 BGR 颜色，用于绘制结果（cv::circle 和 cv::line 可以绘制彩色线条）
    // 其实就是：遍历kp2_single[i]的点，如果跟踪成功， 使用cv::circle在第二帧彩色图像上绘制一个绿色圆圈，表示成功跟踪的关键点位置
    // 然后使用 cv::line 连接第一帧和第二帧对应的关键点以显示它们的匹配关系
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);//这里使用cv::cvtColor将输入的灰度图像img2转换为彩色图像
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {//布尔数组，标记跟踪是否成功（true 表示跟踪成功，false 表示失败）
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
            //kp1[i]第一帧的关键点，kp2_single[i]第二帧单一关键点跟踪的结果，
        }
    }

    Mat img2_multi;//同上，区别在于用的是多关键点跟踪kp2_multi
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    //同上，区别在于光流法（pt2），前两种的关键点类型为cv::KeyPoint，包含更多信息（如尺度、方向等），第三种是二维坐标cv::Point2f
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);//生成图片，等待命令

    return 0;
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // 范围
    int half_patch_size = 4;//使用 8x8 的窗口计算光流
    int iterations = 10;//迭代 10 次寻找光流解
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; 
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;//如果已有初始估计则 dx, dy 设置为关键点在第二帧的初始位移
            dy = kp2[i].pt.y - kp.pt.y;//如果没有初始估计，dx, dy 初始化为 0
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // 通过高斯牛顿迭代求解光流
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // Hessian 矩阵
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // 误差向量
        Eigen::Vector2d J;  // 雅可比矩阵
        for (int iter = 0; iter < iterations; iter++) {//每个关键点的光流计算迭代最多 10 次
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            //误差和雅可比计算
            //雅可比矩阵与当前估计的位移 (dx, dy) 有关，因此每次更新 (dx, dy) 后，梯度需要重新计算，但这种做法计算量大。
            //逆向模式中，雅可比矩阵只在第一次计算，并假设梯度在后续迭代中保持不变。能够显著加速计算，同时不显著影响结果的精度。
            //逆向模式因其高效性，通常是默认选择，普通模式则适用于较大运动估计或初始匹配不准确的情况
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;//通过灰度差计算当前位移(dx,dy)下的灰度误差
                    if (inverse == false) {//梯度计算分为两种模式：false为普通模式：重新计算图像梯度，求解雅可比矩阵
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),//x 方向梯度用右像素的灰度值减去左像素的灰度值
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))//y 方向梯度用下像素的灰度值减去上像素的灰度值
                        );
                    } else if (iter == 0) {
                        //逆向模式 (inverse == true)，雅可比矩阵不随 (dx, dy) 更新，只在第一轮迭代时计算一次
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))//计算同上，但是不考虑位移
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;//b：累积雅可比矩阵与误差的乘积
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // H：累积雅可比矩阵的外积（仅在非逆向模式或第一次迭代时更新）
                        H += J * J.transpose();
                    }
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);//通过求解线性方程 H * update = b，计算位移更新量

            if (std::isnan(update[0])) {
                // 如果更新量为 NaN（通常是因为 Hessian 不可逆），则标记失败
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {//如果位移更新量很小（小于 1e-2），说明误差已收敛，终止迭代
                break;
            }
        }

        success[i] = succ;// 保存是否成功标志

        // 更新第二帧关键点位置
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

//多层金字塔光流算法
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {

    // parameters
    int pyramids = 4;//金字塔4层
    double pyramid_scale = 0.5;//宽度和高度都乘以 0.5，面积变为1/4，分辨率变为1/2
    double scales[] = {1.0, 0.5, 0.25, 0.125};//分辨率变化

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // // 分别保存 img1 和 img2 的金字塔层
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);//金字塔的第 0 层直接保存原始图像 img1 和 img2
        } else {
            Mat img1_pyr, img2_pyr;
            //使用cv::resize缩小图像，src是输入图像，dst是缩放后的图像，后两个是缩放后的宽度长度（按yramid_scale缩小）
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);//将缩小后的图像存入金字塔
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine 光流跟踪方法
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);//kp2_pyr：存储第二帧图像金字塔的关键点位置，初始值为 kp1_pyr，将通过逐层迭代更新得到光流跟踪结果
    }

    for (int level = pyramids - 1; level >= 0; level--) {//从分辨率最底的层开始倒序遍历到分辨率最高的层
        // 逐层迭代优化：先在低分辨率层估计关键点的初步运动位移，然后在更高分辨率层对位移进行精细优化
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        //对当前金字塔层的图像进行光流计算，输入当前层的关键点位置，输出关键点的位移结果、跟踪状态、是否适用逆向光流
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        //将当前层的关键点的位置映射到分辨率更高的上一层
        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;//关键点的位置需要除以 0.5 才能映射到上一层
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);//第一帧关键点（未改变），第二帧关键点（更新后的位移结果）
}
