#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// 相机内参，分别是焦距和光心坐标
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 双目相机之间的物理距离（米），用于计算深度
double baseline = 0.573;
// 左目图像、视差图像和其他图像的路径
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;//用于 Hessian 矩阵
typedef Eigen::Matrix<double, 2, 6> Matrix26d;//用于雅可比矩阵
typedef Eigen::Matrix<double, 6, 1> Vector6d;//用于位移和旋转增量

/// 辅助类，用于并行化累积雅可比矩阵、Hessian 矩阵，以及计算误差
class JacobianAccumulator {
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,//第一帧中的像素坐标
        const vector<double> depth_ref_,//每个像素点的深度值
        Sophus::SE3d &T21_) ://第一帧到第二帧的初始位姿变换
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));//用于保存投影到第二帧的像素坐标，初始化为零
    }

    /// 累积雅可比矩阵和 Hessian 矩阵的函数
    void accumulate_jacobian(const cv::Range &range);

    /// 返回 Hessian 矩阵 
    Matrix6d hessian() const { return H; }

    /// 返回偏置向量
    Vector6d bias() const { return b; }

    /// 返回总误差（代价函数值）
    double cost_func() const { return cost; }

    /// 返回投影到第二帧的像素坐标
    VecVector2d projected_points() const { return projection; }

    /// 重置 Hessian 矩阵、偏置向量和误差值为零
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;//使用 mutex 确保多线程计算时对 Hessian 和偏置向量的安全更新
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

// 双线性插值，同光流（从浮点坐标 (x,y) 获取插值后的像素值，返回结果是一个加权平均值）
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);//参数 0 表示以灰度图的形式读取左目图像
    cv::Mat disparity_img = cv::imread(disparity_file, 0);//同理，以灰度图的形式读取视差图像

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;//OpenCV 的随机数生成器，用于生成随机坐标
    int nPoints = 2000;//随机选择 2000 个像素点
    int boarder = 20;//边界大小，防止选择到图像边缘的像素点
    VecVector2d pixels_ref;//存储参考帧中选取的像素点坐标
    vector<double> depth_ref;//存储对应像素点的深度值

    // 随机选取图像中的像素点（x，y），然后通过视差图得到像素的视差值，根据视差值计算这些像素的深度值z
    for (int i = 0; i < nPoints; i++) {//随机选取 2000 个像素点
        int x = rng.uniform(boarder, left_img.cols - boarder);  // 在图像宽度范围内随机选取
        int y = rng.uniform(boarder, left_img.rows - boarder);  // 在图像高度范围内随机选取，boarder=20
        int disparity = disparity_img.at<uchar>(y, x);//从视差图中获取对应像素的视差值
        double depth = fx * baseline / disparity; // 根据视差计算深度：z=水平焦距*基线长度/像素点视差
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));//将随机选取的像素点坐标 (x,y) 和对应的深度值存储
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;//当前帧到参考帧的位姿变换

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);//调用直接法位姿估计
    }
    return 0;
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {//参考帧到当前帧的位姿（待优化）

    const int iterations = 10;//最大迭代次数=10
    double cost = 0, lastCost = 0;//当前迭代的总误差，以及上一次迭代的误差，用于判断优化是否收敛或发散
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);//创建类

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();//每次迭代前重置雅可比累加器，清除上一轮的 Hessian 矩阵、偏置向量和误差累计
        cv::parallel_for_(cv::Range(0, px_ref.size()),//OpenCV 的多线程工具，并行化像素点的误差和雅可比计算
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
                          //累积每个像素点的雅可比矩阵、Hessian 矩阵和误差，处理所有像素点
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();//提取 Hessian 矩阵和偏置向量

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;//解线性方程H deltaX=b得位姿更新量update
        T21 = Sophus::SE3d::exp(update) * T21;//新的位姿通过左乘T21的方式更新
        cost = jaco_accu.cost_func();//提取本轮迭代的总误差 cost

        if (std::isnan(update[0])) {
            // 如果 update 中的值为 nan，可能是由于 Hessian 矩阵不可逆（例如误差区域无亮度梯度），终止优化
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            //如果误差 cost 比上一轮的误差 lastCost 更大，说明优化方向可能错误，终止优化
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // 如果参数增量 update 的范数小于阈值，说明优化已收敛，终止迭代
            break;
        }

        lastCost = cost;//将当前误差存入 lastCost，为下一轮迭代使用
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;//输出优化后的位姿矩阵T21
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // 将当前帧中投影的像素点和参考帧中的像素点用图像可视化
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);//当前帧的灰度图像转换为三通道彩色图像img2_show
    VecVector2d projection = jaco_accu.projected_points();//获取计算得到的投影点
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];//参考帧中的像素点坐标
        auto p_cur = projection[i];//当前帧中的像素点坐标
        if (p_cur[0] > 0 && p_cur[1] > 0) {//只有当投影点在图像的有效范围内时，才进行后续的绘制操作
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            //绘制绿色圆点，表示当前帧中的投影点，半径2，线宽2
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));//绘制绿色线段，连接参考帧中的像素点和当前帧中的投影点
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // parameters
    const int half_patch_size = 1;//定义像素邻域的半径，表示3×3 的小窗口
    int cnt_good = 0;//用于统计当前有效的像素点数量
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;//局部代价函数值，用于存储误差的平方和

    for (size_t i = range.start; i < range.end; i++) {

        // 将参考帧的像素坐标（x，y），转换为归一化相机坐标，再乘以深度z得到参考帧中三维点的相机坐标
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;//将参考帧中三维点的相机坐标通过位姿变换T21变换到到当前帧
        if (point_cur[2] < 0)   //检查深度值z否小于0。如果深度为负，表示无效点，跳过
            continue;

        //使用投影公式将当前帧的三维点转换到当前帧的像素坐标（u，v）
        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)//检查（u，v）是否在图像有效范围内，如果超出边界则跳过
            continue;

        projection[i] = Eigen::Vector2d(u, v);//更新projection[i] 为（u,v）
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;//提取三维点的当前帧坐标
        cnt_good++;//有效点计数加 1

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);//计算参考帧和当前帧对应像素的亮度差
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;//像素坐标对位姿的雅可比

                J_img_pixel = Eigen::Vector2d(//图像梯度
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // 总雅可比矩阵
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();//累积 Hessian 矩阵
                bias += -error * J;//累积偏置
                cost_tmp += error * error;//累积代价
            }
    }

    if (cnt_good) {//如果有有效点，使用互斥锁保护H、b 和 cost 的更新
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 同光流，金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // 根据当前金字塔层的比例调整相机的焦距和光心位置
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
