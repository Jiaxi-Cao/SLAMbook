//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

//前端主要负责接收帧数据，进行特征检测、跟踪、位姿估计，并决定是否插入关键帧，同时与后端交互
Frontend::Frontend() {
    //初始化特征点检测器（GFTT）和配置参数
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);//num_features是每帧提取的特征点数，0.01是角点响应阈值，20是特征点间的最小距离
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

//添加帧处理，可根据当前状态（初始化、跟踪中、丢失）执行不同命令，能更新当前帧保存为“上一帧”
bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame; // frame里面不会引用frontend的指针，所以直接可以全引用shared_ptr

    switch (status_) {//根据status_进行操作
        case FrontendStatus::INITING://如果是INITING则调用StereoInit() 初始化
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD://是TRACKING_GOOD或TRACKING_BAD则调用Track() 跟踪
            Track();
            break;
        case FrontendStatus::LOST://如果是LOST则调用 Reset() 重置前端
            Reset();
            break;
    }

    last_frame_ = current_frame_;//把“当前帧”保存为“上一帧”
    return true;
}

//跟踪功能：跟踪上一帧，估计当前帧的位姿，并更新跟踪状态
bool Frontend::Track() {
    if (last_frame_) {
        //使用上一个relative_motion作为先验？
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose()); //使用两帧间的相对运动relative_motion_ * last_frame_计算当前位姿
    }

    int num_track_last = TrackLastFrame();//调用num_track_last跟踪上一帧的特征点
    tracking_inliers_ = EstimateCurrentPose();//调用 EstimateCurrentPose() 计算当前帧位姿，并返回内点数量

    if (tracking_inliers_ > num_features_tracking_) {
        // 如果内点数量多于阈值，这里是50，则认为好的跟踪，调整前端的状态
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // 如果内点数量达不到阈值，但是比最低的阈值高，则认为是坏的跟踪
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // 如果内点数量比最低的阈值还低，则认为是丢失跟踪
        status_ = FrontendStatus::LOST;
    }
    
    //调用 InsertKeyframe() 插入关键帧，计算相对运动
    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse(); //Tcw * Twl = Tcl

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

//插入关键帧
bool Frontend::InsertKeyframe() { //关键帧要重新进行三角化
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // 如果内点数量足够，则不插入关键帧
        return false;
    }
    // 将当前帧标记为关键帧，并插入地图，为了后续的地图更新、特征管理和优化操作
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;//记录日志，输出当前帧被设为关键帧的信息

    SetObservationsForKeyFrame();//为关键帧设置观测信息，将当前帧与地图中的已有地图点进行关联，标记哪些地图点是由当前帧观测到的
    DetectFeatures();  // 在当前帧中检测新的特征点
    FindFeaturesInRight();// 在当前帧的右目图像中寻找与左目图像中检测到的特征点的匹配
    TriangulateNewPoints();// 对新的特征点进行三角化，生成新的地图点
    
    backend_->UpdateMap();//通知后端模块更新地图

    if (viewer_) viewer_->UpdateMap();//如果有可视化模块viewer_，通知它更新显示内容

    return true;//返回 true，表示关键帧插入成功
}

//将每个特征点的观测添加到它所关联的地图点
//每当一个新的帧被添加时，系统会对当前帧中的特征点进行匹配，判断这些特征点是否已被映射到已有的地图点。
//如果是，系统就需要将当前特征点的观测信息加入到该地图点中。这样做的目的是不断更新地图点的观测数据，为后续的地图优化和位置估计提供更准确的数据。
void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {// 遍历当前帧左图的特征点集合
        auto mp = feat->map_point_.lock();//获取特征点的地图点
        if (mp) mp->AddObservation(feat);//如果特征点已经与地图点关联，将当前特征点添加到地图点的观测列表
    }
}

//初始化所需变量并准备三角化操作
int Frontend::TriangulateNewPoints() {
    std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};//获取左右相机的相机位姿。SE3d是一个表示位姿的变换矩阵
    Sophus::SE3d current_pose_Twc = current_frame_->Pose().inverse();//获取当前帧相机的世界坐标系到相机坐标系的变换（位姿的逆）
    int cnt_triangulated_pts = 0;// 初始化计数器，记录成功三角化的点数

    //遍历当前帧的特征点，并判断该特征点是否有有效的地图点关联，以及是否有右图的匹配特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() && //判断当前特征点是否与地图点关联，因为有一些在移除旧帧时把地图点清除了
            current_frame_->features_right_[i] != nullptr) {//判断当前特征点是否有有效的右图匹配点
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化，目的是把特征点成功插入地图中
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();//将左右图的特征点从像素坐标系转为相机坐标系


            //根据左右相机的位姿和特征点位置进行三角化，计算出三维点在世界坐标系中的位置，判断得到的点是否在相机前面
            // 如果三角化成功，则创建一个新的地图点，更新其位置，并将其加入到当前帧的特征点和地图中
            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();//创建一个新的地图点
                pworld = current_pose_Twc * pworld;//将三角化结果从当前相机坐标系转换到世界坐标系中
                new_map_point->SetPos(pworld);//设置新地图点的位置为三角化的结果
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);//将该地图点的观测信息添加到左右图的特征点中

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);//将新地图点插入到地图中
                cnt_triangulated_pts++;//增加成功三角化的点数
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

//设置g2o优化框架的求解器类型
int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // 定义块求解器类型，这里是6自由度位姿优化（3个平移自由度和3个旋转自由度）
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 定义线性求解器类型，采用稠密矩阵求解

    // 创建线性求解器
    auto linearSolver = new LinearSolverType();
    // 创建块求解器（依赖于线性求解器）
    auto blockSolver = new BlockSolverType(linearSolver);
    // 创建优化算法Levenberg
    auto solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    // 创建优化器并设置算法
    g2o::SparseOptimizer optimizer;//创建稀疏优化器
    optimizer.setAlgorithm(solver);//将优化算法Levenberg设置到优化器中

    // 创建顶点：相机位姿
    VertexPose *vertex_pose = new VertexPose();//创建一个位姿顶点
    vertex_pose->setId(0);//设置ID
    vertex_pose->setEstimate(current_frame_->Pose()); //设置顶点的初始估计为当前帧的位姿
    optimizer.addVertex(vertex_pose);

    // 获取相机内参矩阵
    Mat33 K = camera_left_->K();

    // 初始化边（优化中的约束）和特征点数组
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;

    //遍历当前帧的特征点，为每个与地图点关联的特征点创建一个投影边，并将其添加到优化器中
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);//创建一个投影边，表示特征点与地图点的投影关系
            edge->setId(index);//为每条边设置ID
            edge->setVertex(0, vertex_pose);//将位姿顶点与边关联
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));//设置边的观测值（特征点的位置）
            edge->setInformation(Eigen::Matrix2d::Identity());//设置信息矩阵（单位矩阵）
            edge->setRobustKernel(new g2o::RobustKernelHuber);//使用鲁棒核函数来避免异常值的影响
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // 优化位姿，并标记外点
    const double chi2_th = 5.991; //卡方检验阈值，卡方检验用于检测特征点是否为外点，5.991 是一个常见的卡方值，适用于2个自由度的分布
    int cnt_outlier = 0;//初始化记录外点的数量

    for (int iteration = 0; iteration < 4; ++iteration) { // 迭代4次
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10); // 每次迭代优化10次

        cnt_outlier = 0; // 外点计数

        // 遍历边并标记外点
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            e->computeError(); // 计算观测值与当前位姿的匹配误差

            if (e->chi2() > chi2_th) { // 如果误差大于阈值
                features[i]->is_outlier_ = true; // 标记为外点
                e->setLevel(1); // 设置边的优化等级为低
                cnt_outlier++;//计数点加1
            } else {
                features[i]->is_outlier_ = false; // 标记为内点
                e->setLevel(0); // 设置边的优化等级为高
            }

            // 第三次迭代后移除鲁棒核函数
            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    //输出内点和外点数量
    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;

    //将优化后的位姿赋给当前帧，并打印输出
    current_frame_->SetPose(vertex_pose->estimate());
    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    //将所有外点的关联地图点重置，意味着这些外点不会再参与后续的地图构建
    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset(); // 解除关联
            feat->is_outlier_ = false; // 重置外点标记，便于后续使用
        }
    }

    // 返回内点数量
    return features.size() - cnt_outlier;
}

//特征点追踪
int Frontend::TrackLastFrame() {
    // 准备特征点对
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) { //遍历上一帧的特征点，判断每个特征点是否已关联地图点
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose()); //这里的pose是之前设置的先验
            kps_last.push_back(kp->position_.pt);//上一帧中的特征点
            kps_current.push_back(cv::Point2f(px[0], px[1]));//对已关联地图点的特征点，将其三维坐标转换为当前帧图像中的像素坐标，作为追踪的初始点
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        } //注意这些点要一一对应
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK( //kps_current是给定初值方便更好追踪吗？？
        last_frame_->left_img_, current_frame_->left_img_, kps_last,//函数调用结构，输入图像和上一帧中的特征点
        kps_current, status, error, cv::Size(11, 11), 3,//分别是当前帧的特征点（计算后的位置）、每个特征点的跟踪状态（成功或失败）、每个特征点的光流误差、图像金字塔的窗口大小、金字塔层数
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01), // 终止条件
        cv::OPTFLOW_USE_INITIAL_FLOW); // 使用初始估计流

    int num_good_pts = 0;//初始化追踪好的点数量
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_; // 可以为空
            current_frame_->features_left_.push_back(feature);//根据光流计算结果，判断哪些特征点追踪成功（status[i] 为真），并将这些特征点添加到当前帧的特征点列表中
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

//在右图中找到匹配的特征，进而建立一个初步的地图
bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();//检测左图特征点数量并储存
    int num_coor_features = FindFeaturesInRight();//右图中寻找与左图中检测到的特征点相对应的匹配点
    if (num_coor_features < num_features_init_) {//如果找到的匹配点数量小于阈值就认为初始化失败
        return false;
    }

    bool build_map_success = BuildInitMap();//如果右图的特征点匹配足够，程序会调用 BuildInitMap() 函数来构建初始地图
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;//如果地图建立成功，更新前端的状态表示当前前端系统已经成功初始化，并且进入了“良好跟踪”状态
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();//如果可视化工具viewer存在,将当前帧添加到可视化界面,接着更新地图
        }
        return true;
    }
    return false;
}

//特征检测和特征匹配
int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);//创建一个与当前左图大小相同的二值掩模 mask。初始时，掩模的所有像素值设为 255（即全部可检测）
    for (auto &feat : current_frame_->features_left_) { //对于每个特征点，在 mask 中设置一个矩形区域（宽高为 20x20 像素）为 0（即不再检测这个区域内的特征点）。这样可以避免在已经有特征的区域内重复检测新特征
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);//使用GFTT来检测新的特征点，keypoints存储检测到的特征点。mask确保不在已有特征的附近检测特征
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back( //frame直接加入feature的shared_ptr, feature里为frame的weak_ptr
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    //利用光流估计右图中的关键点
    std::vector<cv::Point2f> kps_left, kps_right;//初始化两个列表 kps_left 和 kps_right 来存储左图和右图中的特征点坐标
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt); //把每个左feature里面的点坐标送入kps_left
        auto mp = kp->map_point_.lock(); //.lock 返回指向同一内容的shared_ptr, 若没有则返回空指针
        if (mp) { //若该左图中的feature对应的地图点存在，就使用该地图点的三维位置通过相机的右图投影模型 world2pixel 计算右图中的预期位置
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else { //否则直接使用左图相同像素坐标传给右图
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;//记录每个特征点是否成功匹配
    Mat error;//存储匹配误差
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,//同上，调用格式
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),//设置终止条件
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {//遍历status数组，检查每个特征点是否成功匹配，如果成功
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;//设置为false，表明特征点属于右图
            current_frame_->features_right_.push_back(feat); //将该特征点添加到当前帧的右图特征列表
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr); //要和左指针维度对上，一一对应
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

//BuildInitMap():
//1. 初始化左右相机位姿
//2. 遍历左图特征点
//    - 检查右图是否存在匹配的特征点
//    - 使用左右相机的特征点进行三角测量，计算地图点
//    - 如果三角测量成功且深度大于0，创建新地图点
//        - 设定地图点位置
//        - 将地图点与左、右图的特征点相关联
//        - 将新地图点加入地图
//3. 将当前帧设为关键帧，并将其插入地图
//4. 更新地图
bool Frontend::BuildInitMap() {
    std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()}; 
    //初始化一个包含左右相机位姿的向量poses，分别获取左、右相机的外部位姿
    size_t cnt_init_landmarks = 0;//初始化一个计数器，用于记录成功创建的地图点数
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {//遍历当前帧中的所有左图特征点
        if (current_frame_->features_right_[i] == nullptr) continue;
        //如果当前特征点在右图中没有匹配的特征点则跳过该特征点

        // vector points中分别加入左右相机的归一化三维坐标
        std::vector<Vec3> points{
            camera_left_->pixel2camera( //归一化三维坐标, depth = 0 在头文件里面定义
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera( //归一化三维坐标
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};

        Vec3 pworld = Vec3::Zero();//初始化一个三维向量 pworld，表示通过三角测量得到的世界坐标系下的地图点位置。初始值为零

        //使用三角测量方法triangulation（视差原理），通过左右相机的位姿poses和特征点的归一化三维坐标points计算出地图点的世界坐标pworld
        //深度要>0，则新建地图点并将其指针传入map类的landmarks里面
        if (triangulation(poses, points, pworld) && pworld[2] > 0 ) {
            auto new_map_point = MapPoint::CreateNewMappoint();//新建地图点
            new_map_point->SetPos(pworld);//设置地图点的位置，即将通过三角测量得到的世界坐标pworld赋值给新地图点的位置信息
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);//将当前帧中左图和右图的特征点分别加入到新地图点的观测列表中。即左图和右图的特征点观测到了该地图点。
            current_frame_->features_left_[i]->map_point_ = new_map_point; //注意 feature可能没有map_point
            current_frame_->features_right_[i]->map_point_ = new_map_point;//将新地图点的指针赋值给当前帧的左图和右图特征点的map_point_，表示这两个特征点关联了这个地图点
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);//将新创建的地图点插入到地图中，更新地图
        }
    }
    current_frame_->SetKeyFrame();//将当前帧标记为关键帧。这意味着当前帧包含了足够的信息，可以作为地图构建和后续优化的参考
    map_->InsertKeyFrame(current_frame_);//将当前帧作为关键帧插入到地图中
    backend_->UpdateMap();//调用后端的 UpdateMap() 方法，更新地图

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam