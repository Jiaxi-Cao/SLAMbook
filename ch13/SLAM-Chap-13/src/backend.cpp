//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"
#include <memory>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

namespace myslam {

Backend::Backend() {
    backend_running_.store(true); //原子值设为真，表示线程处于活动状态
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this)); //开启后端线程
    // this指针指向该类的一个实例的地址
    // 每个非静态函数的第一个参数都是隐式的this, 这里直接传入则可以跑该实例的这个函数
    // 并开启 backend_thread_ 线程
    // https://blog.51cto.com/u_15127662/4035706
}

//地图发生变化时由前端调用
void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lock(data_mutex_);//通过unique_lock<std::mutex>锁住data_mutex_这样做是为了防止在多线程环境下同时访问和修改map_update，确保同步操作
    map_update_.notify_one();//唤醒等待在条件变量上的后端线程，通知后端开始优化操作
}

//停止后端线程
void Backend::Stop() {
    backend_running_.store(false);//设置为false，停止优化线程
    map_update_.notify_one();
    backend_thread_.join();//唤醒后端线程并等待线程退出，注意，唤醒之后进入等待状态，而不是唤醒之后直接开始优化
}

//后端线程的主循环
void Backend::BackendLoop() {
    while (backend_running_.load()) {//只要backend_running_为 true，循环会持续运行，不断执行任务
        std::unique_lock<std::mutex> lock(data_mutex_);//通过加锁，避免与前端或其他线程之间的并发访问问题
        map_update_.wait(lock);//后端线程在此处等待，直到接收到前端发出的优化信号

        /// 后端仅优化map中激活的关键帧和地图点
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);//提取当前激活的关键帧和地图点，调用 Optimize 函数进行优化
    }
}

//调用的优化函数
void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // 定义块求解器和线性求解器类型
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

// 使用 new 创建线性求解器和块求解器
LinearSolverType* linearSolver = new LinearSolverType();
BlockSolverType* blockSolver = new BlockSolverType(linearSolver);

// 使用块求解器创建优化算法
g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

// 用图优化G2O通过最小化残差（误差）来优化 关键帧位姿和地图点位置
//步骤大致为：配置优化器、初始化关键帧的位姿顶点、初始化地图点顶点、为每个地图点创建边、为地图点添加新的顶点
//为边设置信息和鲁棒核、执行优化、外点剔除、更新外点标记并移除观测、更新优化后的位姿和地图点位置
g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(solver);

    // 每个keyframe的 pose为顶点VertexPose，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;// 声明一个 std::map，用于存储每个关键帧对应的 VertexPose 对象
    unsigned long max_kf_id = 0;//关键帧ID
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;//获取当前关键帧对象并赋值给 kf
        VertexPose *vertex_pose = new VertexPose();  // 创建一个新的 VertexPose 对象，代表该关键帧的位姿
        vertex_pose->setId(kf->keyframe_id_);//设置 vertex_pose 的 ID，ID 对应关键帧的 ID
        vertex_pose->setEstimate(kf->Pose());//设置 vertex_pose 的初始估计值为当前关键帧的位姿
        optimizer.addVertex(vertex_pose);//添加到优化器（optimizer）中，表示该顶点参与优化
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;//跟踪最大的关键帧 ID，使其保持为目前最大的关键帧 ID
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // K 和左右外参
    Mat33 K = cam_left_->K();
    Sophus::SE3d left_ext = cam_left_->pose();
    Sophus::SE3d right_ext = cam_right_->pose();

    // edges
    double chi2_th = 5.991;//鲁棒核函数的阈值（用于检测外点）
    int index = 1;//ID
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue; //地图点或观测点是外点则跳过
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->GetObs(); //得到该地图点对观测
        for (auto &obs : observations) { //遍历所有观测feature
            if (obs.lock() == nullptr) continue;//bs.lock() == nullptr则该观测已经失效，跳过
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue; //外点或对应的帧无效，则跳过

            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr; //创立该观测对应的残差边
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }//根据哪边的图像用哪边相机的内外参

            // 如果该地图点还没有对应的顶点（加入优化），则新加一个顶点并添加到优化器中
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);//为每条边设置一个ID
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    // 获取一个Keyframe对应的位姿，并将其设置为边的第一个顶点
            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // 获取landmark对应的地图点，设置为边的第二个顶点
            edge->setMeasurement(toVec2(feat->position_.pt));//获取像素坐标，转为二维向量，然后设置为边的测量值
            edge->setInformation(Mat22::Identity());// 设置边的信息矩阵（单位矩阵）
            auto rk = new g2o::RobustKernelHuber();//创建一个鲁棒核函数
            rk->setDelta(chi2_th); //设置鲁棒核函数的阈值，大于该阈值则视为外点，降低影响
            edge->setRobustKernel(rk);//把鲁棒核函数绑定到边，使得在优化过程中，边的残差根据Huber核函数进行加权，从而降低外点对优化结果的影响
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);//将当前的边 edge 添加到优化器中

            index++;
        }
    }

    //初始化优化并执行优化过程，最多执行 10 次迭代
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // robust kernel 阈值 可以理解为像素距离差吗？
    int cnt_outlier = 0, cnt_inlier = 0;//统计外点和内点的数量，都初始化为0
    int iteration = 0;//用于控制最大迭代次数，初始化为0
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {//遍历每条边和特征点
            if (ef.first->chi2() > chi2_th) {//遍历边的时候检查残差值chi2，若大于阈值则为外点，否则为内点
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);//内点的数量除以内点和外点的总数
        if (inlier_ratio > 0.5) {
            break;
        } else { //如果内点还不到一半，则增加误差阈值，让内点变多
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;//如果边的残差大于阈值则认为该特征点是外点
            // 从该特征点对应的地图点map_point中移除该观测
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;//残差小于阈值，认为它是内点，设置为false
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
        //v.second->estimate() 返回优化后的位姿
        //然后->SetPose将优化后的位姿更新到对应的关键帧
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam