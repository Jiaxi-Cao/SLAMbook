#include <gtest/gtest.h>
#include "myslam/common_include.h"
#include "myslam/algorithm.h"

TEST(MyslamTest, Triangulation) {
    Eigen::Vector3d pt_world(30, 20, 10), pt_world_estimated;
    std::vector<Sophus::SE3d> poses{
        Sophus::SE3d(Eigen::Quaterniond(0, 0, 0, 1), Eigen::Vector3d(0, 0, 0)),
        Sophus::SE3d(Eigen::Quaterniond(0, 0, 0, 1), Eigen::Vector3d(0, -10, 0)),
        Sophus::SE3d(Eigen::Quaterniond(0, 0, 0, 1), Eigen::Vector3d(0, 10, 0)),
    };

    std::vector<Eigen::Vector3d> points; // 修改为 Vector3d
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Vector3d pc = poses[i] * pt_world;
        pc /= pc[2];
        points.push_back(Eigen::Vector3d(pc[0], pc[1], 1.0)); // 添加第三个维度
    }

    EXPECT_TRUE(myslam::triangulation(poses, points, pt_world_estimated));
    EXPECT_NEAR(pt_world[0], pt_world_estimated[0], 0.01);
    EXPECT_NEAR(pt_world[1], pt_world_estimated[1], 0.01);
    EXPECT_NEAR(pt_world[2], pt_world_estimated[2], 0.01);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}