#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // 检查命令行参数
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <dataset_dir>" << endl;
        return 1;
    }

    string dataset_dir = argv[1];
    ifstream fin(dataset_dir + "/associate.txt");
    if (!fin) {
        cout << "Could not open associate.txt in directory: " << dataset_dir << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof()) {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        if (!fin.good()) break;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);
    }
    fin.close();

    cout << "Generating features ... " << endl;
    vector<Mat> descriptors;
    Ptr<Feature2D> detector = ORB::create();
    int index = 1;
    for (string rgb_file : rgb_files) {
        Mat image = imread(rgb_file);
        if (image.empty()) {
            cout << "Failed to load image: " << rgb_file << endl;
            continue; // 跳过无法加载的图像
        }

        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        if (descriptor.empty()) {
            cout << "No descriptors found in image: " << rgb_file << endl;
            continue; // 跳过没有描述符的图像
        }
        descriptors.push_back(descriptor);
        cout << "Extracting features from image " << index++ << endl;
    }
    cout << "Extracted total " << descriptors.size() * 500 << " features." << endl;

    // 创建词汇表
    cout << "Creating vocabulary, please wait ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "Vocabulary info: " << vocab << endl;
    vocab.save("vocab_larger.yml.gz");
    cout << "Done" << endl;

    return 0;
}
