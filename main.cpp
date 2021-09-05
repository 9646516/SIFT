#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "SIFT.h"


int main() {
    cv::Mat canvas = cv::imread("E:\\img\\1.jpg");
    cv::Mat img = cv::imread("E:\\img\\1.jpg", 0);

    std::vector<std::vector<uint8_t>> V2;
    V2.resize(img.rows);
    for (int i = 0; i < (int) V2.size(); i++) {
        V2[i].resize(img.cols);
        for (int j = 0; j < (int) V2[i].size(); j++) {
            V2[i][j] = img.at<uint8_t>(i, j);
        }
    }

    std::vector<std::pair<int, int>> pos;
    std::vector<std::vector<float>> feature;
    SIFT::encode(V2, pos, feature);
    std::cout << pos.size() << std::endl;

    for (auto[a, b]: pos) {
        cv::circle(canvas, cv::Point(a, b), 3, cv::Vec3b(255, 0, 0), -1);;
    }
    cv::imwrite("E:\\img\\out6.jpg", canvas);
    return 0;
}
