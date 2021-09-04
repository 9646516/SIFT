#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "utils.h"

int main() {
    cv::Mat img = cv::imread("E:\\img\\1.jpg", 0);
    auto imgOld = img.clone();
    cv::GaussianBlur(img, img, cv::Size(25, 25), 4, 4);
    cv::imwrite("E:\\img\\out1.jpg", img);


    std::vector<std::vector<uint8_t>> V2;
    V2.resize(img.rows);
    for (int i = 0; i < (int) V2.size(); i++) {
        V2[i].resize(img.cols);
        for (int j = 0; j < (int) V2[i].size(); j++) {
            V2[i][j] = imgOld.at<uint8_t>(i, j);
        }
    }

    utils::gaussBlur(V2, V2, 4, 4);
    for (int i = 0; i < (int) V2.size(); i++) {
        for (int j = 0; j < (int) V2[i].size(); j++) {
            img.at<uint8_t>(i, j) = V2[i][j];
        }
    }
    cv::imwrite("E:\\img\\out2.jpg", img);
    return 0;

    return 0;
}
