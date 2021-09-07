#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "lib.h"
#include <random>
#include <set>

#define INPORT_SIFT_DLL

uint8_t gen() {
    static std::mt19937 rng(114514);
    static std::uniform_int_distribution<int> G(0, 255);
    return G(rng);
}

float dis(const std::vector<float> &x, const std::vector<float> &y) {
    __m256 res = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 8) {
        __m256 X = _mm256_load_ps(&x[i]);
        __m256 Y = _mm256_load_ps(&y[i]);
        __m256 Z = _mm256_sub_ps(X, Y);
        Z = _mm256_mul_ps(Z, Z);
        res = _mm256_add_ps(res, Z);
    }
    float temp[8];
    _mm256_store_ps(temp, res);
    float ret = 0;
    for (float i: temp) {
        ret += i;
    }
    ret = std::sqrt(ret);
    return ret;
}


int main() {
    cv::Mat img1 = cv::imread("E:\\img\\1.jpg", 0);
    cv::resize(img1, img1, cv::Size(512, 512));
    auto img2 = img1.clone();
    cv::rotate(img2, img2, cv::ROTATE_90_CLOCKWISE);
    std::vector<std::vector<uint8_t>> V1;
    V1.resize(img1.rows);
    for (int i = 0; i < (int) V1.size(); i++) {
        V1[i].resize(img1.cols);
        for (int j = 0; j < (int) V1[i].size(); j++) {
            V1[i][j] = img1.at<uint8_t>(i, j);
        }
    }
    std::vector<std::pair<int, int>> pos1;
    std::vector<std::vector<float>> feature1;

    SIFT::encode(V1, pos1, feature1);

    std::vector<std::vector<uint8_t>> V2;
    V2.resize(img2.rows);
    for (int i = 0; i < (int) V2.size(); i++) {
        V2[i].resize(img2.cols);
        for (int j = 0; j < (int) V2[i].size(); j++) {
            V2[i][j] = img2.at<uint8_t>(i, j);
        }
    }
    std::vector<std::pair<int, int>> pos2;
    std::vector<std::vector<float>> feature2;

    SIFT::encode(V2, pos2, feature2);

    cv::Mat canvas(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8U, cv::Scalar(0));
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            canvas.at<uint8_t>(i, j) = img1.at<uint8_t>(i, j);
        }
    }
    for (int i = 0; i < img2.rows; i++) {
        for (int j = 0; j < img2.cols; j++) {
            canvas.at<uint8_t>(i, j + img1.cols) = img2.at<uint8_t>(i, j);
        }
    }
    std::vector<std::tuple<float, int, int>> res;
    for (int i = 0; i < (int) pos1.size(); i++) {
        std::set<std::pair<float, int>> st;
        for (int j = 0; j < (int) pos2.size(); j++) {
            float d = ::dis(feature1[i], feature2[j]);
            st.emplace(d, j);
            if (st.size() > 2)st.erase(std::prev(st.end()));
        }
        auto[d1, idx1]=*st.begin();
        auto[d2, idx2]=*std::next(st.begin());
        if (d1 < d2 * 0.7)
            res.emplace_back(d1, i, idx1);
    }
    if (res.size() > 40)
        res.resize(40);
    cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
    for (auto &[d, i, j]: res) {
        cv::circle(canvas, cv::Point(pos1[i].first, pos1[i].second), 2, cv::Vec3b(255, 0, 0));
        cv::circle(canvas, cv::Point(pos2[j].first + img1.cols, pos2[j].second), 2, cv::Vec3b(0, 255, 0));
        cv::line(canvas, cv::Point(pos1[i].first, pos1[i].second), cv::Point(pos2[j].first + img1.cols, pos2[j].second),
                 cv::Vec3b(gen(), gen(), gen()), 1);
    }
    cv::imshow("", canvas);
    cv::waitKey(0);

    cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
    for (auto[a, b]: pos1) {
        cv::circle(img1, cv::Point(a, b), 2, cv::Scalar(255, 0, 0), -1);
    }
    cv::imshow("", img1);
    cv::waitKey();
    return 0;
}