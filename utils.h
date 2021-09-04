#ifndef SIFT_UTILS_H
#define SIFT_UTILS_H

#include <numeric>
#include <ctime>
#include <cstdint>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>
#include <cmath>
#include <immintrin.h>

namespace utils {
    void resize(const std::vector<std::vector<uint8_t>> &src, std::vector<std::vector<uint8_t>> &dst, int row, int col);

    bool gaussianElimination(std::array<std::array<float, 4>, 3> &A);

    void gaussBlur(const std::vector<std::vector<uint8_t>> &src, std::vector<std::vector<uint8_t>> &dst, float sigmaX, float sigmaY);

    void generateGaussianKernel(float sigma, std::vector<float> &dst);

    void
    mapHorizonKernelWithAVX(const std::vector<std::vector<float>> &src, const std::vector<float> &kernel, std::vector<std::vector<float>> &dst);

    void transformMatrix(std::vector<std::vector<float>> &mat);
};
#endif
