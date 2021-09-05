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

namespace SIFT::Utils {
    template<typename T, typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, uint8_t> || std::is_same_v<std::decay_t<T>, float>, void *>>
    void resize(const std::vector<std::vector<T>> &src, std::vector<std::vector<float>> &dst, int row, int col);

    bool gaussianElimination(std::vector<std::vector<float>> &A);

    void gaussBlur(const std::vector<std::vector<float>> &src, std::vector<std::vector<float>> &dst, float sigmaX, float sigmaY);

    void generateGaussianKernel(float sigma, std::vector<float> &dst);

    void
    mapHorizonKernelWithAVX(const std::vector<std::vector<float>> &src, const std::vector<float> &kernel, std::vector<std::vector<float>> &dst);

    void transformMatrix(std::vector<std::vector<float>> &mat);

    void SubtractWithAVX(const std::vector<std::vector<float>> &lhs, const std::vector<std::vector<float>> &rhs,
                         std::vector<std::vector<float>> &dst);
}
#endif
