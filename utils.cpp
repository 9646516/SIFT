#include "utils.h"

#define x0 __SIFT_UTILS_x0
#define y0 __SIFT_UTILS_y0
#define x1 __SIFT_UTILS_x1
#define y1 __SIFT_UTILS_y1

void utils::resize(const std::vector<std::vector<uint8_t>> &src, std::vector<std::vector<uint8_t>> &dst, int row, int col) {
    int srcRow = (int) src.size();
    int srcCol = (int) src.front().size();
    float dx = static_cast<float>(srcRow) / static_cast<float>(row);
    float dy = static_cast<float>(srcCol) / static_cast<float>(col);
    dst.resize(row);
    for (int i = 0; i < row; i++) {
        dst[i].resize(col);
        for (int j = 0; j < col; j++) {
            float posX = dx * static_cast<float>(i);
            float posY = dy * static_cast<float>(j);
            int x0 = int(posX);
            int y0 = int(posY);
            int x1 = x0 + 1;
            if (x1 >= srcRow)x1 = x0;
            int y1 = y0 + 1;
            if (y1 >= srcCol)y1 = y0;
            float ox = posX - x0;
            float oy = posY - y0;
            float topInterpolationResult = std::lerp(src[x0][y0], src[x0][y1], oy);
            float bottomnterpolationResult = std::lerp(src[x1][y0], src[x1][y1], oy);
            float res = std::lerp(topInterpolationResult, bottomnterpolationResult, ox);
            dst[i][j] = static_cast<uint8_t>(res);
        }
    }
}

bool utils::gaussianElimination(std::array<std::array<float, 4>, 3> &A) {
    for (int i = 0; i < 3; i++) {
        int chooseIndex = -1;
        for (int j = i; j < 3; j++) {
            if (A[j][i] != 0) {
                chooseIndex = j;
                break;
            }
        }
        if (chooseIndex == -1) {
            return false;
        } else {
            if (chooseIndex != i) {
                std::swap(A[chooseIndex], A[i]);
            }
            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    float K = A[j][i] / A[i][i];
                    for (int k = i; k < 4; k++) {
                        A[j][k] -= A[i][k] * K;
                    }
                }
            }
        }
    }
    return true;
}

void utils::generateGaussianKernel(float sigma, std::vector<float> &dst) {
    int ksize = int(sigma * 6 + 1) | 1;
    dst.resize(ksize);
    float sum = 0;
    for (int i = 0; i < ksize; i++) {
        int r = i - (ksize - 1) / 2;
        float dis = float(r) * float(r);
        dst[i] = std::exp(-dis / (2 * sigma * sigma));
        sum += dst[i];
    }
    for (float &i: dst) {
        i /= sum;
    }
    std::cout << sigma << ' ' << ksize << ' ' << std::accumulate(dst.begin(), dst.end(), 0.0) << std::endl;
    for (auto i: dst) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}

void utils::gaussBlur(const std::vector<std::vector<uint8_t>> &src, std::vector<std::vector<uint8_t>> &dst, float sigmaX, float sigmaY) {
    const int row = (int) src.size();
    const int col = (int) src.front().size();
    const int packedSizeAVX = sizeof(__m256) / sizeof(float);

    std::vector<float> kernel;
    std::vector<std::vector<float>> mid, temp;

    mid.resize(row);
    for (int i = 0; i < row; i++) {
        mid[i].resize(col);
        for (int j = 0; j < col; j++) {
            mid[i][j] = src[i][j];
        }
    }
    #define doGaussBlur() \
    do{\
        const int kernelSize = (int) kernel.size();\
        const int leftPadding = kernelSize / 2;\
        while (kernel.size() % packedSizeAVX) {\
            kernel.push_back(0);\
        }\
        const int rightPadding = (int) kernel.size() - kernelSize + leftPadding;\
        for (auto &i: mid) {\
            for (int j = 0; j < rightPadding; j++) {\
                i.push_back(0);\
            }\
            std::reverse(i.begin(), i.end());\
            for (int j = 0; j < leftPadding; j++) {\
                i.push_back(0);\
            }\
            std::reverse(i.begin(), i.end());\
        }\
        utils::mapHorizonKernelWithAVX(mid, kernel, temp);\
        std::swap(mid, temp);\
    }while(0)

    utils::generateGaussianKernel(sigmaX, kernel);
    if (kernel.size() > 1) {
        doGaussBlur();
    }

    utils::generateGaussianKernel(sigmaY, kernel);
    if (kernel.size() > 1) {
        utils::transformMatrix(mid);
        doGaussBlur();
        utils::transformMatrix(mid);
    }

    dst.resize(row);
    for (int i = 0; i < row; i++) {
        dst[i].resize(col);
        for (int j = 0; j < col; j++) {
            dst[i][j] = std::max(uint8_t(0), std::min(uint8_t(255), uint8_t(mid[i][j])));
        }
    }
}

void utils::mapHorizonKernelWithAVX(const std::vector<std::vector<float>> &src, const std::vector<float> &kernel,
                                    std::vector<std::vector<float>> &dst) {
    const int ksize = (int) kernel.size();
    const int packedSizeAVX = sizeof(__m256) / sizeof(float);
    const int step = ksize / packedSizeAVX;
    const int row = (int) src.size();
    const int col = (int) src.front().size();
    const int rightBoundOfCol = col - packedSizeAVX + 1;
    std::vector<__m256> packedKernel;
    packedKernel.reserve(step);
    for (int i = 0; i < step; i++) {
        packedKernel.push_back(_mm256_load_ps(&kernel[i * packedSizeAVX]));
    }
    dst.resize(row);
    for (int i = 0; i < row; i++) {
        dst[i].resize(col);
        for (int j = 0; j < rightBoundOfCol; j++) {
            dst[i][j] = 0;
            const float *ptr = &src[i][j];
            for (int k = 0; k < step; k++) {
                __m256 now = _mm256_loadu_ps(ptr);
                ptr += packedSizeAVX;
                now = _mm256_dp_ps(now, packedKernel[k], 0xF1);
                __m128 hi = _mm256_extractf128_ps(now, 1);
                __m128 lo = _mm256_castps256_ps128(now);
                dst[i][j] += _mm_cvtss_f32(hi) + _mm_cvtss_f32(lo);
            }
        }
    }
}

void utils::transformMatrix(std::vector<std::vector<float>> &mat) {
    const int row = (int) mat.size();
    const int col = (int) mat.front().size();
    std::vector<std::vector<float>> temp;
    temp.resize(col);
    for (int i = 0; i < col; i++) {
        temp[i].resize(row);
        for (int j = 0; j < row; j++) {
            temp[i][j] = mat[j][i];
        }
    }
    std::swap(mat, temp);
}
