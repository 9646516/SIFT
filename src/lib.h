#ifndef SIFT_LIB_H
#define SIFT_LIB_H

#include <set>
#include <map>
#include <functional>
#include <array>
#include <optional>
#include <algorithm>
#include <vector>
#include "utils.h"

#ifdef IMPORT_SIFT_DLL
#define EXPORT_DLL __declspec(dllimport)
#else
#define EXPORT_DLL __declspec(dllexport)
#endif

namespace SIFT {
    struct keyPoint {
        int x, y;
        int indexOfLayers;
        int indexOfImage;
        int xInPyramid;
        int yInPyramid;
    };

    bool operator<(const keyPoint &lhs, const keyPoint &rhs);

    bool operator==(const keyPoint &lhs, const keyPoint &rhs);

    using cube = std::array<std::array<std::array<float, 3>, 3>, 3>;

    EXPORT_DLL void
    encode(const std::vector<std::vector<uint8_t>> &src, std::vector<std::pair<int, int>> &dstOfPos, std::vector<std::vector<float>> &dstOfFeature,
           float sigma = 1.6, int S = 3, float sigmaBefore = 0.5, int border = 5);

    std::tuple<float, float, float> calcGrad(const cube &now);

    void calcHessian(const cube &now, std::vector<std::vector<float>> &dst);

    void
    findPointsOfInterest(const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG,
                         std::vector<keyPoint> &dst, int border);

    void getArc(const keyPoint &kp, const std::vector<std::vector<float>> &src, std::vector<float> &dst, float sigma);

    void getFeature(const keyPoint &kp, const std::vector<std::vector<float>> &src, std::vector<float> &dst, float sigma, float angle);
}
#endif