#ifndef SIFT_SIFT_H
#define SIFT_SIFT_H

#include <functional>
#include <array>
#include <optional>
#include <algorithm>
#include <vector>
#include "utils.h"

namespace SIFT {
    struct keyPoint {
        int x, y;
        int indexOfLayers;

        keyPoint(int _x, int _y, int _indexOfLayers);
    };

    bool operator<(const keyPoint &lhs, const keyPoint &rhs);

    bool operator==(const keyPoint &lhs, const keyPoint &rhs);

    using cube = std::array<std::array<std::array<float, 3>, 3>, 3>;

    void
    encode(const std::vector<std::vector<uint8_t>> &src, std::vector<std::pair<int, int>> &dstOfPos, std::vector<std::vector<float>> &dstOfFeature,
           float sigma = 1.6, int S = 3, float sigmaBefore = 0.5, int border = 5);

    std::tuple<float, float, float> calcGrad(const cube &now);

    void calcHessian(const cube &now, std::vector<std::vector<float>> &dst);

    void
    findPointsOfInterest(const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG,
                         std::vector<keyPoint> &dst, int border, float sigma);

    void getFeatures(int x, int y, int layer, const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG, std::vector<float>& dst);

}

#endif
