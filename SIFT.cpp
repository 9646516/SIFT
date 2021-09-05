#include "SIFT.h"

void
SIFT::encode(const std::vector<std::vector<uint8_t>> &src, std::vector<std::pair<int, int>> &dstOfPos, std::vector<std::vector<float>> &dstOfFeature,
             float sigma, int S, float sigmaBefore, int border) {
    //prepare base image
    std::vector<std::vector<float>> baseImage;
    const int rowBefore = (int) src.size();
    const int colBefore = (int) src.front().size();
    sigmaBefore = sigmaBefore * 512.0f / (float) src.size();
    Utils::resize<uint8_t>(src, baseImage, rowBefore << 1, colBefore << 1);
    float sigmaNow = std::max(sqrt(sigma * sigma - sigmaBefore * sigmaBefore), 0.1f);
    Utils::gaussBlur(baseImage, baseImage, sigmaNow, sigmaNow);

    //prepare sigma of kernels
    int numberOfShots = S + 3;
    float K = std::powf(2.0, 1.0f / (float) S);
    std::vector<float> sigmaOfLayers(numberOfShots);
    sigmaOfLayers[0] = sigma;
    for (int i = 1; i < numberOfShots; i++) {
        sigmaOfLayers[i] = sigmaOfLayers[i - 1] * K;
    }
    for (int i = numberOfShots - 1; i > 1; i--) {
        sigmaOfLayers[i] = std::sqrt(sigmaOfLayers[i] * sigmaOfLayers[i] - sigmaOfLayers[i - 1] * sigmaOfLayers[i - 1]);
    }

    //prepare pyramid
    int numberOfLayers = int(std::log2(std::min(rowBefore, colBefore)));
    std::vector<std::vector<std::vector<std::vector<float>>>> pyramid;
    pyramid.resize(numberOfLayers);
    for (auto &now: pyramid) {
        now.resize(numberOfShots);
        now[0] = baseImage;
        for (int j = 1; j < numberOfShots; j++) {
            Utils::gaussBlur(baseImage, baseImage, sigmaOfLayers[j], sigmaOfLayers[j]);
            now[j] = baseImage;
        }
        const auto &last = now.end()[-3];
        Utils::resize(last, baseImage, (int) last.size() / 2, (int) last.front().size() / 2);
    }

    //prepare DOG
    std::vector<std::vector<std::vector<std::vector<float>>>> DOG;
    DOG.resize(numberOfLayers);
    for (int i = 0; i < numberOfLayers; i++) {
        DOG[i].resize(numberOfShots - 1);
        for (int j = 1; j < numberOfShots; j++) {
            Utils::SubtractWithAVX(pyramid[i][j], pyramid[i][j - 1], DOG[i][j - 1]);
        }
    }

    //find point of interest
    std::vector<keyPoint> keyPoints;
    findPointsOfInterest(DOG, keyPoints, border, sigma);
    std::sort(keyPoints.begin(), keyPoints.end());
    keyPoints.erase(std::unique(keyPoints.begin(), keyPoints.end()), keyPoints.end());

    //extract features
    dstOfPos.resize(keyPoints.size());
    dstOfFeature.resize(keyPoints.size());
    for (int i = 0; i < (int) keyPoints.size(); i++) {
        auto&[x, y, layer]=keyPoints[i];
        dstOfPos[i].first = x;
        dstOfPos[i].second = y;
        SIFT::getFeatures(x, y, layer, DOG, dstOfFeature[i]);
    }
}

void SIFT::findPointsOfInterest(const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG, std::vector<keyPoint> &dst,
                                int border,
                                float sigma) {
    const int S = (int) DOG.front().size() - 2;
    const float EPS = std::floor(0.5 * 0.04 / S * 255);
    for (int indexOfLayers = 0; indexOfLayers < (int) DOG.size(); indexOfLayers++) {
        auto &layers = DOG[indexOfLayers];
        const int row = (int) layers.front().size();
        const int col = (int) layers.front().front().size();
        for (int i = 1; i <= S; i++) {
            for (int j = border; j < row - border; j++) {
                for (int k = border; k < col - border; k++) {
                    if ([&]() -> bool {
                        if (std::abs(layers[i][j][k]) < EPS) {
                            return false;
                        } else {
                            std::function<bool(float, float)> cmp;
                            if (layers[i][j][k] > 0) {
                                cmp = std::greater_equal<>();
                            } else {
                                cmp = std::less_equal<>();
                            }
                            for (int dx: {-1, 0, 1}) {
                                for (int dy: {-1, 0, 1}) {
                                    for (int dz: {-1, 0, 1}) {
                                        if (!cmp(layers[i][j][k], layers[i + dx][j + dy][k + dz]))return false;
                                    }
                                }
                            }
                            return true;
                        }
                    }()) {
                        std::optional<std::tuple<int, int, int, float, float, float>> res = [&]()
                                -> std::optional<std::tuple<int, int, int, float, float, float>> {
                            int indexOfImage = i, indexOfRow = j, indexOfCol = k;
                            for (int attempt = 0; attempt < 5; attempt++) {
                                cube now;
                                for (int dx: {-1, 0, 1}) {
                                    for (int dy: {-1, 0, 1}) {
                                        for (int dz: {-1, 0, 1}) {
                                            now[dx + 1][dy + 1][dz + 1] = layers[indexOfImage + dx][indexOfRow + dy][indexOfCol + dz] / 255.0f;
                                        }
                                    }
                                }
                                auto[gx, gy, gz]= calcGrad(now);
                                std::vector<std::vector<float>> hessian;
                                calcHessian(now, hessian);
                                hessian[0].push_back(gx);
                                hessian[1].push_back(gy);
                                hessian[2].push_back(gz);
                                Utils::gaussianElimination(hessian);
                                float offsetOfX = hessian[0][3] / hessian[0][0];
                                float offsetOfY = hessian[1][3] / hessian[1][1];
                                float offsetOfZ = hessian[2][3] / hessian[2][2];
                                if (std::abs(offsetOfX) < 0.5 && std::abs(offsetOfY) < 0.5 && std::abs(offsetOfZ) < 0.5) {
                                    return std::make_optional(std::make_tuple(indexOfImage, indexOfRow, indexOfCol, offsetOfX, offsetOfY, offsetOfZ));
                                }
                                indexOfImage += int(std::round(gz));
                                indexOfRow += int(std::round(gy));
                                indexOfCol += int(std::round(gx));
                                if (indexOfRow < border || indexOfRow >= row - border ||
                                    indexOfCol < border || indexOfCol >= col - border ||
                                    indexOfImage < 1 || indexOfImage > S) {
                                    break;
                                }
                            }
                            return std::nullopt;
                        }();
                        if (res.has_value()) {
                            auto[indexOfImage, indexOfRow, indexOfCol, offsetOfX, offsetOfY, offsetOfZ]=res.value();
                            cube now;
                            for (int dx: {-1, 0, 1}) {
                                for (int dy: {-1, 0, 1}) {
                                    for (int dz: {-1, 0, 1}) {
                                        now[dx + 1][dy + 1][dz + 1] = layers[indexOfImage + dx][indexOfRow + dy][indexOfCol + dz];
                                    }
                                }
                            }
                            std::vector<std::vector<float>> hessian;
                            calcHessian(now, hessian);
                            float trace = hessian[0][0] + hessian[1][1];
                            float det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
                            auto[gx, gy, gz]= calcGrad(now);
                            float fx = now[1][1][1] + 0.5f * (gx * offsetOfX + gy * offsetOfY + gz * offsetOfZ);

                            if (std::abs(fx) * (float) S >= 0.04f && det > 0 && trace * trace / det < 12.1) {
                                float K = std::powf(2, (float) indexOfLayers - 1);
                                float x = ((float) indexOfCol + offsetOfX) * K;
                                float y = ((float) indexOfRow + offsetOfY) * K;
                                dst.emplace_back(int(x), int(y), indexOfLayers);
                            }
                        }
                    }
                }
            }
        }
    }
}

std::tuple<float, float, float> SIFT::calcGrad(const cube &now) {
    float dx = 0.5f * (now[1][1][2] - now[1][1][0]);
    float dy = 0.5f * (now[1][2][1] - now[1][0][1]);
    float ds = 0.5f * (now[2][1][1] - now[0][1][1]);
    return std::make_tuple(dx, dy, ds);

}

void SIFT::calcHessian(const cube &now, std::vector<std::vector<float>> &dst) {
    float center = now[1][1][1];
    float dxx = now[1][1][2] - 2 * center + now[1][1][0];
    float dyy = now[1][2][1] - 2 * center + now[1][0][1];
    float dss = now[2][1][1] - 2 * center + now[0][1][1];
    float dxy = 0.25f * (now[1][2][2] - now[1][2][0] - now[1][0][2] + now[1][0][0]);
    float dxs = 0.25f * (now[2][1][2] - now[2][1][0] - now[0][1][2] + now[0][1][0]);
    float dys = 0.25f * (now[2][2][1] - now[2][0][1] - now[0][2][1] + now[0][0][1]);
    dst = {
            {dxx, dxy, dxs},
            {dxy, dyy, dys},
            {dxs, dys, dss}
    };
}


void SIFT::getFeatures(int x, int y, int layer, const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG, std::vector<float> &dst) {

}


bool SIFT::operator<(const SIFT::keyPoint &lhs, const SIFT::keyPoint &rhs) {
    return lhs.x == rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y;
}

bool SIFT::operator==(const SIFT::keyPoint &lhs, const SIFT::keyPoint &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

SIFT::keyPoint::keyPoint(int _x, int _y, int _indexOfLayers) : x(_x), y(_y), indexOfLayers(_indexOfLayers) {}
