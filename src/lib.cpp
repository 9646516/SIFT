#include "lib.h"

void
SIFT::encode(const std::vector<std::vector<uint8_t>> &src, std::vector<std::pair<int, int>> &dstOfPos, std::vector<std::vector<float>> &dstOfFeature,
             float sigma, int S, float sigmaBefore, int border) {
    //prepare base image
    std::vector<std::vector<float>> baseImage;
    const int rowBefore = (int) src.size();
    const int colBefore = (int) src.front().size();
    {
        sigmaBefore = sigmaBefore + sigmaBefore;
        Utils::resize<uint8_t>(src, baseImage, rowBefore << 1, colBefore << 1);
        float sigmaNow = sqrt(sigma * sigma - sigmaBefore * sigmaBefore);
        Utils::gaussBlur(baseImage, baseImage, sigmaNow, sigmaNow);
    }

    //prepare sigma of kernels
    const int numberOfShots = S + 3;
    std::vector<float> totalSigmaOfLayers(numberOfShots);
    std::vector<float> sigmaOfLayers(numberOfShots);
    {
        const float K = std::powf(2.0, 1.0f / (float) S);
        sigmaOfLayers[0] = sigma;
        for (int i = 1; i < numberOfShots; i++) {
            sigmaOfLayers[i] = sigmaOfLayers[i - 1] * K;
        }
        for (int i = 0; i < numberOfShots; i++) {
            totalSigmaOfLayers[i] = sigmaOfLayers[i];
        }
        for (int i = numberOfShots - 1; i >= 1; i--) {
            sigmaOfLayers[i] = std::sqrt(sigmaOfLayers[i] * sigmaOfLayers[i] - sigmaOfLayers[i - 1] * sigmaOfLayers[i - 1]);
        }
    }

    //prepare pyramid
    const int numberOfLayers = int(std::log2(std::min(rowBefore, colBefore)));
    std::vector<std::vector<std::vector<std::vector<float>>>> pyramid;
    {
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
    }
    //prepare DOG
    std::vector<std::vector<std::vector<std::vector<float>>>> DOG;
    {
        DOG.resize(numberOfLayers);
        for (int i = 0; i < numberOfLayers; i++) {
            DOG[i].resize(numberOfShots - 1);
            for (int j = 1; j < numberOfShots; j++) {
                Utils::SubtractWithAVX(pyramid[i][j], pyramid[i][j - 1], DOG[i][j - 1]);
            }
        }
    }
    //find point of interest
    std::vector<keyPoint> keyPoints;
    {
        findPointsOfInterest(DOG, keyPoints, border);
        std::sort(keyPoints.begin(), keyPoints.end());
        keyPoints.erase(std::unique(keyPoints.begin(), keyPoints.end()), keyPoints.end());
    }

    //extract features
    {
        std::vector<float> arc;
        for (auto &i: keyPoints) {
            SIFT::getArc(i, pyramid[i.indexOfLayers][i.indexOfImage + 1], arc, totalSigmaOfLayers[i.indexOfImage + 1]);
            for (float angle: arc) {
                dstOfPos.emplace_back(i.x, i.y);
                dstOfFeature.resize(dstOfFeature.size() + 1);
                SIFT::getFeature(i, pyramid[i.indexOfLayers][i.indexOfImage + 1], dstOfFeature.end()[-1],
                                 totalSigmaOfLayers[i.indexOfImage + 1], angle);
            }
        }
    }
}

void SIFT::findPointsOfInterest(const std::vector<std::vector<std::vector<std::vector<float>>>> &DOG, std::vector<keyPoint> &dst, int border) {
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
                                ///x,y represent cv::Point(x,y)
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
                                float x = ((float) indexOfCol + offsetOfX);
                                float y = ((float) indexOfRow + offsetOfY);
                                keyPoint kp{};
                                kp.x = int(std::round(x * K));
                                kp.y = int(std::round(y * K));
                                kp.indexOfLayers = indexOfLayers;
                                kp.indexOfImage = int(std::round((float) i + offsetOfZ));
                                kp.xInPyramid = int(std::round(x));
                                kp.yInPyramid = int(std::round(y));
                                dst.push_back(kp);
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

void SIFT::getArc(const keyPoint &kp, const std::vector<std::vector<float>> &src, std::vector<float> &dst, float sigma) {
    dst.clear();
    const int row = (int) src.size();
    const int col = (int) src.front().size();
    int r = int(1.5f * sigma);

    std::vector<float> histogram(36, 0);
    std::vector<float> smoothedHistogram(36);
    for (int i = -r; i <= r; i++) {
        int x = kp.yInPyramid + i;
        if (x > 0 && x < row - 1) {
            for (int j = -r; j <= r; j++) {
                int y = kp.xInPyramid + j;
                if (y > 0 && y < col - 1) {
                    float dx = src[x][y + 1] - src[x][y - 1];
                    float dy = src[x - 1][y] - src[x + 1][y];
                    float mag = std::sqrt(dx * dx + dy * dy);
                    float arc = std::atan2(dy, dx) / 3.1415926f * 180.0f;
                    int idx = int(std::round(arc / 10.0f)) % 36;
                    if (idx < 0)idx += 36;
                    histogram[idx] += mag;
                }
            }
        }
    }
    for (int i = 0; i < 36; i++) {
        smoothedHistogram[i] = (6.0f * histogram[i] +
                                4.0f * (histogram[(i + 35) % 36] + histogram[(i + 1) % 36]) +
                                histogram[(i + 34) % 36] + histogram[(i + 2) % 36]
                               ) / 16.0f;
    }
    float max = -1;
    for (auto i: smoothedHistogram) {
        max = std::max(max, i);
    }
    for (int i = 0; i < 36; i++) {
        float value = smoothedHistogram[i];
        if (value >= 0.8 * max) {
            int prev = (i + 35) % 36;
            int nxt = (i + 1) % 36;
            float lValue = smoothedHistogram[prev];
            float rValue = smoothedHistogram[nxt];
            if (value > lValue && value > rValue) {
                float newIdx = std::fmod(float(i) + 0.5f * (lValue - rValue) / (lValue - 2 * value + rValue), 36.0f);
                float newArc = 360.0f - newIdx * 10.0f;
                if (std::fabs(newArc - 360.0f) < 1e-4) {
                    newArc = 0.0f;
                }
                dst.push_back(newArc);
            }
        }
    }
}

void SIFT::getFeature(const keyPoint &kp, const std::vector<std::vector<float>> &src, std::vector<float> &dst, float sigma, float angle) {
    const int row = (int) src.size();
    const int col = (int) src.front().size();
    const float rotate = 360.0f - angle;
    float cos = std::cos(rotate / 180.0f * 3.1415926f);
    float sin = std::sin(rotate / 180.0f * 3.1415926f);
    const float histWidth = 3 * sigma;
    const float K = -0.125f;

    int radius = int(std::round(histWidth * 1.414f * (4 + 1) * 0.5f));
    radius = std::min(radius, (int) sqrt(((double) col) * col + ((double) row) * row));
    std::vector<float> X, Y, W;
    std::vector<float> Rbin, Cbin, Angle, Mag;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float ox = float(i) * cos + float(j) * sin;
            float oy = -float(i) * sin + float(j) * cos;
            float rbin = ox / histWidth + 2.0f - 0.5f;
            float cbin = oy / histWidth + 2.0f - 0.5f;
            int c = kp.xInPyramid + i;
            int r = kp.yInPyramid + j;
            if (rbin >= 0 && rbin < 4 && cbin >= 0 && cbin < 4 && r > 0 && r < row - 1 && c > 0 && c < col - 1) {
                float dx = src[r][c + 1] - src[r][c - 1];
                float dy = src[r - 1][c] - src[r + 1][c];
                X.push_back(dx);
                Y.push_back(dy);
                Rbin.push_back(rbin);
                Cbin.push_back(cbin);
                W.push_back((ox * ox + oy * oy) * K / histWidth / histWidth);
                Mag.push_back(std::sqrt(dx * dx + dy * dy));
            }
        }
    }

    const int sz = (int) Rbin.size();
    const int step = sz / SIFT::Utils::packedSizeAVX;

    {
        // 180 / 3.1415926 = 57.295780490442965
        //Angle=(atan(Y,X)*57.295-angle)/45
        Angle.resize(sz);
        const float sub[8] = {angle, angle, angle, angle, angle, angle, angle, angle};
        const static float mul[8] = {57.295, 57.295, 57.295, 57.295, 57.295, 57.295, 57.295, 57.295};
        const static float divide[8] = {45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0};

        __m256 subVector = _mm256_load_ps(sub);
        __m256 divVector = _mm256_load_ps(divide);
        __m256 mulVector = _mm256_load_ps(mul);
        const float *ptrY = Y.data();
        const float *ptrX = X.data();
        float *ptrZ = Angle.data();
        for (int i = 0; i < step; i++) {
            __m256 vectorOfX = _mm256_load_ps(ptrX);
            __m256 vectorOfY = _mm256_load_ps(ptrY);
            __m256 res = _mm256_atan2_ps(vectorOfY, vectorOfX);

            res = _mm256_mul_ps(res, mulVector);
            res = _mm256_sub_ps(res, subVector);
            res = _mm256_div_ps(res, divVector);
            _mm256_store_ps(ptrZ, res);
            ptrX += SIFT::Utils::packedSizeAVX;
            ptrY += SIFT::Utils::packedSizeAVX;
            ptrZ += SIFT::Utils::packedSizeAVX;
        }
        for (int i = step * SIFT::Utils::packedSizeAVX; i < sz; i++) {
            Angle[i] = (std::atan2(Y[i], X[i]) * 57.295f - angle) / 45.0f;
        }
        for (auto &i: Angle) {
            i = std::fmod(i, 8.0f);
            if (i < 0)i += 8;
        }
    }

    {
//        Mag=Mag*exp(W)
        float *ptrW = W.data();
        float *ptrMag = Mag.data();
        for (int i = 0; i < step; i++) {
            __m256 vectorOfW = _mm256_load_ps(ptrW);
            vectorOfW = _mm256_exp_ps(vectorOfW);
            __m256 vectorOfMag = _mm256_load_ps(ptrMag);
            vectorOfMag = _mm256_mul_ps(vectorOfMag, vectorOfW);
            _mm256_store_ps(ptrMag, vectorOfMag);

            ptrW += SIFT::Utils::packedSizeAVX;
            ptrMag += SIFT::Utils::packedSizeAVX;
        }
        for (int i = step * Utils::packedSizeAVX; i < sz; i++) {
            Mag[i] = std::exp(W[i]) * Mag[i];
        }
    }

    std::vector<float> RLow(sz), CLow(sz), ArcLow(sz);
    std::vector<float> c111(sz), c110(sz), c101(sz), c011(sz), c100(sz), c010(sz), c001(sz), c000(sz);
    {
        //RLow=floor(R)
        //CLow=floor(C)
        //ArcLow=floor(Arc)
        //offsetOfR=R-RLow
        //offsetOfC=C-CLow
        //offsetOfArc=Arc-ArcLow

        //c1=Mag*offsetOfR
        //c0=Mag-c1

        //c11=c1*offsetOfC
        //c10=c1-c11
        //c01=c0*offsetOfC
        //c00=c0-c01

        //c111=c11*offsetOfArc
        //c110=c11-c111
        //c101=c10*offsetOfArc
        //c100=c10-c101
        //c011=c01*offsetOfArc
        //c010=c01-c011
        //c001=c00*offsetOfArc
        //c000=c00-c001
        int idx = 0;
        for (int i = 0; i < step; i++) {
            __m256 R = _mm256_load_ps(&Rbin[idx]);
            __m256 C = _mm256_load_ps(&Cbin[idx]);
            __m256 Arc = _mm256_load_ps(&Angle[idx]);
            __m256 M = _mm256_load_ps(&Mag[idx]);
            __m256 RLow_v = _mm256_floor_ps(R);
            __m256 CLow_v = _mm256_floor_ps(C);
            __m256 ArcLow_v = _mm256_floor_ps(Arc);
            __m256 offsetOfR = _mm256_sub_ps(R, RLow_v);
            __m256 offsetOfC = _mm256_sub_ps(C, CLow_v);
            __m256 offsetOfArc = _mm256_sub_ps(Arc, ArcLow_v);

            __m256 c1_v = _mm256_mul_ps(M, offsetOfR);
            __m256 c0_v = _mm256_sub_ps(M, c1_v);

            __m256 c11_v = _mm256_mul_ps(c1_v, offsetOfC);
            __m256 c10_v = _mm256_sub_ps(c1_v, c11_v);
            __m256 c01_v = _mm256_mul_ps(c0_v, offsetOfC);
            __m256 c00_v = _mm256_sub_ps(c0_v, c01_v);

            __m256 c111_v = _mm256_mul_ps(c11_v, offsetOfArc);
            __m256 c110_v = _mm256_sub_ps(c11_v, c111_v);
            __m256 c101_v = _mm256_mul_ps(c10_v, offsetOfArc);
            __m256 c100_v = _mm256_sub_ps(c10_v, c101_v);
            __m256 c011_v = _mm256_mul_ps(c01_v, offsetOfArc);
            __m256 c010_v = _mm256_sub_ps(c01_v, c011_v);
            __m256 c001_v = _mm256_mul_ps(c00_v, offsetOfArc);
            __m256 c000_v = _mm256_sub_ps(c00_v, c001_v);

            _mm256_store_ps(&RLow[idx], RLow_v);
            _mm256_store_ps(&CLow[idx], CLow_v);
            _mm256_store_ps(&ArcLow[idx], ArcLow_v);

            _mm256_store_ps(&c000[idx], c000_v);
            _mm256_store_ps(&c001[idx], c001_v);
            _mm256_store_ps(&c010[idx], c010_v);
            _mm256_store_ps(&c011[idx], c011_v);
            _mm256_store_ps(&c100[idx], c100_v);
            _mm256_store_ps(&c101[idx], c101_v);
            _mm256_store_ps(&c110[idx], c110_v);
            _mm256_store_ps(&c111[idx], c111_v);

            idx += SIFT::Utils::packedSizeAVX;
        }
        for (; idx < sz; idx++) {
            RLow[idx] = std::floor(Rbin[idx]);
            CLow[idx] = std::floor(Cbin[idx]);
            ArcLow[idx] = std::floor(Angle[idx]);

            float offsetOfR = Rbin[idx] - RLow[idx];
            float offsetOfC = Cbin[idx] - CLow[idx];
            float offsetOfArc = Angle[idx] - ArcLow[idx];

            float c1 = Mag[idx] * offsetOfR;
            float c0 = Mag[idx] - c1;

            float c11 = c1 * offsetOfC;
            float c10 = c1 - c11;
            float c01 = c0 * offsetOfC;
            float c00 = c0 - c01;

            c111[idx] = c11 * offsetOfArc;
            c110[idx] = c11 - c111[idx];
            c101[idx] = c10 * offsetOfArc;
            c100[idx] = c10 - c101[idx];
            c011[idx] = c01 * offsetOfArc;
            c010[idx] = c01 - c011[idx];
            c001[idx] = c00 * offsetOfArc;
            c000[idx] = c00 - c001[idx];
        }
    }
    std::vector<std::vector<std::vector<float>>> res(6, std::vector<std::vector<float>>(6, std::vector<float>(8, 0)));
    for (int i = 0; i < sz; i++) {
        int r = int(RLow[i]), c = int(CLow[i]), arc = int(ArcLow[i]);
        res[r + 1][c + 1][arc] += c000[i];
        res[r + 1][c + 1][(arc + 1) % 8] += c001[i];
        res[r + 1][c + 2][arc] += c010[i];
        res[r + 1][c + 2][(arc + 1) % 8] += c011[i];
        res[r + 2][c + 1][arc] += c100[i];
        res[r + 2][c + 1][(arc + 1) % 8] += c101[i];
        res[r + 2][c + 2][arc] += c110[i];
        res[r + 2][c + 2][(arc + 1) % 8] += c111[i];
    }

    {
        //dst=flatten(crop(res))
        dst.resize(128);
        float *ptr = dst.data();
        for (int i = 1; i <= 4; i++) {
            for (int j = 1; j <= 4; j++) {
                __m256 t = _mm256_load_ps(&res[i][j][0]);
                _mm256_store_ps(ptr, t);
                ptr += SIFT::Utils::packedSizeAVX;
            }
        }
    }

    std::function<float(std::vector<float> &)> calcL2WithAVX = [](const std::vector<float> &src) -> float {
        float L2 = 0;
        const float *ptr = src.data();
        __m256 total = _mm256_setzero_ps();
        int step = 128 / SIFT::Utils::packedSizeAVX;
        for (int i = 0; i < step; i++) {
            __m256 now = _mm256_load_ps(ptr);
            now = _mm256_mul_ps(now, now);
            total = _mm256_add_ps(now, total);
            ptr += SIFT::Utils::packedSizeAVX;
        }
        static float temp[8];
        _mm256_store_ps(temp, total);
        for (float i: temp) {
            L2 += i;
        }
        L2 = std::sqrt(L2);
        return L2;
    };
    float L2 = calcL2WithAVX(dst);
    float threshold = L2 * 0.2f;
    for (auto &i: dst) {
        if (i > threshold) {
            i = threshold;
        }
    }
    float normal = std::max(calcL2WithAVX(dst), 1e-5f);
    for (auto &i: dst) {
        i = std::round(i * 512 / normal);
        if (i < 0)i = 0;
        else if (i > 255)i = 255;
    }
}

bool SIFT::operator<(const SIFT::keyPoint &lhs, const SIFT::keyPoint &rhs) {
    return lhs.x == rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y;
}

bool SIFT::operator==(const SIFT::keyPoint &lhs, const SIFT::keyPoint &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}