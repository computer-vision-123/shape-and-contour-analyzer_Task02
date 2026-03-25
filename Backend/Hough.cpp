#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include "image_utils.hpp"

namespace py = pybind11;

py::bytes detect_lines(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);

    if (edgeImg.channels() == 3)
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    cv::threshold(edgeImg, edgeImg, 127, 255, cv::THRESH_BINARY);

    if (drawImg.channels() == 1)
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);

    int rows = edgeImg.rows, cols = edgeImg.cols;

    // ── 1. Accumulator setup ────────────────────────────────────────────
    // theta: 0..179 degrees  (1° steps)
    // rho:   -diag .. +diag  (1 px steps), stored with offset so index >= 0
    const int nTheta = 180;
    double diagonal = std::sqrt(rows * rows + cols * cols);
    int nRho = 2 * static_cast<int>(std::ceil(diagonal)) + 1;
    int rhoOffset = nRho / 2;   // index for rho == 0

    // Pre-compute sin/cos tables
    std::vector<double> cosT(nTheta), sinT(nTheta);
    for (int t = 0; t < nTheta; ++t) {
        double angle = t * CV_PI / nTheta;
        cosT[t] = std::cos(angle);
        sinT[t] = std::sin(angle);
    }

    // ── 2. Vote ─────────────────────────────────────────────────────────
    std::vector<int> acc(nRho * nTheta, 0);

    for (int y = 0; y < rows; ++y) {
        const uchar* row = edgeImg.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            if (row[x] == 0) continue;
            for (int t = 0; t < nTheta; ++t) {
                int rho = static_cast<int>(std::round(x * cosT[t] + y * sinT[t]));
                acc[(rho + rhoOffset) * nTheta + t]++;
            }
        }
    }

    // ── 3. Find peaks (simple threshold + local-max in 5×5 window) ──────
    const int voteThresh = 80;
    struct Peak { int rho; int theta; int votes; };
    std::vector<Peak> peaks;

    for (int r = 0; r < nRho; ++r) {
        for (int t = 0; t < nTheta; ++t) {
            int v = acc[r * nTheta + t];
            if (v < voteThresh) continue;
            // Check if this is the local max in a 5×5 neighbourhood
            bool isMax = true;
            for (int dr = -2; dr <= 2 && isMax; ++dr)
                for (int dt = -2; dt <= 2 && isMax; ++dt) {
                    int nr = r + dr, nt = t + dt;
                    if (nr < 0 || nr >= nRho || nt < 0 || nt >= nTheta) continue;
                    if (acc[nr * nTheta + nt] > v) isMax = false;
                }
            if (isMax) peaks.push_back({r - rhoOffset, t, v});
        }
    }

    // Sort by votes (strongest first) and keep top 40
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b) { return a.votes > b.votes; });
    if (peaks.size() > 40) peaks.resize(40);

    // ── 4. Extract line segments from each peak ─────────────────────────
    // For each (rho, theta) line, walk across the image and collect edge
    // pixels within 2 px.  Group consecutive hits into segments (min 30 px).
    const int proximity = 2;
    const int minSegLen = 30;

    for (const auto& pk : peaks) {
        double cosA = cosT[pk.theta];
        double sinA = sinT[pk.theta];

        // Collect all edge points close to this line
        struct Pt { int x, y; double proj; };
        std::vector<Pt> pts;

        for (int y = 0; y < rows; ++y) {
            const uchar* row = edgeImg.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x) {
                if (row[x] == 0) continue;
                double dist = std::abs(x * cosA + y * sinA - pk.rho);
                if (dist <= proximity)
                    pts.push_back({x, y, x * cosA + y * sinA}); // proj unused but keep for sort
            }
        }

        if (pts.empty()) continue;

        // Sort points along the line direction (perpendicular component)
        // Direction vector of the line: (sinA, -cosA)
        for (auto& p : pts) p.proj = p.x * sinA - p.y * cosA;
        std::sort(pts.begin(), pts.end(),
                  [](const Pt& a, const Pt& b) { return a.proj < b.proj; });

        // Group into segments: break when gap > 8 px
        const double maxGap = 8.0;
        int segStart = 0;
        for (int i = 1; i <= (int)pts.size(); ++i) {
            bool gap = (i == (int)pts.size()) ||
                       (pts[i].proj - pts[i - 1].proj > maxGap);
            if (gap) {
                int dx = pts[i - 1].x - pts[segStart].x;
                int dy = pts[i - 1].y - pts[segStart].y;
                int len = static_cast<int>(std::sqrt(dx * dx + dy * dy));
                if (len >= minSegLen)
                    cv::line(drawImg,
                             cv::Point(pts[segStart].x, pts[segStart].y),
                             cv::Point(pts[i - 1].x, pts[i - 1].y),
                             cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                segStart = i;
            }
        }
    }

    return encode_image(drawImg);
}

py::bytes detect_circles(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);

    if (edgeImg.channels() == 3)
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    cv::threshold(edgeImg, edgeImg, 127, 255, cv::THRESH_BINARY);

    if (drawImg.channels() == 1)
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);

    int rows = edgeImg.rows, cols = edgeImg.cols;
    const int minR = 15;
    const int maxR = std::min(rows, cols) / 2;

    // ── 1. Compute Sobel gradients ──────────────────────────────────────
    cv::Mat gx, gy;
    cv::Sobel(edgeImg, gx, CV_32F, 1, 0, 3);
    cv::Sobel(edgeImg, gy, CV_32F, 0, 1, 3);

    // ── 2. Vote for circle centers along gradient direction ─────────────
    // For each edge pixel, cast votes at (x ± r·gx/|g|, y ± r·gy/|g|)
    // for r in [minR, maxR].
    std::vector<int> acc(rows * cols, 0);

    for (int y = 0; y < rows; ++y) {
        const uchar* eRow = edgeImg.ptr<uchar>(y);
        const float* gxRow = gx.ptr<float>(y);
        const float* gyRow = gy.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (eRow[x] == 0) continue;
            float mag = std::sqrt(gxRow[x] * gxRow[x] + gyRow[x] * gyRow[x]);
            if (mag < 1e-5f) continue;
            float dx = gxRow[x] / mag;
            float dy = gyRow[x] / mag;
            // Vote in both gradient directions (+ and -)
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int r = minR; r <= maxR; r += 2) {  // step 2 for speed
                    int cx = static_cast<int>(std::round(x + sign * r * dx));
                    int cy = static_cast<int>(std::round(y + sign * r * dy));
                    if (cx >= 0 && cx < cols && cy >= 0 && cy < rows)
                        acc[cy * cols + cx]++;
                }
            }
        }
    }

    // ── 3. Find center peaks (threshold + non-maximum suppression) ──────
    const int centerThresh = 40;
    const int nmsRadius = 20;   // suppress duplicates within this distance

    struct Center { int x, y, votes; };
    std::vector<Center> candidates;

    for (int y = nmsRadius; y < rows - nmsRadius; ++y) {
        for (int x = nmsRadius; x < cols - nmsRadius; ++x) {
            int v = acc[y * cols + x];
            if (v < centerThresh) continue;
            // Local max in nmsRadius × nmsRadius window (check 5×5 for speed)
            bool isMax = true;
            for (int dy = -3; dy <= 3 && isMax; ++dy)
                for (int dx = -3; dx <= 3 && isMax; ++dx) {
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < rows && nx >= 0 && nx < cols)
                        if (acc[ny * cols + nx] > v) isMax = false;
                }
            if (isMax) candidates.push_back({x, y, v});
        }
    }

    // Keep the strongest 20 candidates
    std::sort(candidates.begin(), candidates.end(),
              [](const Center& a, const Center& b) { return a.votes > b.votes; });
    if (candidates.size() > 20) candidates.resize(20);

    // Suppress centres that are too close to a stronger one
    std::vector<Center> filtered;
    for (const auto& c : candidates) {
        bool tooClose = false;
        for (const auto& f : filtered) {
            int ddx = c.x - f.x, ddy = c.y - f.y;
            if (ddx * ddx + ddy * ddy < nmsRadius * nmsRadius) { tooClose = true; break; }
        }
        if (!tooClose) filtered.push_back(c);
    }

    // ── 4. For each center, find the best radius ────────────────────────
    // Histogram of edge-pixel distances from the center; pick the peak.
    for (const auto& c : filtered) {
        std::vector<int> rHist(maxR + 1, 0);
        for (int y = 0; y < rows; ++y) {
            const uchar* eRow = edgeImg.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x) {
                if (eRow[x] == 0) continue;
                int ddx = x - c.x, ddy = y - c.y;
                int dist = static_cast<int>(std::round(std::sqrt(ddx * ddx + ddy * ddy)));
                if (dist >= minR && dist <= maxR)
                    rHist[dist]++;
            }
        }
        // Find the radius with the most support
        int bestR = minR, bestVotes = 0;
        for (int r = minR; r <= maxR; ++r) {
            // Sum a small window [r-1, r+1] to handle discretisation
            int sum = rHist[r];
            if (r > minR) sum += rHist[r - 1];
            if (r < maxR) sum += rHist[r + 1];
            if (sum > bestVotes) { bestVotes = sum; bestR = r; }
        }

        // Only draw if there is reasonable circular support
        if (bestVotes < 15) continue;

        cv::circle(drawImg, cv::Point(c.x, c.y), bestR,
                   cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(drawImg, cv::Point(c.x, c.y), 2,
                   cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }

    return encode_image(drawImg);
}

py::bytes detect_ellipses(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);

    if (edgeImg.channels() == 3) {
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(edgeImg, edgeImg, 127, 255, cv::THRESH_BINARY);

    if (drawImg.channels() == 1) {
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);
    }

    // ── Algorithm parameters (matching MATLAB defaults) ──────────────────
    const int    minMajorAxis   = 10;
    const int    maxMajorAxis   = static_cast<int>(std::sqrt(
                     edgeImg.rows * edgeImg.rows + edgeImg.cols * edgeImg.cols));
    const double rotation       = 0.0;
    const double rotationSpan   = 0.0;       // 0 → no angular constraint
    const double minAspectRatio = 0.1;
    const int    randomize      = 2;
    const int    numBest        = 3;
    const bool   uniformWeights = true;
    const double smoothStddev   = 1.0;
    const double eps            = 0.0001;

    // ── 1. Collect non-zero edge points ──────────────────────────────────
    std::vector<cv::Point> pts;
    cv::findNonZero(edgeImg, pts);
    int N = static_cast<int>(pts.size());

    // bestFits: each row = {x0, y0, a, b, angle_deg, score}
    std::vector<std::array<double, 6>> bestFits(numBest, {0, 0, 0, 0, 0, 0});

    if (N < 2) {
        return encode_image(drawImg);
    }

    // Extract X, Y as float vectors
    std::vector<float> X(N), Y(N);
    for (int i = 0; i < N; ++i) {
        X[i] = static_cast<float>(pts[i].x);
        Y[i] = static_cast<float>(pts[i].y);
    }

    // ── 2. Find valid point pairs (candidate major axes) ─────────────────
    const float minMASq = static_cast<float>(minMajorAxis) * minMajorAxis;
    const float maxMASq = static_cast<float>(maxMajorAxis) * maxMajorAxis;

    struct Pair { int i, j; float distSq; };
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(N) * 4);   // heuristic reservation

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float dx = X[i] - X[j];
            float dy = Y[i] - Y[j];
            float dSq = dx * dx + dy * dy;
            if (dSq >= minMASq && dSq <= maxMASq) {
                pairs.push_back({i, j, dSq});
            }
        }
    }

    // ── 3. Angular constraint ────────────────────────────────────────────
    if (rotationSpan > 0 && rotationSpan < 90) {
        double tanLo = std::tan((rotation - rotationSpan) * CV_PI / 180.0);
        double tanHi = std::tan((rotation + rotationSpan) * CV_PI / 180.0);
        std::vector<Pair> filtered;
        filtered.reserve(pairs.size());
        for (auto& p : pairs) {
            float tangent = (Y[p.i] - Y[p.j]) / (X[p.i] - X[p.j] + static_cast<float>(eps));
            bool keep = (tanLo < tanHi)
                ? (tangent > tanLo && tangent < tanHi)
                : (tangent > tanLo || tangent < tanHi);
            if (keep) filtered.push_back(p);
        }
        pairs = std::move(filtered);
    }

    int npairs = static_cast<int>(pairs.size());
    if (npairs == 0) {
        return encode_image(drawImg);
    }

    // ── 4. Random subsampling ────────────────────────────────────────────
    std::vector<int> pairSubset;
    if (randomize > 0) {
        int nSample = std::min(npairs, std::max(1, N * randomize));
        pairSubset.resize(npairs);
        std::iota(pairSubset.begin(), pairSubset.end(), 0);
        // Fisher–Yates partial shuffle
        std::mt19937 rng(42);   // fixed seed for reproducibility
        for (int i = 0; i < nSample && i < npairs; ++i) {
            std::uniform_int_distribution<int> dist(i, npairs - 1);
            std::swap(pairSubset[i], pairSubset[dist(rng)]);
        }
        pairSubset.resize(nSample);
    } else {
        pairSubset.resize(npairs);
        std::iota(pairSubset.begin(), pairSubset.end(), 0);
    }

    // ── 5. Gaussian smoothing kernel (1-D) ───────────────────────────────
    int kSize = std::max(1, static_cast<int>(std::round(smoothStddev * 6)));
    if (kSize % 2 == 0) kSize++;          // make it odd for symmetry
    std::vector<float> kernel(kSize);
    {
        int half = kSize / 2;
        float sum = 0;
        for (int k = 0; k < kSize; ++k) {
            float x = static_cast<float>(k - half);
            kernel[k] = std::exp(-0.5f * x * x / static_cast<float>(smoothStddev * smoothStddev));
            sum += kernel[k];
        }
        for (auto& v : kernel) v /= sum;
    }

    // ── 6. Main loop: evaluate each candidate major axis ─────────────────
    for (int pidx : pairSubset) {
        const Pair& pr = pairs[pidx];
        float x1 = X[pr.i], y1 = Y[pr.i];
        float x2 = X[pr.j], y2 = Y[pr.j];

        // Centre and semi-major axis squared
        float x0 = (x1 + x2) * 0.5f;
        float y0 = (y1 + y2) * 0.5f;
        float aSq = pr.distSq * 0.25f;

        // Accumulator indexed by minor-axis length (1..maxMajorAxis)
        std::vector<float> accumulator(maxMajorAxis + 1, 0.0f);

        for (int k = 0; k < N; ++k) {
            float dx0 = X[k] - x0;
            float dy0 = Y[k] - y0;
            float dSq = dx0 * dx0 + dy0 * dy0;
            if (dSq > aSq) continue;          // point must be within semi-major distance

            float dx2 = X[k] - x2;
            float dy2 = Y[k] - y2;
            float fSq = dx2 * dx2 + dy2 * dy2;

            float denom = 2.0f * std::sqrt(aSq * dSq);
            if (denom < eps) continue;

            float cosTau = (aSq + dSq - fSq) / denom;
            cosTau = std::min(1.0f, std::max(-1.0f, cosTau));
            float sinTauSq = 1.0f - cosTau * cosTau;

            float bDenom = aSq - dSq * cosTau * cosTau + static_cast<float>(eps);
            if (bDenom <= 0) continue;

            float bVal = std::sqrt((aSq * dSq * sinTauSq) / bDenom);
            int bin = static_cast<int>(std::ceil(bVal + eps));
            if (bin < 1 || bin > maxMajorAxis) continue;

            float w = uniformWeights
                ? 1.0f
                : static_cast<float>(edgeImg.at<uchar>(static_cast<int>(Y[k]), static_cast<int>(X[k])));
            accumulator[bin] += w;
        }

        // Smooth the accumulator
        {
            std::vector<float> smoothed(maxMajorAxis + 1, 0.0f);
            int half = kSize / 2;
            for (int b = 1; b <= maxMajorAxis; ++b) {
                float s = 0;
                for (int kk = 0; kk < kSize; ++kk) {
                    int idx = b + kk - half;
                    if (idx >= 1 && idx <= maxMajorAxis)
                        s += accumulator[idx] * kernel[kk];
                }
                smoothed[b] = s;
            }
            accumulator = std::move(smoothed);
        }

        // Zero out bins below minAspectRatio * a
        int cutoff = static_cast<int>(std::ceil(std::sqrt(aSq) * minAspectRatio));
        for (int b = 0; b <= std::min(cutoff, maxMajorAxis); ++b)
            accumulator[b] = 0;

        // Find peak
        float bestScore = 0;
        int   bestBin   = 0;
        for (int b = 1; b <= maxMajorAxis; ++b) {
            if (accumulator[b] > bestScore) {
                bestScore = accumulator[b];
                bestBin   = b;
            }
        }

        // Keep top-numBest
        if (bestScore > bestFits.back()[5]) {
            float angleDeg = std::atan2(y1 - y2, x1 - x2) * 180.0f / static_cast<float>(CV_PI);
            bestFits.back() = {
                static_cast<double>(x0), static_cast<double>(y0),
                std::sqrt(static_cast<double>(aSq)), static_cast<double>(bestBin),
                static_cast<double>(angleDeg), static_cast<double>(bestScore)
            };
            if (numBest > 1) {
                std::sort(bestFits.begin(), bestFits.end(),
                    [](const auto& a, const auto& b) { return a[5] > b[5]; });
            }
        }
    }

    // ── 7. Draw the detected ellipses ────────────────────────────────────
    for (const auto& fit : bestFits) {
        if (fit[5] <= 0) continue;   // skip empty slots

        double cx    = fit[0];
        double cy    = fit[1];
        double semiA = fit[2];       // semi-major
        double semiB = fit[3];       // semi-minor
        double angle = fit[4];

        cv::ellipse(drawImg,
            cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
            cv::Size(static_cast<int>(semiA), static_cast<int>(semiB)),
            angle, 0, 360,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    return encode_image(drawImg);
}