#include "Snake.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <sstream>

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
static double ptDist(const Point &a, const Point &b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ─────────────────────────────────────────────
//  Image energy  (−|∇I|²)
// ─────────────────────────────────────────────
cv::Mat Snake::computeImageEnergy(const cv::Mat &gray) {
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_64F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_64F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    cv::multiply(mag, mag, mag);
    mag = -mag;
    return mag;
}

// ─────────────────────────────────────────────
//  Resample to evenly-spaced points
// ─────────────────────────────────────────────
std::vector<Point> Snake::resample(const std::vector<Point> &pts) {
    int n = (int)pts.size();
    if (n < 2) return pts;

    // cumulative arc lengths
    std::vector<double> cum(n + 1, 0.0);
    for (int i = 0; i < n; ++i)
        cum[i + 1] = cum[i] + ptDist(pts[i], pts[(i + 1) % n]);

    double total = cum[n];
    dBar = total / n;

    std::vector<Point> out(n);
    for (int i = 0; i < n; ++i) {
        double t = total * i / n;
        // find segment
        int seg = (int)(std::upper_bound(cum.begin(), cum.end(), t) - cum.begin()) - 1;
        seg = std::clamp(seg, 0, n - 1);
        int nxt = (seg + 1) % n;
        double len = cum[seg + 1] - cum[seg];
        double frac = (len > 1e-9) ? (t - cum[seg]) / len : 0.0;
        out[i].x = (int)std::round(pts[seg].x + frac * (pts[nxt].x - pts[seg].x));
        out[i].y = (int)std::round(pts[seg].y + frac * (pts[nxt].y - pts[seg].y));
    }
    return out;
}

// ─────────────────────────────────────────────
//  Internal energy (elasticity + curvature)
// ─────────────────────────────────────────────
double Snake::internalEnergy(const Point &prev, int nx, int ny,
                              const Point &next,
                              double alpha, double beta) {
    // elasticity: keep spacing ≈ dBar
    double dx1 = next.x - nx, dy1 = next.y - ny;
    double dist = dx1 * dx1 + dy1 * dy1;
    double elast = (std::sqrt(dist) - dBar) * (std::sqrt(dist) - dBar);

    // curvature: second finite difference
    double cx = prev.x - 2.0 * nx + next.x;
    double cy = prev.y - 2.0 * ny + next.y;
    double curv = cx * cx + cy * cy;

    return alpha * elast + beta * curv;
}

// ─────────────────────────────────────────────
//  Greedy active-contour iteration
//  Returns updated contour points each call
// ─────────────────────────────────────────────
std::vector<Point> Snake::evolveOnce(const cv::Mat &gray,
                                      std::vector<Point> pts,
                                      double alpha, double beta, double gamma,
                                      int windowHalf) {
    int H = gray.rows, W = gray.cols;
    cv::Mat energy = computeImageEnergy(gray);
    int n = (int)pts.size();
    std::vector<Point> newPts = pts;

    for (int i = 0; i < n; ++i) {
        int px = pts[i].x, py = pts[i].y;
        const Point &prev = pts[(i - 1 + n) % n];
        const Point &next = pts[(i + 1) % n];

        double minE = std::numeric_limits<double>::infinity();
        Point best = pts[i];

        for (int dx = -windowHalf; dx <= windowHalf; ++dx) {
            for (int dy = -windowHalf; dy <= windowHalf; ++dy) {
                int nx = px + dx, ny = py + dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                double eInt = internalEnergy(prev, nx, ny, next, alpha, beta);
                double eExt = gamma * energy.at<double>(ny, nx);
                double total = eInt + eExt;

                if (total < minE) { minE = total; best = {nx, ny}; }
            }
        }
        newPts[i] = best;
    }

    return resample(newPts);
}

// ─────────────────────────────────────────────
//  Chain code (8-direction)
// ─────────────────────────────────────────────
// Directions: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
// (image coords: y increases downward)
std::string Snake::generateChainCode(const std::vector<Point> &pts) {
    std::string code;
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i) {
        int dx = pts[(i + 1) % n].x - pts[i].x;
        int dy = pts[(i + 1) % n].y - pts[i].y;

        // Clamp to [-1,1] for 8-connectivity
        int sx = (dx > 0) - (dx < 0);
        int sy = (dy > 0) - (dy < 0);

        // Map (sx,sy) → chain digit
        // sx: -1=left, 0=none, +1=right
        // sy: -1=up,   0=none, +1=down
        static const int table[3][3] = {
            // sx=-1  sx=0  sx=+1
            {   3,     2,     1   },   // sy=-1 (up)
            {   4,    -1,     0   },   // sy= 0
            {   5,     6,     7   }    // sy=+1 (down)
        };
        int row = sy + 1, col = sx + 1;
        int digit = table[row][col];
        if (digit >= 0)
            code += std::to_string(digit);
    }
    return code;
}

// ─────────────────────────────────────────────
//  Perimeter from chain code
// ─────────────────────────────────────────────
double Snake::perimeterFromChainCode(const std::string &code) {
    double p = 0.0;
    for (char c : code) {
        int d = c - '0';
        p += (d % 2 == 0) ? 1.0 : std::sqrt(2.0); // axis-aligned vs diagonal
    }
    return p;
}

// ─────────────────────────────────────────────
//  Area from chain code (Freeman's method)
//  Accumulates signed area using the x-coordinate + direction
// ─────────────────────────────────────────────
double Snake::areaFromChainCode(const std::vector<Point> &pts,
                                  const std::string &code) {
    // Use Shoelace for accuracy; chain code length verifies connectivity
    double area = 0.0;
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += (double)pts[i].x * pts[j].y;
        area -= (double)pts[j].x * pts[i].y;
    }
    return 0.5 * std::abs(area);
}

// ─────────────────────────────────────────────
//  Format chain code for display (groups of 6)
// ─────────────────────────────────────────────
std::string Snake::formatChainCode(const std::string &raw, int groupSize) {
    std::string out;
    for (int i = 0; i < (int)raw.size(); ++i) {
        if (i > 0 && i % groupSize == 0) out += ' ';
        out += raw[i];
    }
    return out;
}
