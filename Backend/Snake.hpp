#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Point {
    int x, y;
};

class Snake {
public:
    Snake() : dBar(1.0) {}

    // One greedy iteration — call repeatedly from Python to animate
    std::vector<Point> evolveOnce(const cv::Mat &gray,
                                   std::vector<Point> pts,
                                   double alpha, double beta, double gamma,
                                   int windowHalf = 2);

    // Chain code + metrics
    std::string generateChainCode(const std::vector<Point> &pts);
    double      perimeterFromChainCode(const std::string &code);
    double      areaFromChainCode(const std::vector<Point> &pts,
                                   const std::string &code);
    std::string formatChainCode(const std::string &raw, int groupSize = 6);

private:
    double dBar;   // mean spacing between resampled points

    cv::Mat              computeImageEnergy(const cv::Mat &gray);
    std::vector<Point>   resample(const std::vector<Point> &pts);
    double               internalEnergy(const Point &prev, int nx, int ny,
                                         const Point &next,
                                         double alpha, double beta);
};
