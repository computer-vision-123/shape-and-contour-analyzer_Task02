#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <cmath>
#include <vector>
#include <limits>

#include "image_utils.hpp" 

namespace py = pybind11;

// Defining the gaussian and prewitt kernals

static const double GAUSSIAN_KERNEL[7][7] = {
    { 1, 1, 2,  2, 2, 1, 1 },
    { 1, 2, 2,  4, 2, 2, 1 },
    { 2, 2, 4,  8, 4, 2, 2 },
    { 2, 4, 8, 16, 8, 4, 2 }, 
    { 2, 2, 4,  8, 4, 2, 2 },
    { 1, 2, 2,  4, 2, 2, 1 },
    { 1, 1, 2,  2, 2, 1, 1 }
};

static const double PREWITT_X[3][3] = {
    {-1, 0, 1},
    {-1, 0, 1},
    {-1, 0, 1}
};
static const double PREWITT_Y[3][3] = {
    { 1,  1,  1},
    { 0,  0,  0},
    {-1, -1, -1}
};

// Checks for invalid pixels to avoid convolution

static bool hasInvalidNeighbors(const cv::Mat& img, int row, int col) {
    for (int r = row - 1; r <= row + 1; ++r) {
        for (int c = col - 1; c <= col + 1; ++c) {
            
            if (std::isnan(img.at<double>(r, c))) {
                return true; 
            }
        }
    }
    return false; 
}

// 1. Gaussian Smoothing
static cv::Mat gaussianSmoothing(const cv::Mat& image) {
    
    cv::Mat imageArray;
    image.convertTo(imageArray, CV_64F);
    cv::Mat gaussianArr = imageArray.clone();

    for (int i = 3; i < image.rows - 3; ++i) {
        for (int j = 3; j < image.cols - 3; ++j) {

            double sum = 0.0;

            for (int r = -3; r <= 3; ++r)
                for (int c = -3; c <= 3; ++c)
            
                    sum += (GAUSSIAN_KERNEL[r+3][c+3] / 140.0) * imageArray.at<double>(i+r, j+c);

            gaussianArr.at<double>(i, j) = sum;
        }
    }
    return gaussianArr;
}

// 2. Gradients (X, Y, Magnitude, Angle)
static void calculateGradients(const cv::Mat& smoothImage, cv::Mat& magnitude, cv::Mat& angle) {
    const int height = smoothImage.rows;
    const int width = smoothImage.cols;
    
    cv::Mat gradXMatrix(height, width, CV_64F, std::numeric_limits<double>::quiet_NaN());
    cv::Mat gradYMatrix(height, width, CV_64F, std::numeric_limits<double>::quiet_NaN());
    
    magnitude = cv::Mat::zeros(height, width, CV_64F);
    angle = cv::Mat::zeros(height, width, CV_64F);

    for (int row = 3; row < height - 5; ++row) {
        for (int col = 3; col < width - 5; ++col) {
            
            if (hasInvalidNeighbors(smoothImage, row, col)) {
                continue; 
            }

            double horizontalSum = 0.0;
            double verticalSum = 0.0;

            for (int kernelRow = 0; kernelRow < 3; ++kernelRow) {
                for (int kernelCol = 0; kernelCol < 3; ++kernelCol) {
                    
                    double pixelValue = smoothImage.at<double>(row + kernelRow, col + kernelCol);
                    
                    horizontalSum += pixelValue * (PREWITT_X[kernelRow][kernelCol] / 3.0);
                    verticalSum += pixelValue * (PREWITT_Y[kernelRow][kernelCol] / 3.0);
                }
            }

            int targetRow = row + 1;
            int targetCol = col + 1;

            gradXMatrix.at<double>(targetRow, targetCol) = std::abs(horizontalSum);
            gradYMatrix.at<double>(targetRow, targetCol) = std::abs(verticalSum);
            
            double absGradX = gradXMatrix.at<double>(targetRow, targetCol);
            double absGradY = gradYMatrix.at<double>(targetRow, targetCol);

            magnitude.at<double>(targetRow, targetCol) = std::sqrt((absGradX * absGradX) + (absGradY * absGradY)) / 1.4142;
            
            double calculatedAngle = 0.0;
            
            if (absGradX == 0.0) {
                calculatedAngle = (absGradY > 0.0) ? 90.0 : -90.0;
            } else {
                calculatedAngle = std::atan(absGradY / absGradX) * (180.0 / M_PI);
            }

            if (calculatedAngle < 0.0) {
                calculatedAngle += 360.0;
            }
            
            angle.at<double>(targetRow, targetCol) = calculatedAngle;
        }
    }
}


// 3.Non-Maxima Suppression
struct NMSResult { 
    cv::Mat suppressed; 
    std::vector<int> histogram; 
    int edgeCount; 
};

static NMSResult localMaximization(const cv::Mat& magnitude, const cv::Mat& angle) {
    NMSResult result;
    result.suppressed = cv::Mat::zeros(magnitude.size(), CV_64F);
    result.histogram.assign(256, 0);
    result.edgeCount = 0;

    for (int row = 5; row < magnitude.rows - 5; ++row) {
        for (int col = 5; col < magnitude.cols - 5; ++col) {
            
            double currentAngle = angle.at<double>(row, col);
            double currentMagnitude = magnitude.at<double>(row, col);
            double suppressedMagnitude = 0.0;

            if ((currentAngle >= 0 && currentAngle <= 22.5) || 
                (currentAngle > 157.5 && currentAngle <= 202.5) || 
                (currentAngle > 337.5 && currentAngle <= 360)) {
                
                if (currentMagnitude > magnitude.at<double>(row, col + 1) && 
                    currentMagnitude > magnitude.at<double>(row, col - 1)) {
                    suppressedMagnitude = currentMagnitude;
                }
            } 

            else if ((currentAngle > 22.5 && currentAngle <= 67.5) || 
                     (currentAngle > 202.5 && currentAngle <= 247.5)) {
                
                if (currentMagnitude > magnitude.at<double>(row + 1, col - 1) && 
                    currentMagnitude > magnitude.at<double>(row - 1, col + 1)) {
                    suppressedMagnitude = currentMagnitude;
                }
            } 
            else if ((currentAngle > 67.5 && currentAngle <= 112.5) || 
                     (currentAngle > 247.5 && currentAngle <= 292.5)) {
                
                if (currentMagnitude > magnitude.at<double>(row + 1, col) && 
                    currentMagnitude > magnitude.at<double>(row - 1, col)) {
                    suppressedMagnitude = currentMagnitude;
                }
            } 
            else {
                if (currentMagnitude > magnitude.at<double>(row + 1, col + 1) && 
                    currentMagnitude > magnitude.at<double>(row - 1, col - 1)) {
                    suppressedMagnitude = currentMagnitude;
                }
            }

            if (suppressedMagnitude > 0) {
                result.suppressed.at<double>(row, col) = suppressedMagnitude;
                result.edgeCount++;
                int histogramBin = std::min(255, std::max(0, static_cast<int>(suppressedMagnitude)));
                result.histogram[histogramBin]++;
            }
        }
    }
    
    return result;
}

// 4. P-Tile Thresholding & Binarization
static cv::Mat applyPTileThreshold(const NMSResult& nms, int t_high_percent) {

    double target = std::round(nms.edgeCount * (t_high_percent / 100.0));
    double sum = 0;
    int cutoff = 255;
    
    for (int i = 255; i > 0; --i) {
        sum += nms.histogram[i];
        
        if (sum >= target) { 
            cutoff = i; 
            break; 
        }
    }

    cv::Mat binary;
    nms.suppressed.convertTo(binary, CV_8U); 

    for (int i = 0; i < binary.rows; ++i) {
        for (int j = 0; j < binary.cols; ++j) {
            uchar& p = binary.at<uchar>(i, j);
            p = (p < cutoff) ? 0 : 255;
        }
    }
    
    return binary;
}

// controller
py::bytes run_canny(const py::bytes& imageBytes, int t_high_percent) {
    // 1. Decode
    cv::Mat src = decode_image(imageBytes);
    cv::Mat gray;
    
    // Ensure the image is single-channel grayscale
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    // 2. Blur and Gradients
    cv::Mat smooth = gaussianSmoothing(gray);
    cv::Mat mag, ang;
    calculateGradients(smooth, mag, ang);
    
    // 3. Non-Maxima Suppression
    NMSResult nms = localMaximization(mag, ang);

    // 4. P-Tile Thresholding (Called via our newly separated function)
    cv::Mat binary = applyPTileThreshold(nms, t_high_percent);

    // 5. Encode & Return back to Python
    return encode_image(binary); 
}