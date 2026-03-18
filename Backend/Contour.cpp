#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <vector>
#include "image_utils.hpp"

namespace py = pybind11;

py::bytes detect_lines(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);
    
    if (edgeImg.channels() == 3) {
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    }
    // threshold just in case
    cv::threshold(edgeImg, edgeImg, 127, 255, cv::THRESH_BINARY);
    
    std::vector<cv::Vec4i> lines;
    // probabilistic hough lines
    cv::HoughLinesP(edgeImg, lines, 1, CV_PI/180, 50, 50, 10);
    
    if (drawImg.channels() == 1) {
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);
    }
    
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(drawImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    
    return encode_image(drawImg);
}

py::bytes detect_circles(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);
    
    if (edgeImg.channels() == 3) {
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    }
    
    std::vector<cv::Vec3f> circles;
    
    // HoughCircles uses Sobel gradients internally. Passing a raw binary edge image 
    // produces noisy gradients. A slight blur helps it compute correct center directions.
    cv::Mat blurredEdge;
    cv::GaussianBlur(edgeImg, blurredEdge, cv::Size(5, 5), 1.5);

    // param1 is the canny highest threshold.
    // param2 is accumulator threshold. Increasing to 80 to strictly require more circular votes.
    // minRadius=15 prevents detecting tiny noisy blobs as circles.
    cv::HoughCircles(blurredEdge, circles, cv::HOUGH_GRADIENT, 1, blurredEdge.rows/8, 200, 80, 15, 0);
    
    if (drawImg.channels() == 1) {
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);
    }
    
    for(size_t i = 0; i < circles.size(); i++) {
        cv::Vec3i c = circles[i];
        cv::circle(drawImg, cv::Point(c[0], c[1]), c[2], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(drawImg, cv::Point(c[0], c[1]), 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
    
    return encode_image(drawImg);
}

py::bytes detect_ellipses(const py::bytes& drawBytes, const py::bytes& edgeBytes) {
    cv::Mat drawImg = decode_image(drawBytes);
    cv::Mat edgeImg = decode_image(edgeBytes);
    
    if (edgeImg.channels() == 3) {
        cv::cvtColor(edgeImg, edgeImg, cv::COLOR_BGR2GRAY);
    }
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edgeImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    if (drawImg.channels() == 1) {
        cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);
    }
    
    for(size_t i = 0; i < contours.size(); i++) {
        // Require at least 40 points to form a reliable closed shape
        if(contours[i].size() >= 40) { 
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
            
            double w = ellipse.size.width;
            double h = ellipse.size.height;
            
            // Filter out tiny ellipses and extremely large ones
            if (w < 15 || h < 15 || w > edgeImg.cols * 1.5 || h > edgeImg.rows * 1.5) {
                continue;
            }
            
            // Compare the actual area of the contour to the mathematical area of the fitted ellipse.
            // A perfect closed ellipse has a contourArea equal to its mathematical area.
            // An open curve / squiggle will have a contourArea close to 0.
            double contour_area = cv::contourArea(contours[i]);
            double ellipse_area = CV_PI * (w / 2.0) * (h / 2.0);
            
            // Allow a 20% margin of error for imperfectly drawn ellipses.
            if (contour_area < 0.80 * ellipse_area) {
                continue; // Reject open lines and non-elliptical closed shapes
            }
            
            // Draw the accepted ellipse
            cv::ellipse(drawImg, ellipse, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }
    
    return encode_image(drawImg);
}