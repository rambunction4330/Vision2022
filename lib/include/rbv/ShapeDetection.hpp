#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rbv {
struct Circle {
  cv::Point2f center;
  float radius;
};

cv::Point2f intersection(float rho0, float theat0, float rho1, float theta1);

bool approxNGonHough(const std::vector<cv::Point2f> &contour,
                     std::vector<cv::Point2f> &n_gon, int numSide = 4,
                     int houghThreshold = 50);

bool approxNGonPolyDP(const std::vector<cv::Point2f> &contour,
                      std::vector<cv::Point2f> &n_gon, int numSides = 4,
                      double start = 0, double step = 0.01, double max = 5);

bool approxCircleBounding(const std::vector<cv::Point2f> &contour,
                          Circle &circle);

bool approxCircleHough(const std::vector<cv::Point2f> &contour, Circle &circle);
} // namespace rbv