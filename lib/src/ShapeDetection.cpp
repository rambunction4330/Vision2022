#include "rbv/ShapeDetection.hpp"

namespace rbv {
cv::Point2f intersection(float rho0, float theat0, float rho1, float theta1) {
  // Put line into the form sin(theta)*y + cos(theta)*x = rho;
  float s0 = std::sin(theat0), c0 = std::cos(theat0);
  float s1 = std::sin(theta1), c1 = std::cos(theta1);

  // Determinate
  float d = (s0 * c1) - (s1 * c0);

  // If the determinate is zero, the lins are parellel;
  if (d == 0) {
    return {std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
  }

  // Calculates the intersection.
  float x = ((rho0 * c1) - (rho1 * c0)) / d;
  float y = ((rho1 * s0) - (rho0 * s1)) / d;

  return {y, x};
}

bool approxNGonHough(const std::vector<cv::Point2f> &contour,
                     std::vector<cv::Point2f> &n_gon, int numSides,
                     int houghThreshold) {

  // This method requires the contour be strickly convex.
  assert(numSides > 2 && cv::isContourConvex(contour));

  // Find the contour center (or any pointinside the contour).
  cv::Rect bounding = cv::boundingRect(contour);
  cv::Point2f center((bounding.x + bounding.width) / 2,
                     (bounding.y + bounding.height) / 2);
  std::vector<cv::Point2f> centredPoints;
  centredPoints.reserve(contour.size());

  // Center the contour at the origin.
  for (const auto &point : contour) {
    centredPoints.emplace_back(point.x - center.x, point.y - center.y);
  }

  float maxDistance = std::sqrt((bounding.width * bounding.width) +
                                (bounding.height * bounding.height));

  // Find the polygon sides.
  std::vector<cv::Vec3d> lines;
  cv::HoughLinesPointSet(centredPoints, lines, numSides, houghThreshold,
                         -maxDistance, maxDistance, 1, 0, CV_PI, CV_PI / 180);

  // Wrong number of sides
  if (lines.size() != numSides) {
    std::cout << "failed!\n";
    return false;
  }

  // Sort by angle, so that ajacent sides are next to eachother in the vector.
  std::sort(lines.begin(), lines.end(),
            [](const cv::Vec3d &a, const cv::Vec3d &b) { return a[2] < b[2]; });

  // Find side intersections
  for (int i = 0; i < numSides; i++) {
    const cv::Vec3d &a = lines[i];
    const cv::Vec3d &b = lines[(i + 1) % numSides];
    cv::Point2f p = intersection(a[1], a[2], b[1], b[2]);

    if (p == cv::Point2f{std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity()}) {
      return false;
    }

    // Transform points back to thier starting location.
    n_gon.emplace_back(p.x + center.x, p.y + center.y);
  }

  return true;
}

bool approxNGonPolyDP(const std::vector<cv::Point2f> &contour,
                      std::vector<cv::Point2f> &n_gon, int numSides,
                      double start, double step, double max) {

  // Assures that the function won't be caught in aninfinite loop
  assert(numSides > 2 && step > 0);

  while (true) {
    // Aproximatation
    cv::approxPolyDP(contour, n_gon, start, true);

    // Check aproximation
    if (n_gon.size() == numSides) {
      return true;
    }

    // Increment allowabel error
    start += step;

    // Fail consdition (too few sides or too large of a error)
    if (n_gon.size() < numSides || start > max) {
      return false;
    }
  }

  return false;
}

bool approxCircleBounding(const std::vector<cv::Point2f> &contour,
                          Circle &circle) {
  // Farly self-explanatory
  cv::minEnclosingCircle(contour, circle.center, circle.radius);
  return true;
}

bool approxCircleHough(const std::vector<cv::Point2f>& contour, Circle& circle) {

  // Determine image size for contour
  cv::Rect bounding = cv::boundingRect(contour);
  cv::Mat image(bounding.height, bounding.width, CV_8UC1, cv::Scalar{0});

  // draw contour onto image
  cv::drawContours(image, {contour}, 0, {255});

  // Find circles in image
  std::vector<cv::Vec4f> circles;
  cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, 1000);

  // Failed if yoy found no images
  if (circles.size() < 1) {
    return false;
  }

  // Retrieve best circle
  cv::Vec4f &best = *std::max_element(
      circles.begin(), circles.end(),
      [](const cv::Vec4f &a, const cv::Vec4f &b) { return a[3] < b[3]; });
  circle = {{best[0], best[1]}, best[2]};
  return true;
}
} // namespace rbv