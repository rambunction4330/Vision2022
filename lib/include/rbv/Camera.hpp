
#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace rbv {

/*
 * Class encapsulating a usb camera
 */
class Camera {
public:
  Camera() {}

  Camera(int id, const cv::Size &imageSize,
         const cv::Mat &cameraMatrix = cv::Mat(),
         const cv::Mat &distortion = cv::Mat());

  Camera(int id, const cv::Size &imageSize,
         const std::vector<std::vector<cv::Point3f>> &objectPoints,
         const std::vector<std::vector<cv::Point2f>> &imagerPoints);

  const int &getID() const { return id; }
  const cv::VideoCapture &getVideoCapture() const { return videoCapture; }
  const cv::Mat &getCameraMatrix() const { return cameraMatrix; }
  const cv::Mat &getDistortion() const { return distortion; }
  const cv::Size &getImageSize() const { return imageSize; }

  bool openCapture() { return videoCapture.open(id); }
  void releaseCapture() { videoCapture.release(); }
  bool getNextFrame(cv::Mat &frame) { return videoCapture.read(frame); }
  bool getNextFrameUndistorted(cv::Mat &undistorted);

  void undistortFrame(const cv::Mat &src, cv::Mat &dst) const;

  double calibrate(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                   const std::vector<std::vector<cv::Point2f>> &imagerPoints,
                   const cv::Size &imageSize);

  void solvePnP(const std::vector<cv::Point3f> &objectPoints,
                const std::vector<cv::Point2f> &imagePoints, cv::Mat &rvec,
                cv::Mat &tvec) const;

  void projectPoints(const std::vector<cv::Point3f> &objectPoints, cv::Mat rvec,
                     cv::Mat tvec, std::vector<cv::Point2f> &imagePoints) const;

  void write(cv::FileStorage &fs) const;
  void read(const cv::FileNode &node);

private:
  int id;
  cv::VideoCapture videoCapture;
  cv::Mat cameraMatrix;
  cv::Mat distortion;
  cv::Size imageSize;
};

static void write(cv::FileStorage &fs, const std::string &, const Camera &x) {
  x.write(fs);
}

static void read(const cv::FileNode &node, Camera &x,
                 const Camera &default_value = Camera()) {
  if (node.empty()) {
    x = default_value;
  } else {
    x.read(node);
  }
}
} // namespace rbv