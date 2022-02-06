#include "rbv/Camera.hpp"

#include <assert.h>

#include <opencv2/calib3d.hpp>

namespace rbv {

Camera::Camera(int id, const cv::Size &imageSize, const cv::Mat &cameraMatrix,
               const cv::Mat &distortion)
    : id(id), cameraMatrix(cameraMatrix), distortion(distortion),
      imageSize(imageSize) {}

Camera::Camera(int id, const cv::Size &imageSize,
               const std::vector<std::vector<cv::Point3f>> &objectPoints,
               const std::vector<std::vector<cv::Point2f>> &imagerPoints)
    : id(id), cameraMatrix(cv::Mat()), distortion(cv::Mat()),
      imageSize(imageSize) {
  calibrate(objectPoints, imagerPoints, imageSize);
}

bool Camera::getNextFrameUndistorted(cv::Mat &undistorted) {
  cv::Mat temp;
  if (getNextFrame(temp)) {
    undistortFrame(temp, undistorted);
    return false;
  }
  return false;
}

void Camera::undistortFrame(const cv::Mat &src, cv::Mat &dst) const {
  cv::undistort(src, dst, cameraMatrix, distortion);
}

double
Camera::calibrate(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                  const std::vector<std::vector<cv::Point2f>> &imagerPoints,
                  const cv::Size &imageSize) {
  return cv::calibrateCamera(objectPoints, imagerPoints, imageSize,
                             cameraMatrix, distortion, cv::noArray(),
                             cv::noArray());
}

void Camera::solvePnP(const std::vector<cv::Point2f> &imagePoints,
                      const std::vector<cv::Point3f> &objectPoints,
                      cv::Mat &rvec, cv::Mat &tvec) const {
  cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distortion, rvec, tvec);
}

void Camera::write(cv::FileStorage &fs) const {
  fs << "{"
     << "ID" << id << "CameraMatrix" << cameraMatrix << "Distortion"
     << distortion << "ImageSize" << imageSize << "}";
}

void Camera::read(const cv::FileNode &node) {
  node["ID"] >> id;
  node["CameraMatrix"] >> cameraMatrix;
  node["Distortion"] >> distortion;
  node["ImageSize"] >> imageSize;
}
} // namespace rbv