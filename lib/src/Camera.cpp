#include "rbv/Camera.hpp"

#include <assert.h>

#include <opencv2/calib3d.hpp>

namespace rbv {

void Camera::Settings::apply(cv::VideoCapture &cap) const {
  assert(cap.isOpened());

  if (!cap.set(cv::CAP_PROP_AUTO_EXPOSURE, autoExposure ? 1 : 0))
    std::cerr << "Camera does not support `cv::CAP_PROP_AUTO_EXPOSURE`\n";
  if (!cap.set(cv::CAP_PROP_AUTOFOCUS, autoFocus ? 1 : 0))
    std::cerr << "Camera does not support `cv::CAP_PROP_AUTOFOCUS`\n";
  if (!cap.set(cv::CAP_PROP_AUTO_WB, autoWB ? 1 : 0))
    std::cerr << "Camera does not support `cv::CAP_PROP_AUTO_WB`\n";

  if (!std::isnan(brightness) && !cap.set(cv::CAP_PROP_BRIGHTNESS, brightness))
    std::cerr << "Camera does not support `cv::CAP_PROP_BRIGHTNESS`\n";
  if (!std::isnan(contrast) && cap.set(cv::CAP_PROP_CONTRAST, contrast))
    std::cerr << "Camera does not support `cv::CAP_PROP_CONTRAST`\n";
  if (!std::isnan(saturation) && cap.set(cv::CAP_PROP_SATURATION, saturation))
    std::cerr << "Camera does not support `cv::CAP_PROP_SATURATION`\n";
  if (!std::isnan(hue) && cap.set(cv::CAP_PROP_HUE, hue))
    std::cerr << "Camera does not support `cv::CAP_PROP_HUE`\n";
  if (!std::isnan(gain) && cap.set(cv::CAP_PROP_GAIN, gain))
    std::cerr << "Camera does not support `cv::CAP_PROP_GAIN`\n";
  if (!std::isnan(exposure) && cap.set(cv::CAP_PROP_EXPOSURE, exposure))
    std::cerr << "Camera does not support `cv::CAP_PROP_EXPOSURE`\n";
}

void Camera::Settings::write(cv::FileStorage &fs) const {
  fs << "{"
     << "CaptureAPI" << captureAPI << "AutoExposure" << autoExposure
     << "AutoFocus" << autoFocus << "AutoWB" << autoWB << "Brightness"
     << brightness << "Contrast" << contrast << "Saturation" << saturation
     << "Hue" << hue << "Gain" << gain << "Exposure" << exposure << "}";
}

void Camera::Settings::read(const cv::FileNode &node) {
  node["CaptureAPI"] >> captureAPI;
  node["AutoExposure"] >> autoExposure;
  node["AutoFocus"] >> autoFocus;
  node["AutoWB"] >> autoWB;
  node["Brightness"] >> brightness;
  node["Contrast"] >> contrast;
  node["Saturation"] >> saturation;
  node["Hue"] >> hue;
  node["Gain"] >> gain;
  node["Exposure"] >> exposure;
}

Camera::Camera(int id, const Settings &settings, const cv::Mat &cameraMatrix,
               const cv::Mat &distortion)
    : id(id), settings(settings), cameraMatrix(cameraMatrix),
      distortion(distortion) {}

Camera::Camera(int id, const Settings &settings,
               const std::vector<std::vector<cv::Point3f>> &objectPoints,
               const std::vector<std::vector<cv::Point2f>> &imagerPoints,
               cv::Size imageSize)
    : id(id), settings(settings), cameraMatrix(cv::Mat()),
      distortion(cv::Mat()) {
  calibrate(objectPoints, imagerPoints, imageSize);
}

bool Camera::openCapture() {
  if (videoCapture.open(id, settings.captureAPI)) {
    settings.apply(videoCapture);
    return true;
  }
  return false;
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
     << distortion << "Settings" << settings << "}";
}

void Camera::read(const cv::FileNode &node) {
  node["ID"] >> id;
  node["CameraMatrix"] >> cameraMatrix;
  node["Distortion"] >> distortion;
  node["Settings"] >> settings;
}
} // namespace rbv