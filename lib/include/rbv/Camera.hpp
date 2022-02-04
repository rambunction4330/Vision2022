
#pragma once

#include <cmath>
#include <iostream>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace rbv {
class Camera {
public:
  struct Settings {
    Settings() {}
    int captureAPI = cv::VideoCaptureAPIs::CAP_ANY;

    bool autoExposure = true;
    bool autoFocus = true;
    bool autoWB = true;

    double brightness = std::numeric_limits<double>::quiet_NaN();
    double contrast = std::numeric_limits<double>::quiet_NaN();
    double saturation = std::numeric_limits<double>::quiet_NaN();
    double hue = std::numeric_limits<double>::quiet_NaN();
    double gain = std::numeric_limits<double>::quiet_NaN();
    double exposure = std::numeric_limits<double>::quiet_NaN();

    void apply(cv::VideoCapture &capture) const;

    void write(cv::FileStorage &fs) const;
    void read(const cv::FileNode &node);
  };

  Camera() {}

  Camera(int id, const Settings &settings = Settings(),
         const cv::Mat &cameraMatrix = cv::Mat(),
         const cv::Mat &distortion = cv::Mat());

  Camera(int id, const Settings &settings,
         const std::vector<std::vector<cv::Point3f>> &objectPoints,
         const std::vector<std::vector<cv::Point2f>> &imagerPoints,
         cv::Size imageSize);

  const int &getID() const { return id; }
  const Settings &getSettings() const { return settings; }
  const cv::VideoCapture &getVideoCapture() const { return videoCapture; }
  const cv::Mat &getCameraMatrix() const { return cameraMatrix; }
  const cv::Mat &getDistortion() const { return distortion; }

  bool openCapture();
  void releaseCapture() { videoCapture.release(); }
  void updateCaptureSettings() { settings.apply(videoCapture); }
  bool getNextFrame(cv::Mat &frame) { return videoCapture.read(frame); }
  bool getNextFrameUndistorted(cv::Mat &undistorted);

  void undistortFrame(const cv::Mat &src, cv::Mat &dst) const;

  double calibrate(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                   const std::vector<std::vector<cv::Point2f>> &imagerPoints,
                   const cv::Size &imageSize);

  void solvePnP(const std::vector<cv::Point2f> &imagePoints,
                const std::vector<cv::Point3f> &objectPoints, cv::Mat &rvec,
                cv::Mat &tvec) const;

  void projectPoints(const std::vector<cv::Point3f> &worldPoints,
                     std::vector<cv::Point2f> &imagePoints) const;

  void write(cv::FileStorage &fs) const;
  void read(const cv::FileNode &node);

private:
  int id;
  Settings settings;
  cv::VideoCapture videoCapture;
  cv::Mat cameraMatrix;
  cv::Mat distortion;
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

static void write(cv::FileStorage &fs, const std::string &,
                  const Camera::Settings &x) {
  x.write(fs);
}

static void read(const cv::FileNode &node, Camera::Settings &x,
                 const Camera::Settings &default_value = Camera::Settings()) {
  if (node.empty()) {
    x = default_value;
  } else {
    x.read(node);
  }
}
} // namespace rbv