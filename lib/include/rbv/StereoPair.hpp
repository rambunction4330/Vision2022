
#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "rbv/Camera.hpp"

namespace rbv {

struct StereoFrame {
  cv::Mat leftImage, rightImage;
};

class StereoPair {
public:
  StereoPair() {}

  StereoPair(const Camera &leftCamera, const Camera &rightCamera,
             const std::vector<std::vector<cv::Point3f>> &objectPoints,
             const std::vector<std::vector<cv::Point2f>> &leftImagePoints,
             const std::vector<std::vector<cv::Point2f>> &rightImagePoints,
             cv::Ptr<cv::StereoMatcher> matcher = cv::StereoBM::create());

  const Camera &getLeftCamera() { return leftCamera; }
  const Camera &getRightCamera() { return rightCamera; }

  bool openCaptures();
  void releaseCaptures();
  bool getNextFrames(StereoFrame &frames);
  bool getNextFramesUndistorted(StereoFrame &undistorted);
  bool getNextFramesRectified(StereoFrame &rectified);
  bool getNextFrameDisparity(cv::Mat &disparity);
  bool getNextFrameDepth(cv::Mat &depth);

  void undistortStereoFrame(const StereoFrame &src, StereoFrame &dst) const;
  void rectifyStereoFrame(const StereoFrame &src, StereoFrame &dst) const;
  void calculateDisparityRectified(const StereoFrame &rectified,
                                   cv::Mat &disparity) const;
  void calculateDisparity(const StereoFrame &src, cv::Mat &disparity) const;
  void calculateFrameDepth(const cv::Mat &disparity, cv::Mat &depth) const;
  void calculateFrameDepthRectified(const StereoFrame &rectified,
                                    cv::Mat &depth) const;
  void calculateFrameDepth(const StereoFrame &frame, cv::Mat &depth) const;

  void triangulatePoints(const std::vector<cv::Point2f> &rightPoints,
                         const std::vector<cv::Point2f> &leftPoints,
                         std::vector<cv::Point3f> &worldPoints) const;

  void write(cv::FileStorage &fs) const;
  void read(const cv::FileNode &node);

private:
  rbv::Camera leftCamera, rightCamera;
  cv::Size imageSize;
  cv::Mat rotationLeft2Right, tralslationLeft2Right;
  cv::Mat essentail, fundamental;
  cv::Mat rectifiedLeft, rectifiedRight;
  cv::Mat projectionLeft, projectionRight;
  cv::Mat disparityToDepth;
  cv::Mat leftMap1, leftMap2;
  cv::Mat rightMap1, rightMap2;
  cv::Ptr<cv::StereoMatcher> disparityMatcher = cv::StereoBM::create();
};

static void write(cv::FileStorage &fs, const std::string &,
                  const StereoPair &x) {
  x.write(fs);
}

static void read(const cv::FileNode &node, StereoPair &x,
                 const StereoPair &default_value = StereoPair()) {
  if (node.empty()) {
    x = default_value;
  } else {
    x.read(node);
  }
}
} // namespace rbv