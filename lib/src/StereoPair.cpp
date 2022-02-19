#include "rbv/StereoPair.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace rbv {

StereoPair::StereoPair(
    const Camera &leftCamera, const Camera &rightCamera,
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &leftImagePoints,
    const std::vector<std::vector<cv::Point2f>> &rightImagePoints,
    cv::Ptr<cv::StereoMatcher> matcher)
    : leftCamera(leftCamera), rightCamera(rightCamera),
      imageSize(leftCamera.getImageSize()), disparityMatcher(matcher) {

  assert(leftCamera.getImageSize() == rightCamera.getImageSize());

  cv::stereoCalibrate(
      objectPoints, leftImagePoints, rightImagePoints,
      leftCamera.getCameraMatrix(), leftCamera.getDistortion(),
      rightCamera.getCameraMatrix(), rightCamera.getDistortion(), imageSize,
      rotationLeft2Right, tralslationLeft2Right, essentail, fundamental);

  cv::stereoRectify(leftCamera.getCameraMatrix(), leftCamera.getDistortion(),
                    rightCamera.getCameraMatrix(), rightCamera.getDistortion(),
                    imageSize, rotationLeft2Right, tralslationLeft2Right,
                    rectifiedLeft, rectifiedRight, projectionLeft,
                    projectionRight, disparityToDepth);

  cv::initUndistortRectifyMap(
      leftCamera.getCameraMatrix(), leftCamera.getDistortion(), rectifiedLeft,
      projectionLeft, imageSize, CV_16SC2, leftMap1, leftMap2);

  cv::initUndistortRectifyMap(rightCamera.getCameraMatrix(),
                              rightCamera.getDistortion(), rectifiedRight,
                              projectionRight, imageSize, CV_16SC2, rightMap1,
                              rightMap2);
}

bool StereoPair::openCaptures() {
  bool success = leftCamera.openCapture();
  success = success && rightCamera.openCapture();
  return success;
}

void StereoPair::releaseCaptures() {
  leftCamera.releaseCapture();
  rightCamera.releaseCapture();
}

bool StereoPair::getNextFrames(StereoFrame &frames) {
  bool success = leftCamera.getNextFrame(frames.leftImage);
  success = success && rightCamera.getNextFrame(frames.rightImage);
  return success;
}

bool StereoPair::getNextFramesUndistorted(StereoFrame &undistorted) {
  bool success = leftCamera.getNextFrameUndistorted(undistorted.leftImage);
  success =
      success && rightCamera.getNextFrameUndistorted(undistorted.rightImage);
  return success;
}

bool StereoPair::getNextFramesRectified(StereoFrame &rectified) {
  StereoFrame temp;
  if (getNextFrames(temp)) {
    rectifyStereoFrame(temp, rectified);
    return true;
  }
  return false;
}

bool StereoPair::getNextFrameDisparity(cv::Mat &disparity) {
  StereoFrame temp;
  if (getNextFrames(temp)) {
    calculateDisparity(temp, disparity);
    return true;
  }
  return false;
}

bool StereoPair::getNextFrameDepth(cv::Mat &depth) {
  StereoFrame temp;
  if (getNextFrames(temp)) {
    calculateFrameDepth(temp, depth);
    return true;
  }
  return false;
}

void StereoPair::undistortStereoFrame(const StereoFrame &src,
                                      StereoFrame &dst) const {
  cv::undistort(src.leftImage, dst.leftImage, leftCamera.getCameraMatrix(),
                leftCamera.getDistortion());
  cv::undistort(src.rightImage, dst.rightImage, rightCamera.getCameraMatrix(),
                rightCamera.getDistortion());
}

void StereoPair::rectifyStereoFrame(const StereoFrame &src,
                                    StereoFrame &dst) const {
  cv::remap(src.leftImage, dst.leftImage, leftMap1, leftMap2, cv::INTER_LINEAR,
            cv::BORDER_DEFAULT, cv::Scalar());
  cv::remap(src.rightImage, dst.rightImage, rightMap1, rightMap2,
            cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());
}

void StereoPair::calculateDisparityRectified(const StereoFrame &rectified,
                                             cv::Mat &disparity) const {
  StereoFrame temp;
  cv::cvtColor(rectified.leftImage, temp.leftImage, cv::COLOR_BGR2GRAY);
  cv::cvtColor(rectified.rightImage, temp.rightImage, cv::COLOR_BGR2GRAY);
  disparityMatcher->compute(temp.rightImage, temp.leftImage, disparity);
}

void StereoPair::calculateDisparity(const StereoFrame &src,
                                    cv::Mat &disparity) const {
  StereoFrame temp;
  rectifyStereoFrame(src, temp);
  calculateDisparityRectified(temp, disparity);
}

void StereoPair::calculateFrameDepth(const cv::Mat &disparity,
                                     cv::Mat &depth) const {
  cv::reprojectImageTo3D(disparity, depth, disparityToDepth);
}

void StereoPair::calculateFrameDepthRectified(const StereoFrame &rectified,
                                              cv::Mat &depth) const {
  cv::Mat disparity;
  calculateDisparityRectified(rectified, disparity);
  cv::reprojectImageTo3D(disparity, depth, disparityToDepth);
}

void StereoPair::calculateFrameDepth(const StereoFrame &src,
                                     cv::Mat &depth) const {
  cv::Mat disparity;
  calculateDisparity(src, disparity);
  cv::reprojectImageTo3D(disparity, depth, disparityToDepth);
}

void StereoPair::triangulatePoints(
    const std::vector<cv::Point2f> &leftPoints,
    const std::vector<cv::Point2f> &rightPoints,
    std::vector<cv::Point3f> &worldPoints) const {
  cv::Mat points4D;
  cv::triangulatePoints(projectionLeft, projectionRight, leftPoints,
                        rightPoints, points4D);
  cv::convertPointsFromHomogeneous(points4D, worldPoints);
}

void StereoPair::write(cv::FileStorage &fs) const {
  fs << "{"
     << "LeftCamera" << leftCamera << "RightCamera" << rightCamera
     << "RotationLeft2Right" << rotationLeft2Right << "TralslationLeft2Right"
     << tralslationLeft2Right << "Essentail" << essentail << "Fundamental"
     << fundamental << "}";
}

void StereoPair::read(const cv::FileNode &node) {
  node["LeftCamera"] >> leftCamera;
  node["RightCamera"] >> rightCamera;
  node["RotationLeft2Right"] >> rotationLeft2Right;
  node["TralslationLeft2Right"] >> tralslationLeft2Right;
  node["Essentail"] >> essentail;
  node["Fundamental"] >> fundamental;

  assert(leftCamera.getImageSize() == rightCamera.getImageSize());
  imageSize = leftCamera.getImageSize();

  cv::stereoRectify(leftCamera.getCameraMatrix(), leftCamera.getDistortion(),
                    rightCamera.getCameraMatrix(), rightCamera.getDistortion(),
                    imageSize, rotationLeft2Right, tralslationLeft2Right,
                    rectifiedLeft, rectifiedRight, projectionLeft,
                    projectionRight, disparityToDepth);

  cv::initUndistortRectifyMap(
      leftCamera.getCameraMatrix(), leftCamera.getDistortion(), rectifiedLeft,
      projectionLeft, imageSize, CV_16SC2, leftMap1, leftMap2);

  cv::initUndistortRectifyMap(rightCamera.getCameraMatrix(),
                              rightCamera.getDistortion(), rectifiedRight,
                              projectionRight, imageSize, CV_16SC2, rightMap1,
                              rightMap2);
}
} // namespace rbv