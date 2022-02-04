#include "rbv/HSVThreshold.hpp"

namespace rbv {
HSVThreshold::HSVThreshold()
    : low(0, 0, 0), high(180, 255, 255), blurSize(0), openMatrix(cv::Mat()),
      closeMatrix(cv::Mat()) {}

HSVThreshold::HSVThreshold(cv::Scalar_<int> l, cv::Scalar_<int> h, int bSize,
                           int openSize, cv::MorphShapes openShape,
                           int closeSize, cv::MorphShapes closeShape)
    : low(l), high(h), blurSize(bSize), openSize(openSize),
      closeSize(closeSize), openShape(openShape), closeShape(closeShape) {
  if (openSize > 0) {
    openMatrix = cv::getStructuringElement(openShape, {openSize, openSize});
  }

  if (closeSize > 0) {
    closeMatrix = cv::getStructuringElement(closeShape, {closeSize, closeSize});
  }
}

void HSVThreshold::apply(cv::Mat &input, cv::Mat &output) const {
  cv::Mat temp = input;
  if (blurSize > 0) {
    cv::blur(temp, temp, {blurSize, blurSize});
  }

  cv::cvtColor(temp, temp, cv::COLOR_BGR2HSV);
  cv::inRange(temp, cv::Scalar(low), cv::Scalar(high), temp);

  if (!openMatrix.empty()) {
    cv::morphologyEx(temp, temp, cv::MORPH_OPEN, openMatrix);
  }

  if (!closeMatrix.empty()) {
    cv::morphologyEx(temp, temp, cv::MORPH_CLOSE, closeMatrix);
  }

  temp.copyTo(output);
}

void HSVThreshold::write(cv::FileStorage &fs) const {
  fs << "{"
     << "High" << high << "Low" << low << "BlurSize" << blurSize << "OpenSize"
     << openSize << "CloseSize" << closeSize << "OpenShape" << openShape
     << "CloseShape" << closeShape << "}";
}

void HSVThreshold::read(const cv::FileNode &node) {
  node["High"] >> high;
  node["Low"] >> low;
  node["BlurSize"] >> blurSize;
  node["OpenSize"] >> openMatrix;
  node["CloseSize"] >> closeMatrix;
  node["OpenShape"] >> openShape;
  node["CloseShape"] >> closeShape;
}
} // namespace rbv