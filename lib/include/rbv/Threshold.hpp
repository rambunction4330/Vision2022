#include <iostream>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rbv {
  /**
   * @brief Structer holding data to threshold an image.
   * 
   * Holds the high and low values for thresholding in the hsv color space, as
   * well as varables to determin the blur size and morphology matrices to
   * remove noise and close holes.
   * 
   * @see thresholdImage
   */
  struct Threshold {
    cv::Scalar_<int> high = {180, 255, 255}; /**< The upper bound of the hsv threshold. */
    cv::Scalar_<int> low  = {  0,   0,   0}; /**< The lower bound of the hsv threshold. */

    int blurSize = 15; /**< The size of the square blur filter to remove image noise. */

    cv::Mat openMatrix  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15)); /**< The kernal to remove noise in thresholding. */
    cv::Mat closeMatrix = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15)); /**< The kernal to close holes in thresholding. */

    int& highH() { return high[0]; } /**< Gets a refrece to the hue part of the high scalar. */
    int& lowH()  { return low[0];  } /**< Gets a refrece to the hue part of the low scalar. */
    int& highS() { return high[1]; } /**< Gets a refrece to the saturation part of the high scalar. */
    int& lowS()  { return low[1];  } /**< Gets a refrece to the saturation part of the low scalar. */
    int& highV() { return high[2]; } /**< Gets a refrece to the value part of the high scalar. */
    int& lowV()  { return low[2];  } /**< Gets a refrece to the value part of the low scalar. */

    void setHighH(int value) { highH() = std::clamp(value, lowH() + 1, 180); } /**< Sets the hue part of the high scalar with bound checks. */
    void setLowH(int value)  { lowH()  = std::clamp(value,  0, highH() - 1); } /**< Sets the hue part of the low scalar with bound checks. */
    void setHighS(int value) { highV() = std::clamp(value, lowS() + 1, 255); } /**< Sets the saturation part of the high scalar with bound checks. */
    void setLowS(int value)  { lowV()  = std::clamp(value,  0, highS() - 1); } /**< Sets the saturation part of the low scalar with bound checks. */
    void setHighV(int value) { highV() = std::clamp(value, lowV() + 1, 255); } /**< Sets the value part of the high scalar with bound checks. */
    void setLowV(int value)  { lowV()  = std::clamp(value,  0, highV() - 1); } /**< Sets the value part of the low scalar with bound checks. */

    void write(cv::FileStorage& fs) const {
      fs << "{" << "High" << high << "Low" << low << "BlurSize" << blurSize << "OpenMatrix" << openMatrix << "CloseMatrix" << closeMatrix << "}";
    }

    void read(const cv::FileNode& node) {
      node["High"] >> high;
      node["Low"] >> low;
      node["BlurSize"] >> blurSize;
      node["OpenMatrix"] >> openMatrix;
      node["CloseMatrix"] >> closeMatrix;
    }

    void apply(cv::Mat& input, cv::Mat& output) {
      cv::blur(input, output, {blurSize, blurSize});
      cv::cvtColor(output, output, cv::COLOR_BGR2HSV);
      cv::inRange(output, cv::Scalar(low), cv::Scalar(high), output);
      cv::morphologyEx(output, output, cv::MORPH_OPEN, openMatrix);
      cv::morphologyEx(output, output, cv::MORPH_CLOSE, closeMatrix);
    }
  };

  static void write(cv::FileStorage& fs, const std::string&, const Threshold& x) { 
    x.write(fs);
  }
  
  static void read(const cv::FileNode& node, Threshold& x, const Threshold& default_value = Threshold()) {
    if(node.empty()) {
      x = default_value;
    } else {
      x.read(node);
    }
  }
}