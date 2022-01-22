
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace rbv {
  struct Camera {
    int id;
    cv::Mat cameraMatrix;
    cv::Mat distortion;
    cv::Mat worldToCamera;

    void write(cv::FileStorage& fs) const;
    void read(const cv::FileNode& node);
  };

  static void write(cv::FileStorage& fs, const std::string&, const Camera& x) { 
    x.write(fs);
  }
  
  static void read(const cv::FileNode& node, Camera& x, const Camera& default_value = Camera()) {
    if(node.empty()) {
      x = default_value;
    } else {
      x.read(node);
    }
  }
}