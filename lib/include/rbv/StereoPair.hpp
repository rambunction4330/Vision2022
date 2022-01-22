
#pragma once

#include <opencv2/core.hpp>

#include "rbv/Camera.hpp"

namespace rbv {
  struct StereoPair {
    rbv::Camera camera1, camera2;
    cv::Mat rotation1To2, tralslation1To2;
    cv::Mat essentail, fundamental;
    cv::Mat rectified1, rectified2;
    cv::Mat projection1, projection2;
    cv::Mat disparityToDepth;

    void write(cv::FileStorage& fs) const {
      fs << "{" << "Camera1" << camera1 << "Camera2" << camera2 
         << "Rotation1To2" << rotation1To2 << "Tralslation1To2" << tralslation1To2 
         << "Essentail" << essentail << "Fundamental" << fundamental
         << "Rectified1" << rectified1 << "Rectified2" << rectified2
         << "Projection1" << projection1 << "Projection2" << projection2
         << "DisparityToDepth" << disparityToDepth << "}";
    }

    void read(const cv::FileNode& node) {
      node["Camera1"] >> camera1;
      node["Camera2"] >> camera2;
      node["Rotation1To2"] >> rotation1To2;
      node["Tralslation1To2"] >> tralslation1To2;
      node["Essentail"] >> essentail;
      node["Fundamental"] >> fundamental;
      node["Rectified1"] >> rectified1;
      node["Rectified2"] >> rectified2;
      node["Projection1"] >> projection1;
      node["Projection2"] >> projection2;
      node["DisparityToDepth"] >> disparityToDepth;
    }
  };

  static void write(cv::FileStorage& fs, const std::string&, const StereoPair& x) { 
    x.write(fs);
  }
  
  static void read(const cv::FileNode& node, StereoPair& x, const StereoPair& default_value = StereoPair()) {
    if(node.empty()) {
      x = default_value;
    } else {
      x.read(node);
    }
  }
}