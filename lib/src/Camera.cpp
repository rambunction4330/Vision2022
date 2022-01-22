#include "rbv/Camera.hpp"

namespace rbv {
  void Camera::write(cv::FileStorage& fs) const {
    fs << "{" << "ID" << id 
       << "CameraMatrix" << cameraMatrix 
       << "Distortion" << distortion 
       << "WorldToCamers" << worldToCamera << "}";
  }

  void Camera::read(const cv::FileNode& node) {
    node["ID"] >> id;
    node["CameraMatrix"] >> cameraMatrix;
    node["Distortion"] >> distortion;
    node["WorldToCamers"] >> worldToCamera;
  }
}