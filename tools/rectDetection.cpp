#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <rbv/HSVThreshold.hpp>
#include <rbv/Camera.hpp>
#include <rbv/StereoPair.hpp>
#include <rbv/ShapeDetection.hpp>

int main(int argc, char *argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys for argument parsing
  const std::string keys =
      "{ h ? help usage |                 | prints this message            }"
      "{ c calibration  | calibration.xml | calibration file               }"
      "{ t threshold    |  threshold.xml  | flag if stereo calibration     }"
      "{ s rectSize     |    (50.8,127)   | dimensions of the chessboard   }"
      "{ m method       |    approxPoly   | Method of rect detection       }"
      "{ r rotation     |        30.0      | camera rotation from center    }";

  // Parser object
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("\nVision2022 v22.0.0 cameraCalibration"
               "\nTool to calibrate a camera\n");

  // Show help if help is flagged.
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  // Get arguments from the parser
  std::string calibrationFile = parser.get<std::string>("calibration");
  std::string thresholdFile = parser.get<std::string>("threshold");
  std::string method = parser.get<std::string>("method");
  double rotation = parser.get<double>("rotation");

  cv::Matx33d rotationMat( cos(rotation * CV_PI / 180), 0, sin(rotation * CV_PI / 180), 
                                        0,              1,                    0,
                          -sin(rotation * CV_PI / 180), 0, cos(rotation * CV_PI / 180));
  
  cv::Mat rotationVec, transVec = cv::Mat::zeros({1, 3}, CV_64F);
  cv::Rodrigues(rotationMat.inv(), rotationVec);

  cv::Size2f rectSize;
  if (sscanf(parser.get<std::string>("rectSize").c_str(), "(%f,%f)",
             &rectSize.width, &rectSize.height) != 2) {
    std::cerr << "Invalid format for argument 'rectSize'\n";
    return 0;
  }

  // Cheack for errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // Get files
  cv::FileStorage storage;
  rbv::Camera camera;
  if (calibrationFile != "" && storage.open(calibrationFile, cv::FileStorage::READ)) {
    storage["Camera"] >> camera;
  } else {
    std::cerr << "Error opening output file: '" << calibrationFile << "'\n";
    return 0;
  }
  storage.release();

  rbv::HSVThreshold threshold;
  if (thresholdFile != "" && storage.open(thresholdFile, cv::FileStorage::READ)) {
    storage["HSVThreshold"] >> threshold;
  } else {
    std::cerr << "Error opening output file: '" << thresholdFile << "'\n";
    return 0;
  }
  storage.release();

  std::vector<cv::Point3f> objectPoints;
  objectPoints.push_back({-rectSize.width/2, rectSize.height/2, 0});
  objectPoints.push_back({rectSize.width/2, rectSize.height/2, 0});
  objectPoints.push_back({rectSize.width/2, -rectSize.height/2, 0});
  objectPoints.push_back({-rectSize.width/2, -rectSize.height/2, 0});

  // Start capture
  if (!camera.openCapture()) {
    std::cerr << "Could not open camera with id: " << camera.getID() << "\n"; 
    return 0;
  }

  cv::Mat frame, thresh, display;
  bool showThresh = false, show3D = false;
  while (true) {
    // Get next frame
    camera.getNextFrame(frame);

    // Check for connection
    if (frame.empty()) {
      std::cerr << "Lost connection to camera with id:" << camera.getID() << "\n";
      break;
    }

    // Threshold image
    threshold.apply(frame, thresh);

    // Setup display
    if (showThresh) {
      cv::cvtColor(thresh, display, cv::COLOR_GRAY2BGR);
    } else {
      frame.copyTo(display);
    }

    // Find contours
    std::vector<std::vector<cv::Point2i>> contoursI;
    std::vector<std::vector<cv::Point2f>> contours;
    cv::findContours(thresh, contoursI, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto contour: contoursI) {
      std::vector<cv::Point2f> temp;
      for (auto point: contour) {
        temp.push_back(point);
      }
      contours.push_back(temp);
    }

    // Interate over contours
    for (auto contour : contours) {
      // Check rect size
      if (cv::contourArea(contour) > 50) {
        // Approximate rect by given emthod
        std::vector<cv::Point2f> approxRect;
        bool foundRect;
        if (method == "approxPoly") {
          foundRect = rbv::approxNGonPolyDP(contour, approxRect, 4, 0.0, 0.01, 100);
        } else if (method == "hough") {
          cv::convexHull(contour, approxRect);
          foundRect = rbv::approxNGonHough(approxRect, approxRect);
        } else {
          foundRect = rbv::approxNGonPolyDP(contour, approxRect);
        }

        // Draw if found
        if (foundRect) {
          if (show3D) {
            std::sort(approxRect.begin(), approxRect.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

            if (approxRect[0].y > approxRect[1].y) {
              cv::Point2f temp = approxRect[0];
              approxRect[0] = approxRect[1];
              approxRect[1] = temp;
            }

            if (approxRect[2].y < approxRect[3].y) {
              cv::Point2f temp = approxRect[2];
              approxRect[2] = approxRect[3];
              approxRect[3] = temp;
            };
            
            cv::Mat rvec, tvec;
            camera.solvePnP(objectPoints, approxRect, rvec, tvec);

            std::vector<cv::Point2f> reprojection;
            camera.projectPoints(objectPoints, rvec, tvec, reprojection);

            cv::drawFrameAxes(display, camera.getCameraMatrix(), camera.getDistortion(), rvec, tvec, 20);

            cv::composeRT(rvec, tvec, rotationVec, transVec, rvec, tvec);

            std::string lable = "(" + std::to_string(tvec.at<double>(0,0)/1000) + ", " + std::to_string(tvec.at<double>(0,1)/1000) + ", " + std::to_string(tvec.at<double>(0,2)/1000) + ")";
            cv::putText(display, lable, approxRect[0], 0, 0.5, {255,0,0  });

            for (int j = 0; j < 4; j++) {
              cv::line(display, reprojection[j], reprojection[(j+1)%4], {0, 0, 255}, 2);
            }

          } else {
            for (int j = 0; j < 4; j++) {
              cv::line(display, approxRect[j], approxRect[(j+1)%4], {255, 0, 0}, 2);
            }
          }
        }
      }
    }

    // Draw rect
    cv::imshow("Target Detection", display);

    // Prosess keypresses
    char key = cv::waitKey(30);

    // Toggle view settings
    showThresh = (key == 't') ? !showThresh : showThresh;
    show3D = (key == 'd') ? !show3D : show3D;
 
    // Quit
    if (key == 27 || key == 'q') {
      break;
    }
  }

  // Cleanup
  cv::waitKey(1);
  camera.releaseCapture();
  cv::destroyAllWindows();
  cv::waitKey(1);
  return 0;
}