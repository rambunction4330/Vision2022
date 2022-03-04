#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
      "{ h ? help usage |                 | prints this message  }"
      "{ c calibration  | calibration.xml | calibration file     }"
      "{ t threshold    |  threshold.xml  | threshold file       }"
      "{ s rectSize     |    (127, 50.8)  | dimensions of target }"
      "{ a angle        |      0.0        | camera angle (pitch) }"
      "{ v visual       |                 | flag to create gui   }"
      "{ T team         |       4330      | team number          }";

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
  bool useVisual = parser.has("visual");

  cv::Size2f rectSize;
  if (sscanf(parser.get<std::string>("rectSize").c_str(), "(%f,%f)",
             &rectSize.width, &rectSize.height) != 2) {
    std::cerr << "Invalid format for argument 'rectSize'\n";
    return 0;
  }

  // Generate object points
  std::vector<cv::Point3f> objectPoints;
  objectPoints.emplace_back(-rectSize.height/2,  rectSize.width/2, 0.0);
  objectPoints.emplace_back( rectSize.height/2,  rectSize.width/2, 0.0);
  objectPoints.emplace_back( rectSize.height/2, -rectSize.width/2, 0.0);
  objectPoints.emplace_back(-rectSize.height/2, -rectSize.width/2, 0.0);

  // Get Rodrigues vector for rotation
  double angle = parser.get<double>("angle");
  cv::Matx33d rotationMat(1,             0,                         0, 
                          0, cos(angle * CV_PI / 180), -sin(angle * CV_PI / 180),
                          0, sin(angle * CV_PI / 180),  cos(angle * CV_PI / 180));

  cv::Mat rotationVec;
  cv::Rodrigues(rotationMat, rotationVec);

  // Cheack for errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  /*********************
   * NetworkTables Setup
   *********************/
  auto inst = nt::NetworkTableInstance::GetDefault();
  inst.StartClientTeam(parser.get<std::string>("team"));
  auto table = inst.GetTable("vision");
  nt::NetworkTableEntry distanceEntry = table->GetEntry("distance");
  nt::NetworkTableEntry heightEntry = table->GetEntry("height");
  nt::NetworkTableEntry deltaAngleEntry = table->GetEntry("deltaAngle");
  nt::NetworkTableEntry deltaAngleEntry = table->GetEntry("isHubVisible");

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

  // Start capture
  if (!camera.openCapture()) {
    std::cerr << "Could not open camera with id: " << camera.getID() << "\n"; 
    return 0;
  }

  cv::Mat frame, thresh, display;
  bool showThresh = false, show3D = false, printData = false;
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

    cv::dilate(thresh, thresh, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));

    // Setup display
    if (useVisual) {
      if (showThresh) {
        cv::cvtColor(thresh, display, cv::COLOR_GRAY2BGR);
      } else {
        frame.copyTo(display);
      }
    }

    // Find contours
    std::vector<std::vector<cv::Point2i>> intContours;
    cv::findContours(thresh, intContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Convert contours to the proper data type
    std::vector<std::vector<cv::Point2f>> contours;
    for (auto contour: intContours) {
      std::vector<cv::Point2f> temp;
      for (auto point: contour) {
        temp.push_back(point);
      }
      contours.push_back(temp);
    }

    // Vector to hold "good" rects and an arbitrary score
    std::vector<std::vector<cv::Point2f>> goodRects;
    std::vector<double> rectScores;
    
    // Make sure there are enough contours (but not too many)
    if (contours.size() > 0 && contours.size() < 7) {
      for (const auto& contour: contours) {
        // Get boundign rect to determine "rectness"
        cv::RotatedRect rect = cv::minAreaRect(contour);
        double rectArea = rect.size.width * rect.size.height;
        double contourArea = cv::contourArea(contour);

        // Check rect size and "rectness"
        if (contourArea > 15 && contourArea/rectArea > 0.7) {

          // Convex hull
          std::vector<cv::Point2f> hull;
          cv::convexHull(contour, hull);

          // Approximate rect
          std::vector<cv::Point2f> approxRect;
          bool foundRect;
          foundRect = rbv::approxNGonPolyDP(hull, approxRect);

          if (foundRect) {
            // Reorder points
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

            if (useVisual && !show3D) {
              // Draw rect
              for (int j = 0; j < 4; j++) {
                cv::line(display, approxRect[j], approxRect[(j+1)%4], {255, 0, 0}, 2);
              }
            }

            // Add rect and score to list;
            goodRects.push_back(approxRect);
            rectScores.push_back(cv::contourArea(approxRect) / cv::contourArea(hull));
          }
        }
      }
    }

    // Find best rect
    int bestIndex = -1;
    double bestScore = 0.0;
    for (int i = 0; i < rectScores.size(); i++) {
      bestIndex = (rectScores[i] > bestScore) ? i : bestIndex;
    }

    if (bestIndex != -1) {
      // Find the target position
      cv::Mat rvec, tvec;
      camera.solvePnP(objectPoints, goodRects[bestIndex], rvec, tvec);

      if (useVisual && show3D) {
        // Reproject Points
        std::vector<cv::Point2f> reprojection;
        camera.projectPoints(objectPoints, rvec, tvec, reprojection);

        // Draw reprojection
        for (int j = 0; j < 4; j++) {
          cv::line(display, reprojection[j], reprojection[(j+1)%4], {0, 0, 255}, 2);
        }

        // Draw frame axis
        cv::drawFrameAxes(display, camera.getCameraMatrix(), camera.getDistortion(), rvec, tvec, 20);
      }

      // Add in camera angle
      cv::composeRT(rvec, tvec, rotationVec, cv::Mat::zeros({1,3}, CV_64F), rvec, tvec);

      // Extract useful data from vectiors
      double deltaAngle = atan2(tvec.at<double>(0,0), tvec.at<double>(0,2)) * 180/CV_PI;
      double height     = tvec.at<double>(0,1)/1000;
      double distance   = tvec.at<double>(0,2)/1000;

      if (useVisual && printData) {
        // print data to output stream
        printf("deltaAngle: %0.8f, height: %0.8f, distance: %0.8f\n", deltaAngle, height, distance);
      }
    }

    if (useVisual) {
      // Draw rect
      cv::imshow("Target Detection", display);

      // Prosess keypresses
      char key = cv::waitKey(30);

      // Toggle view settings
      showThresh = (key == 't') ? !showThresh : showThresh;
      show3D     = (key == 'd') ? !show3D     : show3D;
      printData  = (key == 'p') ? !printData  : printData;

      // Quit
      if (key == 27 || key == 'q') {
        break;
      }
    }

    distanceEntry.SetDouble(-1);
    heightEntry.SetDouble(-1);
    deltaAngleEntry.SetDouble(-1);
    isHubVisible.SetBoolean(false);
  }

  // Cleanup
  cv::waitKey(1);
  camera.releaseCapture();
  cv::destroyAllWindows();
  cv::waitKey(1);
  return 0;
}