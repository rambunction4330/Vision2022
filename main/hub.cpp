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
      "{ h ? help usage |                 | prints this message }"
      "{ c calibration  | calibration.xml | calibration file    }"
      "{ t threshold    |  threshold.xml  | threshold file      }"
      "{ v visual       |                 | flag to create gui  }"
      "{ T team         |       4330      | team number         }";

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
  bool showThresh = false;
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
    if (useVisual) {
      if (showThresh) {
        cv::cvtColor(thresh, display, cv::COLOR_GRAY2BGR);
      } else {
        frame.copyTo(display);
      }
    }

    // Find contours
    std::vector<std::vector<cv::Point2f>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Make sure there are enough contours (but not too many)
    if (contours.size() > 0 && contours.size() < 5) {
      for (int i = 0; i < contours.size(); i++) {
        // Get boundign rect to determine "rectness"
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        double rectArea = rect.size.width * rect.size.height;
        double contourArea = cv::contourArea(contours[i]);

        // Check rect size and "reectness"
        if (contourArea > 25 && contourArea/rectArea > 0.7) {

          // Approximate rect by given emthod
          std::vector<cv::Point2f> approxRect;
          bool foundRect;
          foundRect = rbv::approxNGonPolyDP(contours[i], approxRect);

          // Draw if found
          if (foundRect) {
            for (int j = 0; j < 4; j++) {
              cv::line(display, approxRect[j], approxRect[(j+1)%4], {255, 0, 0}, 2);
            }
          }
        }
      }
    }

    if (useVisual) {
      // Draw rect
      cv::imshow("Target Detection", display);

      // Prosess keypresses
      char key = cv::waitKey(30);

      // Toggle view settings
      showThresh = (key == 't') ? !showThresh : showThresh;
  
      // Quit
      if (key == 27 || key == 'q') {
        break;
      }
    }

    distanceEntry.SetDouble(-1);
    heighEntry.SetDouble(-1);
    deltaAngleEntry.SetDouble(-1);
  }

  // Cleanup
  cv::waitKey(1);
  camera.releaseCapture();
  cv::destroyAllWindows();
  cv::waitKey(1);
  return 0;
}