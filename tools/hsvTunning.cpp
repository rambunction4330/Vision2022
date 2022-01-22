#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <rvb/Threshold.hpp>

int main(int argc, char* argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys definign command line argumanet behavior
  const std::string parseKeys = 
    "{ h ? help usage |   | prints this message                   }"
    "{ id cameraID    | 0 | Camera id used for thresholding       }"
    "{ b blur         |   | Whether to present a blur slider      }"
    "{ m morph        |   | Whether to present morphology sliders }"
    "{ i in input     |   | Input file                            }"
    "{ o out output   |   | Output file                           }";

  // Object to parse any argument given
  cv::CommandLineParser parser(argc, argv, parseKeys);
  parser.about("\nVision2021 v22.0.0 hsvTunning"
               "\nTool to find hsv thresholding value\n");

  // Show help if help is flagged.
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  // Get arguments from the parser
  int cameraID = parser.get<double>("cameraID");
  bool useBlurSlider = parser.has("blur");
  bool useMorphSlider = parser.has("morph");
  std::string inputFile = parser.get<std::string>("input");
  std::string outputFile = parser.get<std::string>("output");

  // Cheack for errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  /*************
   * File Input
   *************/

  // Variable to hold any thresholding data
  rbv::Threshold threshold;

  // If a file was given, extract the data from that file
  if (inputFile != "") {
    if (std::filesystem::exists(inputFile)) {
      cv::FileStorage storage(inputFile, cv::FileStorage::READ);
      if (storage.isOpened()) {
        storage["Threshold"] >> threshold;
      } else {
        std::cerr << "Error opening input file: '" << inputFile << "'\n";
        return 0;
      }
      storage.release();
    } else {
      std::cerr << "Could not find input file: '" << inputFile << "'\n";
      return 0;
    }
  }

  /************
   * GUI Setup
   ************/

  // Morphology variables
  int openSize = 15, openShape = 0, closeSize = 15, closeShape = 0;

  // Window
  cv::namedWindow("HSV Tunning");

  // Trackbars
  cv::createTrackbar("High H",  "HSV Tunning", &threshold.highH(),  180);
  cv::createTrackbar("Low H", "HSV Tunning", &threshold.lowH(), 180);
  cv::createTrackbar("High S",  "HSV Tunning", &threshold.highS(),  255);
  cv::createTrackbar("Low S", "HSV Tunning", &threshold.lowS(), 255);
  cv::createTrackbar("High V",  "HSV Tunning", &threshold.highV(),  255);
  cv::createTrackbar("Low V", "HSV Tunning", &threshold.lowV(), 255);

  // Conditionaly add extra sliders depending on argument flags
  if (useBlurSlider) {
    cv::createTrackbar("Blur Size", "HSV Tunning", &threshold.blurSize, 100);
  }

  if (useMorphSlider) {
    cv::createTrackbar("Open Size", "HSV Tunning", &openSize, 100);
    cv::createTrackbar("Open Type", "HSV Tunning", &openShape, 2);
    cv::createTrackbar("Close Size", "HSV Tunning", &closeSize, 100);
    cv::createTrackbar("Close Type", "HSV Tunning", &closeShape, 2);
  }

  /************
   * Main Loop
   ************/
 
  // Open camera with the given id
  cv::VideoCapture capture(cameraID);

  // Check camera data
  if (!capture.isOpened()) {
    std::cerr << "Could access camera with id: '" << cameraID << "'\n";
    return 0;
  }

  cv::Mat frame, thresh;
  cv::Mat& display = frame;
  bool showThresh = true;
  while (true) {
    // Get the next frame from the camera.
    capture >> frame;

    // Check camera data.
    if (frame.empty()) {
      std::cerr << "Lost connection to camera\n";
      break;
    }

    // Pull in data from morph sliders.
    // This doesn't have to be done for the others since they are directly modify the value. 
    if (useMorphSlider) {
      threshold.closeMatrix = cv::getStructuringElement(closeShape, {std::max(closeSize, 1), std::max(closeSize, 1)});
      threshold.openMatrix = cv::getStructuringElement(openShape, {std::max(openSize, 1), std::max(openSize, 1)});
    }

    threshold.apply(frame, thresh);

    if (showThresh) {
      display = thresh;
    } else {
      display = frame;
    }

    // Show Image
    cv::imshow("HSV Tunning", display);

    // Get keystrokes
    int key = cv::waitKey(30);

    // Toggle dispay settings
    showThresh = (key == 't') ? !showThresh : showThresh;

    // Save if 's' is pressed, and a file was given to output to.
    if (key == 's' && outputFile != "") {
      cv::FileStorage storage(outputFile, cv::FileStorage::WRITE);
      if (storage.isOpened()) {
        storage << "Threshold" << threshold;
      } else {
        std::cerr << "Error opening output file: '" << outputFile << "'\n";
        break;
      }
      storage.release();
    }

    // Exit the tool
    if (key == 'q' || key == 27) {
      break;
    }
  } 

   // Cleanup when done.
  cv::destroyAllWindows();
  capture.release();
  cv::waitKey(1);
  return 0;
}