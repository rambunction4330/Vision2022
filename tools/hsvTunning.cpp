#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <rbv/Threshold.hpp>

void highHCallback(int pos, void* highH) {
  *((int*)highH) = pos;
}
void lowHCallback(int pos, void* lowH) {
  *((int*)lowH) = pos;
}
void highSCallback(int pos, void* highS) {
  *((int*)highS) = pos;
}
void lowSCallback(int pos, void* lowS) {
  *((int*)lowS) = pos;
}
void highVCallback(int pos, void* highV) {
  *((int*)highV) = pos;
}
void lowVCallback(int pos, void* lowV) {
  *((int*)lowV) = pos;
}

void blurCallback(int pos, void* blur) {
  *((int*)blur) = pos;
}
void closeShapeCallback(int pos, void* shape) {
  *((cv::MorphShapes*)shape) = cv::MorphShapes(pos);
}
void closeSizeCallback(int pos, void* size) {
  *((int*)size) = pos;
}
void openShapeCallback(int pos, void* shape) {
  *((cv::MorphShapes*)shape) = cv::MorphShapes(pos);
}
void openSizeCallback(int pos, void* size) {
  *((int*)size) = pos;
}

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

  // Morphology variables
  int highH = 180, lowH = 0, highS = 255, lowS = 0, highV = 255, lowV = 0;
  int blurSize = 0, openSize = 0, closeSize = 0; 
  cv::MorphShapes openShape = cv::MORPH_RECT, closeShape = cv::MORPH_RECT;

  /*************
   * File Input
   *************/

  // Variable to hold any thresholding data
  rbv::HSVThreshold inThreshold;

  // If a file was given, extract the data from that file
  if (inputFile != "") {
    if (std::filesystem::exists(inputFile)) {
      cv::FileStorage storage(inputFile, cv::FileStorage::READ);
      if (storage.isOpened()) {
        storage["HSVThreshold"] >> inThreshold;

        highH = inThreshold.getHighH();
        lowH = inThreshold.getLowH();
        highS = inThreshold.getHighS();
        lowS = inThreshold.getLowS();
        highV = inThreshold.getHighV();
        lowV = inThreshold.getLowV();

        blurSize = inThreshold.getBlurSize();
        openSize = inThreshold.getOpenSize();
        closeSize = inThreshold.getCloseSize();
        openShape = inThreshold.getOpenShape();
        closeShape = inThreshold.getCloseShape();
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

  // Window
  cv::namedWindow("HSV Tunning");

  // Trackbars
  cv::createTrackbar("High H", "HSV Tunning", NULL, 180, highHCallback, &highH);
  cv::createTrackbar("Low H",  "HSV Tunning", NULL, 180, highHCallback, &lowH);
  cv::createTrackbar("High S", "HSV Tunning", NULL, 255, highHCallback, &highS);
  cv::createTrackbar("Low S",  "HSV Tunning", NULL, 255, highHCallback, &lowS);
  cv::createTrackbar("High V", "HSV Tunning", NULL, 255, highHCallback, &highV);
  cv::createTrackbar("Low V",  "HSV Tunning", NULL, 255, highHCallback, &lowV);

  // Conditionaly add extra sliders depending on argument flags
  if (useBlurSlider) {
    cv::createTrackbar("Blur Size", "HSV Tunning", NULL, 100, blurCallback, &blurSize);
  }

  if (useMorphSlider) {
    cv::createTrackbar("Open Size", "HSV Tunning",   NULL, 100, openSizeCallback,   &openSize);
    cv::createTrackbar("Open Shape", "HSV Tunning",  NULL,   2, openShapeCallback,  &openShape);
    cv::createTrackbar("Close Size", "HSV Tunning",  NULL, 100, closeSizeCallback,  &closeSize);
    cv::createTrackbar("Close Shape", "HSV Tunning", NULL,   2, closeShapeCallback, &closeShape);
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

  cv::Mat frame, thresh, display;
  rbv::HSVThreshold outThreshold;
  bool showThresh = true;
  while (true) {
    // Get the next frame from the camera.
    capture >> frame;

    // Check camera data.
    if (frame.empty()) {
      std::cerr << "Lost connection to camera\n";
      break;
    }

    outThreshold = rbv::HSVThreshold({lowH, lowS, lowV}, {highH, highS, highV}, 
                                      blurSize, openSize, openShape, closeSize, closeShape);

    outThreshold.apply(frame, thresh);

    if (showThresh) {
      thresh.copyTo(display);
    } else {
      frame.copyTo(display);
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
        storage << "HSVThreshold" << outThreshold;
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