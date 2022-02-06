#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <rbv/Camera.hpp>
#include <rbv/StereoPair.hpp>

std::vector<cv::Point3f> generateChessboardPoints(cv::Size boardSize,
                                                  double squareSize);

int main(int argc, char *argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys for argument parsing
  const std::string keys = "{ h ? help usage |                       | prints "
                           "this message            }"
                           "{ lc leftCamera  |                       | left "
                           "camera file               }"
                           "{ rc rightCamera |                       | right "
                           "camera file              }"
                           "{ squareSize     |           25          | size of "
                           "each chessboard square }"
                           "{ chessboardSize |         (9,6)         | "
                           "dimensions of the chessboard   }"
                           "{ out output     | stereoCalibration.xml | output "
                           "file                    }";

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
  std::string leftCameraFile = parser.get<std::string>("leftCamera");
  std::string rightCameraFile = parser.get<std::string>("rightCamera");
  double squareSize = parser.get<double>("squareSize");
  std::string outputFile = parser.get<std::string>("output");

  cv::Size chessboardSize;
  if (sscanf(parser.get<std::string>("chessboardSize").c_str(), "(%d,%d)",
             &chessboardSize.width, &chessboardSize.height) != 2) {
    std::cerr << "Invalid format for argument 'chessboardSize'\n";
    return 0;
  }

  // Cheack for errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  rbv::Camera leftCamera, rightCamera;

  cv::FileStorage storage;
  if (leftCameraFile == "" ||
      storage.open(leftCameraFile, cv::FileStorage::READ)) {
    storage["Camera"] >> leftCamera;
  } else {
    std::cerr << "Error opening output file: '" << leftCameraFile << "'\n";
    return 0;
  }
  storage.release();

  if (rightCameraFile == "" ||
      storage.open(rightCameraFile, cv::FileStorage::READ)) {
    storage["Camera"] >> rightCamera;
  } else {
    std::cerr << "Error opening output file: '" << rightCameraFile << "'\n";
    return 0;
  }
  storage.release();

  if (!storage.open(outputFile, cv::FileStorage::WRITE)) {
    std::cerr << "Error opening output file: '" << outputFile << "'\n";
    return 0;
  }
  storage.release();

  // Check camera data
  if (!leftCamera.openCapture()) {
    std::cerr << "Could access camera with id: '" << leftCamera.getID()
              << "'\n";
    return 0;
  }

  if (!rightCamera.openCapture()) {
    std::cerr << "Could access camera with id: '" << rightCamera.getID()
              << "'\n";
    return 0;
  }

  std::vector<std::vector<cv::Point2f>> leftImagePoints, rightImagePoints;
  cv::Mat leftFrame, rightFrame, display;
  while (true) {
    leftCamera.getNextFrame(leftFrame);
    rightCamera.getNextFrame(rightFrame);

    if (leftFrame.empty()) {
      std::cerr << "Lost connection to camera with id: " << leftCamera.getID()
                << "\n";
      return 0;
    }

    if (rightFrame.empty()) {
      std::cerr << "Lost connection to camera with id: " << rightCamera.getID()
                << "\n";
      return 0;
    }

    std::vector<cv::Point2f> leftCorners, rightCorners;
    bool leftFound =
        cv::findChessboardCorners(leftFrame, chessboardSize, leftCorners);
    bool rightFound =
        cv::findChessboardCorners(rightFrame, chessboardSize, rightCorners);
    cv::drawChessboardCorners(leftFrame, chessboardSize, leftCorners,
                              leftFound);
    cv::drawChessboardCorners(rightFrame, chessboardSize, rightCorners,
                              rightFound);

    cv::hconcat(leftFrame, rightFrame, display);

    char key = cv::waitKey(30);

    if (key == ' ' && leftFound && rightFound) {
      display = cv::Mat(leftFrame.rows, leftFrame.cols * 2, CV_8UC1, 255);
      leftImagePoints.push_back(leftCorners);
      rightImagePoints.push_back(rightCorners);
    }

    cv::imshow("Find Chessboard Points", display);

    if (key == 'q' | key == 27) {
      break;
    }
  }

  cv::waitKey(1);
  cv::destroyAllWindows();
  leftCamera.releaseCapture();
  rightCamera.releaseCapture();
  cv::waitKey(1);

  std::vector<std::vector<cv::Point3f>> objectPoints(
      leftImagePoints.size(),
      generateChessboardPoints(chessboardSize, squareSize));

  rbv::StereoPair pair(leftCamera, rightCamera, objectPoints, leftImagePoints,
                       rightImagePoints);

  if (storage.open(outputFile, cv::FileStorage::WRITE)) {
    storage << "StereoPair" << pair;
  } else {
    std::cerr << "Error opening output file: '" << outputFile << "'\n";
    return 0;
  }
  storage.release();
}

std::vector<cv::Point3f> generateChessboardPoints(cv::Size boardSize,
                                                  double squareSize) {
  std::vector<cv::Point3f> points;
  points.reserve(boardSize.area());
  for (int h = 0; h < boardSize.height; h++) {
    for (int w = 0; w < boardSize.width; w++) {
      points.push_back({static_cast<float>(squareSize * h),
                        static_cast<float>(squareSize * w), 0});
    }
  }
  return points;
}