#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <rbv/Camera.hpp>

std::vector<cv::Point3f> generateChessboardPoints(cv::Size boardSize, double squareSize);

int main(int argc, char* argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys for argument parsing
  const std::string keys = 
  "{ h ? help usage |       | prints this message             }"
  "{ id cameraID    |   0   | Camera id used for thresholding }"
  "{ squareSize     |   25  | Size of each chessboard square  }"
  "{ chessboardSize | (9,6) | Dimensions of the chessboard    }"
  "{ in input       |       | Input file                      }"
  "{ out output     |       | Output file                     }";

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
  int cameraID = parser.get<double>("cameraID");
  double squareSize = parser.get<int>("squareSize");
  std::string inputFile = parser.get<std::string>("input");
  std::string outputFile = parser.get<std::string>("output");

  cv::Size chessboardSize;
  if (sscanf(parser.get<std::string>("chessboardSize").c_str(), "(%d,%d)", &chessboardSize.width, &chessboardSize.height) != 2) {
    std::cerr << "Invalid format for argument 'chessboardSize'\n";
    return 0;
  }

  // Cheack for errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  rbv::Camera camera;
  if (inputFile != "" && std::filesystem::exists(inputFile)) {
    cv::FileStorage fileStorage(inputFile, cv::FileStorage::READ);
    if (fileStorage.isOpened()) {
      fileStorage["Camera"] >> camera;
    }
    fileStorage.release();
  }

  cv::VideoCapture capture(cameraID);

  // Check camera data
  if (!capture.isOpened()) {
    std::cerr << "Could access camera with id: '" << cameraID << "'\n";
    return 0;
  }

  cv::Size imageSize(capture.get(cv::CAP_PROP_FRAME_HEIGHT), capture.get(cv::CAP_PROP_FRAME_WIDTH));

  std::vector<std::vector<cv::Point2f>> imagePoints;
  cv::Mat frame;
  while (true) {
    capture >> frame;

    if (frame.empty()) {
      std::cerr << "Lost connection to camera\n";
      return 0;
    }

    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(frame, chessboardSize, corners);

    if (found) {
      cv::drawChessboardCorners(frame, chessboardSize, corners, found);
    }

    char key = cv::waitKey(30);

    if (key == ' ' && found) {
      frame = cv::Mat(frame.rows, frame.cols, CV_8UC1, 255);
      imagePoints.push_back(corners);
    }

    cv::imshow("Find Chessboard Points", frame);

    if (key == 'q' | key == 27) {
      break;
    }
  }

  cv::waitKey(1);
  cv::destroyAllWindows();
  capture.release();
  cv::waitKey(1);

  cv::Mat rves, tvecs;
  std::vector<std::vector<cv::Point3f>> objectPoints(imagePoints.size(), generateChessboardPoints(chessboardSize, squareSize));
  cv::calibrateCamera(objectPoints, imagePoints, imageSize, camera.cameraMatrix, camera.distortion, rves, tvecs);


  capture.open(cameraID);

  // Check camera data
  if (!capture.isOpened()) {
    std::cerr << "Could access camera with id: '" << cameraID << "'\n";
    return 0;
  }

  bool undistorted = true;
  cv::Mat undistort, display;
  while (true) {
    capture >> frame;

    if (frame.empty()) {
      std::cerr << "Lost connection to camera\n";
      return 0;
    } 

    if (undistorted) {
      cv::undistort(frame, undistort, camera.cameraMatrix, camera.distortion);
      undistort.copyTo(display);
    } else {
      frame.copyTo(display);
    }

    cv::imshow("Calibrated", display);

    char key = cv::waitKey(30);

    // Save if 's' is pressed, and a file was given to output to.
    if (key == 's' && outputFile != "") {
      cv::FileStorage storage(outputFile, cv::FileStorage::WRITE);
      if (storage.isOpened()) {
        storage << "Camera" << camera;
      } else {
        std::cerr << "Error opening output file: '" << outputFile << "'\n";
        break;
      }
      storage.release();
    }

    undistorted = key == 'u' ? !undistorted : undistorted;

    if (key == 'q' || key == 27) {
      break;
    }
  }
  cv::destroyAllWindows();
  capture.release();
  cv::waitKey(1);
  return 0;
}

std::vector<cv::Point3f> generateChessboardPoints(cv::Size boardSize, double squareSize) {
  std::vector<cv::Point3f> points;
  points.reserve(boardSize.area());
  for (int h = 0; h < boardSize.height; h++) {
    for (int w = 0; w < boardSize.width; w++) {
      points.push_back({static_cast<float>(squareSize * h), static_cast<float>(squareSize * w), 0});
    }
  }

  return points;
}