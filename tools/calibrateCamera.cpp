#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <rbv/Camera.hpp>

std::vector<cv::Point3f> generateChessboardPoints(cv::Size boardSize,
                                                  double squareSize);

int main(int argc, char *argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys for argument parsing
  const std::string keys =
      "{ h ? help usage |                 | prints this message            }"
      "{ c cameraID     |        0        | camera id for calibration      }"
      "{ squareSize     |        25       | size of each chessboard square }"
      "{ chessboardSize |      (9,6)      | dimensions of the chessboard   }"
      "{ out output     | calibration.xml | output file                    }";

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
  int cameraID = parser.get<int>("cameraID");
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

  cv::FileStorage storage;
  ;
  if (!storage.open(outputFile, cv::FileStorage::WRITE)) {
    std::cerr << "Error opening output file: '" << outputFile << "'\n";
    return 0;
  }
  storage.release();

  cv::VideoCapture capture;

  // Check camera data
  if (!capture.open(cameraID)) {
    std::cerr << "Could access camera with id: '" << cameraID << "'\n";
    return 0;
  }

  cv::Size imageSize(capture.get(cv::CAP_PROP_FRAME_WIDTH),
                     capture.get(cv::CAP_PROP_FRAME_HEIGHT));

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

  std::vector<std::vector<cv::Point3f>> objectPoints(
      imagePoints.size(), generateChessboardPoints(chessboardSize, squareSize));

  rbv::Camera camera(cameraID, imageSize, objectPoints, imagePoints);

  if (storage.open(outputFile, cv::FileStorage::WRITE)) {
    storage << "Camera" << camera;
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