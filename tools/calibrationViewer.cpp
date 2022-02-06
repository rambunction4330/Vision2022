#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <rbv/Camera.hpp>
#include <rbv/StereoPair.hpp>

void viewStereo(rbv::StereoPair pair) {
  if (!pair.openCaptures()) {
    std::cerr << "Could access pair\n";
    return;
  }

  cv::Ptr<cv::StereoMatcher> matcher = cv::StereoSGBM::create();

  rbv::StereoFrame frame, rectified;
  cv::Mat disparity, disparityNorm, stereoDisplay, disparityDisplay, together;
  bool useRectify = true, normalizeDisparity = true;
  while (true) {
    pair.getNextFrames(frame);

    if (frame.leftImage.empty() || frame.rightImage.empty()) {
      std::cerr << "Could access pair\n";
      break;
    }

    pair.rectifyStereoFrame(frame, rectified);
    pair.calculateDisparity(frame, disparity);

    if (useRectify) {
      cv::hconcat(rectified.leftImage, rectified.rightImage, stereoDisplay);
    } else {
      cv::hconcat(frame.leftImage, frame.rightImage, stereoDisplay);
    }

    if (normalizeDisparity) {
      cv::normalize(disparity, disparityDisplay, 0, 255, cv::NORM_MINMAX,
                    CV_8UC1);
    } else {
      disparity.copyTo(disparityDisplay);
    }

    cv::addWeighted(rectified.leftImage, 0.5, rectified.rightImage, 0.5, 0,
                    together);

    cv::imshow("Stereo Frame", stereoDisplay);
    cv::imshow("Disparity", disparityDisplay);

    char key = cv::waitKey(30);

    useRectify = (key == 'r') ? !useRectify : useRectify;
    normalizeDisparity =
        (key == 'n') ? !normalizeDisparity : normalizeDisparity;

    if (key == 'q' || key == 27) {
      break;
    }
  }

  cv::waitKey(1);
  cv::destroyAllWindows();
  pair.releaseCaptures();
  cv::waitKey(1);
}

void viewCamera(rbv::Camera camera) {

  if (!camera.openCapture()) {
    std::cerr << "Could access camera\n";
    return;
  }

  cv::Mat frame, display;
  bool useUndistort = true;
  while (true) {
    camera.getNextFrame(frame);

    if (frame.empty()) {
      std::cerr << "Lost access to camera\n";
      break;
    }

    if (useUndistort) {
      camera.undistortFrame(frame, display);
    } else {
      frame.copyTo(display);
    }

    cv::imshow("Camera", display);

    char key = cv::waitKey(30);

    useUndistort = (key == 'u') ? !useUndistort : useUndistort;

    if (key == 'q' || key == 27) {
      break;
    }
  }

  cv::waitKey(1);
  cv::destroyAllWindows();
  camera.releaseCapture();
  cv::waitKey(1);
}

int main(int argc, char *argv[]) {

  /***********************
   * Command Line Parsing
   ***********************/

  // Keys for argument parsing
  const std::string keys =
      "{ h ? help usage |                 | prints this message            }"
      "{ c calibration  | calibration.xml | calibration file               }"
      "{ s stereo       |                 | flag if stereo calibration     }"
      "{ squareSize     |        25       | size of each chessboard square }"
      "{ chessboardSize |      (9,6)      | dimensions of the chessboard   }";

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
  bool useStereo = parser.has("stereo");
  double squareSize = parser.get<double>("squareSize");

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
  if (useStereo) {
    if (calibrationFile == "" ||
        storage.open(calibrationFile, cv::FileStorage::READ)) {
      rbv::StereoPair pair;
      storage["StereoPair"] >> pair;
      viewStereo(pair);
    } else {
      std::cerr << "Error opening output file: '" << calibrationFile << "'\n";
      return 0;
    }
    storage.release();
  } else {
    if (calibrationFile == "" ||
        storage.open(calibrationFile, cv::FileStorage::READ)) {
      rbv::Camera camera;
      storage["Camera"] >> camera;
      viewCamera(camera);
    } else {
      std::cerr << "Error opening output file: '" << calibrationFile << "'\n";
      return 0;
    }
    storage.release();
  }
}