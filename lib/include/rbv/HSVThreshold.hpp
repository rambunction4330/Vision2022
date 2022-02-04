#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rbv {
struct HSVThreshold {
public:
  HSVThreshold();

  HSVThreshold(cv::Scalar_<int> low, cv::Scalar_<int> high, int blurSize = 0,
               int openSize = 0, cv::MorphShapes openShape = cv::MORPH_RECT,
               int closeSize = 0, cv::MorphShapes closeShape = cv::MORPH_RECT);

  inline const cv::Scalar_<int> &getLow() const { return low; }
  inline const cv::Scalar_<int> &getHigh() const { return high; }
  inline const int &getBlurSize() const { return blurSize; }
  inline const int &getOpenSize() const { return openSize; }
  inline const int &getCloseSize() const { return openSize; }
  inline const cv::MorphShapes &getOpenShape() const { return openShape; }
  inline const cv::MorphShapes &getCloseShape() const { return closeShape; }
  inline const cv::Mat &getOpenMatrix() const { return openMatrix; }
  inline const cv::Mat &getCloseMatrix() const { return closeMatrix; }

  inline const int &getHighH() { return high[0]; }
  inline const int &getLowH() { return low[0]; }
  inline const int &getHighS() { return high[1]; }
  inline const int &getLowS() { return low[1]; }
  inline const int &getHighV() { return high[2]; }
  inline const int &getLowV() { return low[2]; }

  void apply(const cv::Mat &input, cv::Mat &output) const;

  void write(cv::FileStorage &fs) const;
  void read(const cv::FileNode &node);

  const static int MIN_H = 0;
  const static int MAX_H = 180;
  const static int MIN_S = 0;
  const static int MAX_S = 255;
  const static int MIN_V = 0;
  const static int MAX_V = 255;

private:
  cv::Scalar_<int> high;
  cv::Scalar_<int> low;

  int blurSize;

  int openSize, closeSize;
  cv::MorphShapes openShape, closeShape;

  cv::Mat openMatrix;
  cv::Mat closeMatrix;
};

static void write(cv::FileStorage &fs, const std::string &,
                  const HSVThreshold &x) {
  x.write(fs);
}

static void read(const cv::FileNode &node, HSVThreshold &x,
                 const HSVThreshold &default_value = HSVThreshold()) {
  if (node.empty()) {
    x = default_value;
  } else {
    x.read(node);
  }
}
} // namespace rbv