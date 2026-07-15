#include "rgb_to_gray.h"
#include <opencv2/core/mat.hpp>

cv::Mat to_gray(const cv::Mat &inp) {
  if (inp.empty()) {
    return cv::Mat();
  }

  if (inp.channels() != 3) {
    return inp.clone();
  }

  cv::Mat out(inp.rows, inp.cols, CV_8UC1);

  for (int y = 0; y < inp.rows; ++y) {
    const uchar *row_in = inp.ptr<uchar>(y);
    uchar *row_out = out.ptr<uchar>(y);

    for (int x = 0; x < inp.cols; ++x) {
      int b_index = x * 3;
      int g_index = x * 3 + 1;
      int r_index = x * 3 + 2;

      uchar b = row_in[b_index];
      uchar g = row_in[g_index];
      uchar r = row_in[r_index];

      float gray_value = (0.299f * r) + (0.587f * g) + (0.114f * b);

      row_out[x] = static_cast<uchar>(gray_value);
    }
  }

  return out;
}
