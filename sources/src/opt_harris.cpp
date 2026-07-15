#include "opt_harris.h"
#include "rgb_to_gray.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

/* * * Optimization 1: Loop Fusion.
 * Computes both the Sobel gradients and the squared structure tensor components
 * in a single pass using raw memory pointers. This avoids creating and writing
 * to intermediate Ix and Iy matrices.
 */
void compute_gradients_and_tensors(const cv::Mat &gray, cv::Mat &Ixx,
                                   cv::Mat &Iyy, cv::Mat &Ixy) {
  int rows = gray.rows;
  int cols = gray.cols;

  for (int y = 1; y < rows - 1; ++y) {
    const uchar *r_prev = gray.ptr<uchar>(y - 1);
    const uchar *r_curr = gray.ptr<uchar>(y);
    const uchar *r_next = gray.ptr<uchar>(y + 1);

    float *ptr_xx = Ixx.ptr<float>(y);
    float *ptr_yy = Iyy.ptr<float>(y);
    float *ptr_xy = Ixy.ptr<float>(y);

    for (int x = 1; x < cols - 1; ++x) {
      float gx = -r_prev[x - 1] + r_prev[x + 1] - 2 * r_curr[x - 1] +
                 2 * r_curr[x + 1] - r_next[x - 1] + r_next[x + 1];

      float gy = -r_prev[x - 1] - 2 * r_prev[x] - r_prev[x + 1] +
                 r_next[x - 1] + 2 * r_next[x] + r_next[x + 1];

      ptr_xx[x] = gx * gx;
      ptr_yy[x] = gy * gy;
      ptr_xy[x] = gx * gy;
    }
  }
}

/* * * Optimization 2a: Separable Filter (Horizontal Pass).
 * Instead of a 2D 5x5 sum (25 operations), we do a 1D horizontal sum (5
 * operations) followed later by a 1D vertical sum.
 */
void horizontal_sum(const cv::Mat &Ixx, const cv::Mat &Iyy, const cv::Mat &Ixy,
                    cv::Mat &Hxx, cv::Mat &Hyy, cv::Mat &Hxy) {
  int rows = Ixx.rows;
  int cols = Ixx.cols;

  for (int y = 1; y < rows - 1; ++y) {
    const float *r_xx = Ixx.ptr<float>(y);
    const float *r_yy = Iyy.ptr<float>(y);
    const float *r_xy = Ixy.ptr<float>(y);

    float *h_xx = Hxx.ptr<float>(y);
    float *h_yy = Hyy.ptr<float>(y);
    float *h_xy = Hxy.ptr<float>(y);

    for (int x = 2; x < cols - 2; ++x) {
      h_xx[x] = r_xx[x - 2] + r_xx[x - 1] + r_xx[x] + r_xx[x + 1] + r_xx[x + 2];
      h_yy[x] = r_yy[x - 2] + r_yy[x - 1] + r_yy[x] + r_yy[x + 1] + r_yy[x + 2];
      h_xy[x] = r_xy[x - 2] + r_xy[x - 1] + r_xy[x] + r_xy[x + 1] + r_xy[x + 2];
    }
  }
}

/* * * Optimization 2b: Separable Filter (Vertical Pass) + Loop Fusion for
 * Response. Completes the 5x5 sum by adding vertically, then immediately
 * computes the Harris response (R) without storing the intermediate
 * fully-summed matrices.
 */
void vertical_sum_and_response(const cv::Mat &Hxx, const cv::Mat &Hyy,
                               const cv::Mat &Hxy, cv::Mat &R, float k_param,
                               float &max_R) {
  int rows = Hxx.rows;
  int cols = Hxx.cols;
  max_R = 0.0f;

  for (int y = 2; y < rows - 2; ++y) {
    const float *hxx_m2 = Hxx.ptr<float>(y - 2);
    const float *hxx_m1 = Hxx.ptr<float>(y - 1);
    const float *hxx_0 = Hxx.ptr<float>(y);
    const float *hxx_p1 = Hxx.ptr<float>(y + 1);
    const float *hxx_p2 = Hxx.ptr<float>(y + 2);

    const float *hyy_m2 = Hyy.ptr<float>(y - 2);
    const float *hyy_m1 = Hyy.ptr<float>(y - 1);
    const float *hyy_0 = Hyy.ptr<float>(y);
    const float *hyy_p1 = Hyy.ptr<float>(y + 1);
    const float *hyy_p2 = Hyy.ptr<float>(y + 2);

    const float *hxy_m2 = Hxy.ptr<float>(y - 2);
    const float *hxy_m1 = Hxy.ptr<float>(y - 1);
    const float *hxy_0 = Hxy.ptr<float>(y);
    const float *hxy_p1 = Hxy.ptr<float>(y + 1);
    const float *hxy_p2 = Hxy.ptr<float>(y + 2);

    float *r_ptr = R.ptr<float>(y);

    for (int x = 2; x < cols - 2; ++x) {
      float sxx = hxx_m2[x] + hxx_m1[x] + hxx_0[x] + hxx_p1[x] + hxx_p2[x];
      float syy = hyy_m2[x] + hyy_m1[x] + hyy_0[x] + hyy_p1[x] + hyy_p2[x];
      float sxy = hxy_m2[x] + hxy_m1[x] + hxy_0[x] + hxy_p1[x] + hxy_p2[x];

      float det = (sxx * syy) - (sxy * sxy);
      float trace = sxx + syy;
      float r = det - k_param * (trace * trace);

      r_ptr[x] = r;
      if (r > max_R) {
        max_R = r;
      }
    }
  }
}

/* * * Optimization 3: Fast Non-Maximum Suppression.
 * Hardcodes the 3x3 neighborhood check using adjacent row pointers.
 * This avoids the overhead of tiny nested loops at every pixel.
 */
void apply_fast_nms_and_draw(const cv::Mat &R, cv::Mat &out, float threshold) {
  int rows = R.rows;
  int cols = R.cols;

  for (int y = 3; y < rows - 3; ++y) {
    const float *r_m1 = R.ptr<float>(y - 1);
    const float *r_0 = R.ptr<float>(y);
    const float *r_p1 = R.ptr<float>(y + 1);

    for (int x = 3; x < cols - 3; ++x) {
      float val = r_0[x];

      if (val > threshold) {
        if (val > r_m1[x - 1] && val > r_m1[x] && val > r_m1[x + 1] &&
            val > r_0[x - 1] && val > r_0[x + 1] && val > r_p1[x - 1] &&
            val > r_p1[x] && val > r_p1[x + 1]) {

          cv::circle(out, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 1);
        }
      }
    }
  }
}

} // end anonymous namespace

/* * * Main optimized command function.
 * Orchestrates the accelerated pipeline using the helper functions above.
 */
cv::Mat opt_harris(const cv::Mat &inp, double threshold_ratio) {
  if (inp.empty())
    return cv::Mat();

  cv::Mat gray = to_gray(inp);
  int rows = gray.rows;
  int cols = gray.cols;

  cv::Mat Ixx = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Iyy = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Ixy = cv::Mat::zeros(rows, cols, CV_32F);
  compute_gradients_and_tensors(gray, Ixx, Iyy, Ixy);

  cv::Mat Hxx = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Hyy = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Hxy = cv::Mat::zeros(rows, cols, CV_32F);
  horizontal_sum(Ixx, Iyy, Ixy, Hxx, Hyy, Hxy);

  cv::Mat R = cv::Mat::zeros(rows, cols, CV_32F);
  float max_R = 0.0f;
  vertical_sum_and_response(Hxx, Hyy, Hxy, R, 0.04f, max_R);

  cv::Mat out;
  if (inp.channels() == 1) {
    out = cv::Mat(rows, cols, CV_8UC3);
    for (int i = 0; i < rows * cols; ++i) {
      out.data[i * 3] = inp.data[i];
      out.data[i * 3 + 1] = inp.data[i];
      out.data[i * 3 + 2] = inp.data[i];
    }
  } else {
    out = inp.clone();
  }

  float threshold = max_R * threshold_ratio;
  apply_fast_nms_and_draw(R, out, threshold);

  return out;
}

double parse_opt_harris_config_args(const std::vector<std::string> &args) {
  if (args.empty())
    return 0.01;
  return std::stod(args[0]);
}
