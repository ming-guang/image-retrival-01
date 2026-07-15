#include "harris_corners.h"
#include "rgb_to_gray.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

/*
 * Purpose: Computes spatial gradients (Ix, Iy) using a 3x3 Sobel operator via
 * pointer arithmetic. Input Parameters:
 * - gray: Constant reference to the input grayscale image (8-bit).
 * - Ix: Reference to the output horizontal gradient matrix (32-bit float).
 * - Iy: Reference to the output vertical gradient matrix (32-bit float).
 * Return Parameter: None (void).
 */
void compute_gradients(const cv::Mat &gray, cv::Mat &Ix, cv::Mat &Iy) {
  int rows = gray.rows;
  int cols = gray.cols;
  const uchar *g_ptr = gray.data;
  float *ix_ptr = (float *)Ix.data;
  float *iy_ptr = (float *)Iy.data;

  int g_step = gray.step;
  int f_step = Ix.step / sizeof(float);

  for (int y = 1; y < rows - 1; ++y) {
    for (int x = 1; x < cols - 1; ++x) {
      float gx = -(*(g_ptr + (y - 1) * g_step + (x - 1))) +
                 (*(g_ptr + (y - 1) * g_step + (x + 1))) -
                 2 * (*(g_ptr + y * g_step + (x - 1))) +
                 2 * (*(g_ptr + y * g_step + (x + 1))) -
                 (*(g_ptr + (y + 1) * g_step + (x - 1))) +
                 (*(g_ptr + (y + 1) * g_step + (x + 1)));

      float gy = -(*(g_ptr + (y - 1) * g_step + (x - 1))) -
                 2 * (*(g_ptr + (y - 1) * g_step + x)) -
                 (*(g_ptr + (y - 1) * g_step + (x + 1))) +
                 (*(g_ptr + (y + 1) * g_step + (x - 1))) +
                 2 * (*(g_ptr + (y + 1) * g_step + x)) +
                 (*(g_ptr + (y + 1) * g_step + (x + 1)));

      *(ix_ptr + y * f_step + x) = gx;
      *(iy_ptr + y * f_step + x) = gy;
    }
  }
}

/*
 * Purpose: Calculates the squared components of the structure tensor for each
 * pixel. Input Parameters:
 * - Ix: Constant reference to the horizontal gradient matrix.
 * - Iy: Constant reference to the vertical gradient matrix.
 * - Ixx: Reference to the output matrix for squared Ix.
 * - Iyy: Reference to the output matrix for squared Iy.
 * - Ixy: Reference to the output matrix for Ix * Iy.
 * Return Parameter: None (void).
 */
void compute_tensor_components(const cv::Mat &Ix, const cv::Mat &Iy,
                               cv::Mat &Ixx, cv::Mat &Iyy, cv::Mat &Ixy) {
  int rows = Ix.rows;
  int cols = Ix.cols;
  int step = Ix.step / sizeof(float);

  float *ix_ptr = (float *)Ix.data;
  float *iy_ptr = (float *)Iy.data;
  float *ixx_ptr = (float *)Ixx.data;
  float *iyy_ptr = (float *)Iyy.data;
  float *ixy_ptr = (float *)Ixy.data;

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      float ix = *(ix_ptr + y * step + x);
      float iy = *(iy_ptr + y * step + x);

      *(ixx_ptr + y * step + x) = ix * ix;
      *(iyy_ptr + y * step + x) = iy * iy;
      *(ixy_ptr + y * step + x) = ix * iy;
    }
  }
}

/*
 * Purpose: Sums the tensor components over a local k x k window.
 * Input Parameters:
 * - Ixx: Constant reference to the squared Ix matrix.
 * - Iyy: Constant reference to the squared Iy matrix.
 * - Ixy: Constant reference to the Ix * Iy matrix.
 * - Sxx: Reference to output matrix for summed Ixx.
 * - Syy: Reference to output matrix for summed Iyy.
 * - Sxy: Reference to output matrix for summed Ixy.
 * - k: Size of the local window (integer).
 * Return Parameter: None (void).
 */
void apply_window_sum(const cv::Mat &Ixx, const cv::Mat &Iyy,
                      const cv::Mat &Ixy, cv::Mat &Sxx, cv::Mat &Syy,
                      cv::Mat &Sxy, int k) {
  int rows = Ixx.rows;
  int cols = Ixx.cols;
  int half = k / 2;
  int step = Ixx.step / sizeof(float);

  float *ixx_ptr = (float *)Ixx.data;
  float *iyy_ptr = (float *)Iyy.data;
  float *ixy_ptr = (float *)Ixy.data;
  float *sxx_ptr = (float *)Sxx.data;
  float *syy_ptr = (float *)Syy.data;
  float *sxy_ptr = (float *)Sxy.data;

  for (int y = half; y < rows - half; ++y) {
    for (int x = half; x < cols - half; ++x) {
      float s_xx = 0, s_yy = 0, s_xy = 0;

      for (int wy = -half; wy <= half; ++wy) {
        for (int wx = -half; wx <= half; ++wx) {
          int offset = (y + wy) * step + (x + wx);
          s_xx += *(ixx_ptr + offset);
          s_yy += *(iyy_ptr + offset);
          s_xy += *(ixy_ptr + offset);
        }
      }
      int target_offset = y * step + x;
      *(sxx_ptr + target_offset) = s_xx;
      *(syy_ptr + target_offset) = s_yy;
      *(sxy_ptr + target_offset) = s_xy;
    }
  }
}

/*
 * Purpose: Calculates the Harris corner response score based on local structure
 * tensors. Input Parameters:
 * - Sxx: Constant reference to summed Ixx matrix.
 * - Syy: Constant reference to summed Iyy matrix.
 * - Sxy: Constant reference to summed Ixy matrix.
 * - R: Reference to the output response matrix.
 * - k_param: Empirical constant for the Harris equation (float).
 * - max_R: Reference to a float to store the maximum response found.
 * Return Parameter: None (void).
 */
void compute_harris_response(const cv::Mat &Sxx, const cv::Mat &Syy,
                             const cv::Mat &Sxy, cv::Mat &R, float k_param,
                             float &max_R) {
  int rows = Sxx.rows;
  int cols = Sxx.cols;
  int half = 5 / 2;
  int step = Sxx.step / sizeof(float);

  float *sxx_ptr = (float *)Sxx.data;
  float *syy_ptr = (float *)Syy.data;
  float *sxy_ptr = (float *)Sxy.data;
  float *r_ptr = (float *)R.data;

  max_R = 0.0f;

  for (int y = half; y < rows - half; ++y) {
    for (int x = half; x < cols - half; ++x) {
      int offset = y * step + x;
      float sxx = *(sxx_ptr + offset);
      float syy = *(syy_ptr + offset);
      float sxy = *(sxy_ptr + offset);

      float det = (sxx * syy) - (sxy * sxy);
      float trace = sxx + syy;
      float r = det - k_param * (trace * trace);

      *(r_ptr + offset) = r;
      if (r > max_R) {
        max_R = r;
      }
    }
  }
}

/*
 * Purpose: Applies Non-Maximum Suppression to the response map and draws local
 * maxima. Input Parameters:
 * - R: Constant reference to the Harris response matrix.
 * - out: Reference to the RGB output image matrix.
 * - threshold: Absolute score threshold for a valid corner (float).
 * - half_window: Radius for NMS checking (integer).
 * Return Parameter: None (void).
 */
void apply_nms_and_draw(const cv::Mat &R, cv::Mat &out, float threshold,
                        int half_window) {
  int rows = R.rows;
  int cols = R.cols;
  int step = R.step / sizeof(float);
  float *r_ptr = (float *)R.data;

  for (int y = half_window + 1; y < rows - half_window - 1; ++y) {
    for (int x = half_window + 1; x < cols - half_window - 1; ++x) {
      float val = *(r_ptr + y * step + x);

      if (val > threshold) {
        bool is_local_max = true;
        for (int ny = -1; ny <= 1; ++ny) {
          for (int nx = -1; nx <= 1; ++nx) {
            if (*(r_ptr + (y + ny) * step + (x + nx)) > val) {
              is_local_max = false;
              break;
            }
          }
          if (!is_local_max)
            break;
        }

        if (is_local_max) {
          cv::circle(out, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 1);
        }
      }
    }
  }
}

} // end anonymous namespace

/*
 * Purpose: Main orchestration function for standard Harris Corner Detection.
 * Input Parameters:
 * - inp: Constant reference to the input image matrix.
 * - threshold_ratio: Relative threshold for corner strength (double).
 * Return Parameter: cv::Mat containing the original image overlaid with
 * detected keypoints.
 */
cv::Mat harris_corners(const cv::Mat &inp, double threshold_ratio) {
  if (inp.empty())
    return cv::Mat();

  cv::Mat gray = to_gray(inp);
  int rows = gray.rows;
  int cols = gray.cols;

  cv::Mat Ix = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Iy = cv::Mat::zeros(rows, cols, CV_32F);
  compute_gradients(gray, Ix, Iy);

  cv::Mat Ixx = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Iyy = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Ixy = cv::Mat::zeros(rows, cols, CV_32F);
  compute_tensor_components(Ix, Iy, Ixx, Iyy, Ixy);

  cv::Mat Sxx = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Syy = cv::Mat::zeros(rows, cols, CV_32F);
  cv::Mat Sxy = cv::Mat::zeros(rows, cols, CV_32F);
  apply_window_sum(Ixx, Iyy, Ixy, Sxx, Syy, Sxy, 5);

  cv::Mat R = cv::Mat::zeros(rows, cols, CV_32F);
  float max_R = 0.0f;
  compute_harris_response(Sxx, Syy, Sxy, R, 0.04f, max_R);

  cv::Mat out;
  if (inp.channels() == 1) {
    out = cv::Mat(rows, cols, CV_8UC3);
    uchar *out_ptr = out.data;
    uchar *inp_ptr = inp.data;
    for (int i = 0; i < rows * cols; ++i) {
      *(out_ptr + i * 3) = *(inp_ptr + i);
      *(out_ptr + i * 3 + 1) = *(inp_ptr + i);
      *(out_ptr + i * 3 + 2) = *(inp_ptr + i);
    }
  } else {
    out = inp.clone();
  }

  float threshold = max_R * threshold_ratio;
  apply_nms_and_draw(R, out, threshold, 2);

  return out;
}

double parse_harris_config_args(const std::vector<std::string> &args) {
  if (args.empty())
    return 0.01;
  return std::stod(args[0]);
}
