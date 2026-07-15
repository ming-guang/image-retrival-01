#include "image_features.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <cmath>

cv::Mat preprocess_image(const cv::Mat &bgr) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(256, 256));
  cv::Mat lab;
  cv::cvtColor(resized, lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> ch;
  cv::split(lab, ch);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(ch[0], ch[0]);
  cv::merge(ch, lab);
  cv::Mat out;
  cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
  return out;
}

cv::Mat compute_color_hist(const cv::Mat &bgr, int bins) {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  // H spans 0-179 in OpenCV; S and V span 0-255. Use per-channel ranges.
  float h_range[] = {0, 180};
  float sv_range[] = {0, 256};
  cv::Mat out;
  for (int c = 0; c < 3; c++) {
    const float *ranges = (c == 0) ? h_range : sv_range;
    cv::Mat hist;
    cv::calcHist(&ch[c], 1, nullptr, cv::Mat(), hist, 1, &bins, &ranges);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    out.push_back(hist.reshape(1, 1));
  }
  return out.reshape(1, 1).clone();
}

// Maps a BGR pixel to a single quantised color index in [0, n_colors^3).
static inline int quantise_color(const cv::Vec3b &px, int n_colors) {
  int q = 256 / n_colors;
  int b = px[0] / q, g = px[1] / q, r = px[2] / q;
  return b + n_colors * g + n_colors * n_colors * r;
}

cv::Mat compute_correlogram(const cv::Mat &bgr, int n_colors,
                            const std::vector<int> &gaps) {
  cv::Mat small;
  cv::resize(bgr, small, cv::Size(128, 128));
  int n_bins = n_colors * n_colors * n_colors;

  // Precompute the quantised color index for every pixel once.
  cv::Mat idx(small.size(), CV_32S);
  for (int y = 0; y < small.rows; y++)
    for (int x = 0; x < small.cols; x++)
      idx.at<int>(y, x) = quantise_color(small.at<cv::Vec3b>(y, x), n_colors);

  cv::Mat out(1, n_bins * (int)gaps.size(), CV_32F, cv::Scalar(0));
  const int dy[] = {0, 0, -1, 1};
  const int dx[] = {-1, 1, 0, 0};
  for (size_t gi = 0; gi < gaps.size(); gi++) {
    int d = gaps[gi];
    std::vector<double> same(n_bins, 0.0), total(n_bins, 0.0);
    for (int y = 0; y < small.rows; y++) {
      for (int x = 0; x < small.cols; x++) {
        int c = idx.at<int>(y, x);
        for (int k = 0; k < 4; k++) {
          int ny = y + dy[k] * d, nx = x + dx[k] * d;
          if (ny < 0 || nx < 0 || ny >= small.rows || nx >= small.cols)
            continue;
          total[c] += 1.0;
          if (idx.at<int>(ny, nx) == c)
            same[c] += 1.0;
        }
      }
    }
    for (int c = 0; c < n_bins; c++)
      out.at<float>(0, gi * n_bins + c) =
          (total[c] > 0) ? (float)(same[c] / total[c]) : 0.0f;
  }
  return out;
}

cv::Mat compute_sift_mean(const cv::Mat &bgr, int max_kp) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create(max_kp);
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  sift->detectAndCompute(gray, cv::noArray(), kps, desc);
  if (desc.empty())
    return cv::Mat::zeros(1, 128, CV_32F);
  cv::Mat mean;
  cv::reduce(desc, mean, 0, cv::REDUCE_AVG, CV_32F);
  return mean;
}

cv::Mat compute_orb_mean(const cv::Mat &bgr, int max_kp) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(max_kp);
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  orb->detectAndCompute(gray, cv::noArray(), kps, desc);
  if (desc.empty())
    return cv::Mat::zeros(1, 32, CV_32F);
  cv::Mat fdesc;
  desc.convertTo(fdesc, CV_32F);
  cv::Mat mean;
  cv::reduce(fdesc, mean, 0, cv::REDUCE_AVG, CV_32F);
  return mean;
}

cv::Mat compute_hu_moments(const cv::Mat &bgr) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Moments m = cv::moments(gray, false);
  double hu[7];
  cv::HuMoments(m, hu);
  cv::Mat out(1, 7, CV_32F);
  // Log-scale each moment (sign-preserving) so the huge dynamic range compresses.
  for (int i = 0; i < 7; i++) {
    double v = hu[i];
    out.at<float>(0, i) =
        (v != 0.0)
            ? (float)(std::copysign(1.0, v) * std::log10(std::fabs(v) + 1e-30))
            : 0.0f;
  }
  return out;
}

cv::Mat compute_lbp_hist(const cv::Mat &bgr) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Mat lbp(gray.size(), CV_8U, cv::Scalar(0));
  // Clockwise 8-neighbour offsets starting top-left; each contributes one bit.
  const int dy[] = {-1, -1, -1, 0, 1, 1, 1, 0};
  const int dx[] = {-1, 0, 1, 1, 1, 0, -1, -1};
  for (int y = 1; y < gray.rows - 1; y++) {
    for (int x = 1; x < gray.cols - 1; x++) {
      uchar center = gray.at<uchar>(y, x);
      uchar code = 0;
      for (int b = 0; b < 8; b++)
        if (gray.at<uchar>(y + dy[b], x + dx[b]) >= center)
          code |= (1 << b);
      lbp.at<uchar>(y, x) = code;
    }
  }
  int bins = 256;
  float range[] = {0, 256};
  const float *ranges = range;
  cv::Mat hist;
  cv::calcHist(&lbp, 1, nullptr, cv::Mat(), hist, 1, &bins, &ranges);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
  return hist.reshape(1, 1).clone();
}

cv::Mat compute_hog_desc(const cv::Mat &bgr) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(64, 128));
  cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8),
                        cv::Size(8, 8), 9);
  std::vector<float> feats;
  hog.compute(resized, feats);
  return cv::Mat(1, (int)feats.size(), CV_32F, feats.data()).clone();
}

float dist_chi2(const cv::Mat &h1, const cv::Mat &h2) {
  cv::Mat num, den;
  cv::multiply(h1 - h2, h1 - h2, num);
  cv::add(h1, h2, den);
  den += 1e-10;
  cv::Mat term;
  cv::divide(num, den, term);
  return (float)cv::sum(term)[0];
}

float dist_l2(const cv::Mat &v1, const cv::Mat &v2) {
  return (float)cv::norm(v1, v2, cv::NORM_L2);
}
