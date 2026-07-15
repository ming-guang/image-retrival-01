#ifndef H_FEATURES
#define H_FEATURES

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Enhances an image before extraction: resize to 256x256 then CLAHE on the
// L channel (LAB space) to normalise contrast. Returns a BGR image.
cv::Mat preprocess_image(const cv::Mat &bgr);

// HSV color histogram: `bins` bins per channel, each L1-normalised, concatenated
// into a 1x(3*bins) CV_32F row.
cv::Mat compute_color_hist(const cv::Mat &bgr, int bins = 32);

// Color auto-correlogram: probability that a pixel of a quantised color has a
// same-color neighbour at each given gap. Quantises to n_colors^3 colors and
// samples 4-connected neighbours; returns 1x(n_colors^3 * gaps) CV_32F.
cv::Mat compute_correlogram(const cv::Mat &bgr, int n_colors = 4,
                            const std::vector<int> &gaps = {1, 3, 5, 7});

// Mean of all SIFT descriptors detected in the image -> 1x128 CV_32F.
cv::Mat compute_sift_mean(const cv::Mat &bgr, int max_kp = 500);

// Mean of all ORB descriptors (converted to float) -> 1x32 CV_32F.
cv::Mat compute_orb_mean(const cv::Mat &bgr, int max_kp = 500);

// Seven log-scaled Hu moments capturing global shape -> 1x7 CV_32F.
cv::Mat compute_hu_moments(const cv::Mat &bgr);

// Local Binary Pattern texture histogram: 8-neighbour code per pixel binned into
// 256 L1-normalised bins -> 1x256 CV_32F.
cv::Mat compute_lbp_hist(const cv::Mat &bgr);

// Histogram of Oriented Gradients on a 64x128 resize -> 1x3780 CV_32F.
cv::Mat compute_hog_desc(const cv::Mat &bgr);

// Chi-squared distance between two 1xN histograms.
float dist_chi2(const cv::Mat &h1, const cv::Mat &h2);

// Euclidean (L2) distance between two 1xN vectors.
float dist_l2(const cv::Mat &v1, const cv::Mat &v2);

#endif
