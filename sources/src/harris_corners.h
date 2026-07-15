#ifndef HARRIS_CORNERS_H
#define HARRIS_CORNERS_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

cv::Mat harris_corners(const cv::Mat &inp, double threshold_ratio);
double parse_harris_config_args(const std::vector<std::string> &args);

#endif
