#ifndef OPT_HARRIS_H
#define OPT_HARRIS_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

cv::Mat opt_harris(const cv::Mat &inp, double threshold_ratio);
double parse_opt_harris_config_args(const std::vector<std::string> &args);

#endif
