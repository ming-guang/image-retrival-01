#ifndef H_RETRIEVE
#define H_RETRIEVE

#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

// Ranks the whole database against a query image by brute-force distance and
// returns an annotated result grid image.
// args: <db_path> <feat_name> <top_k> [image_dir]
cv::Mat cmd_retrieve(const cv::Mat &query, const std::vector<std::string> &args);

// Draws the query and its ranked results as a labelled thumbnail grid, overlaying
// `header` (feature name + timing) on top. Shows it in a window only on a TTY.
cv::Mat build_result_grid(
    const cv::Mat &query,
    const std::vector<std::pair<float, std::string>> &ranked,
    const std::string &img_dir, const std::string &header);

// Prints ranked results as a JSON object to stdout (for GUIs / piping).
void print_results_json(
    const std::string &mode, const std::string &feature, int top_k,
    double time_ms, const std::string &img_dir,
    const std::vector<std::pair<float, std::string>> &ranked);

#endif
