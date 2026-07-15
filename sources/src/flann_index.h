#ifndef H_FLANN_INDEX
#define H_FLANN_INDEX

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Builds a FLANN kd-tree over one feature column of the database and saves it.
// args: <db_path> <feat_name> <index_out_path>
void cmd_build_flann_index(const std::vector<std::string> &args);

// Retrieves with a pre-built FLANN index (O(log N) search) and returns a grid.
// args: <db_path> <index_path> <feat_name> <top_k> [image_dir]
cv::Mat cmd_retrieve_fast(const cv::Mat &query,
                          const std::vector<std::string> &args);

#endif
