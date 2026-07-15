#ifndef H_FEATURE_DB
#define H_FEATURE_DB

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// One database row: the image stem plus every feature vector extracted from it.
// Task-2 fields stay empty when the index is built in "basic" mode.
struct FeatureEntry {
  std::string filename; // stem only, e.g. "00001"
  cv::Mat color_hist;   // 1x96   CV_32F
  cv::Mat correlogram;  // 1x256  CV_32F
  cv::Mat sift_mean;    // 1x128  CV_32F
  cv::Mat orb_mean;     // 1x32   CV_32F
  cv::Mat hu_moments;   // 1x7    CV_32F
  cv::Mat lbp_hist;     // 1x256  CV_32F
  cv::Mat hog_desc;     // 1x3780 CV_32F
  cv::Mat combined;     // 1xN    CV_32F
};

// Extracts every feature from one image. When `full` is set it also computes the
// Task-2 features (Hu/LBP/HOG) and the combined vector; otherwise those stay empty.
FeatureEntry extract_all_features(const cv::Mat &bgr, bool full);

// Concatenates all L2-normalised sub-features of an entry into one vector.
cv::Mat build_combined(const FeatureEntry &e);

// Serialises the database to a YAML file (base64-encoded matrices).
void save_feature_db(const std::string &path,
                     const std::vector<FeatureEntry> &db);

// Loads a database previously written by save_feature_db.
std::vector<FeatureEntry> load_feature_db(const std::string &path);

// Returns the feature field named by feat_name from an entry.
cv::Mat get_feature(const FeatureEntry &e, const std::string &feat_name);

// Distance between two vectors using the metric appropriate for feat_name.
float compute_distance(const std::string &feat_name, const cv::Mat &a,
                       const cv::Mat &b);

// Score-level weighted fusion ("combinedw"): for each of the strong features
// (color_hist + correlogram + LBP) computes q-to-db distances with that
// feature's native metric, min-max normalises them across the db, then combines
// with MAP-derived weights. Returns one fused distance per db entry. Unlike the
// vector-concat "combined", this keeps chi-squared for the histograms and drops
// the weak features, so it beats every single feature instead of regressing.
std::vector<float> fused_distances(const FeatureEntry &q,
                                   const std::vector<FeatureEntry> &db);

#endif
