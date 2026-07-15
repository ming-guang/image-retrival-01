#include "feature_db.h"
#include "image_features.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cfloat>
#include <iostream>
#include <stdexcept>

FeatureEntry extract_all_features(const cv::Mat &bgr, bool full) {
  FeatureEntry e;
  e.color_hist = compute_color_hist(bgr);
  e.correlogram = compute_correlogram(bgr);
  e.sift_mean = compute_sift_mean(bgr);
  e.orb_mean = compute_orb_mean(bgr);
  if (full) {
    e.hu_moments = compute_hu_moments(bgr);
    e.lbp_hist = compute_lbp_hist(bgr);
    e.hog_desc = compute_hog_desc(bgr);
    e.combined = build_combined(e);
  }
  return e;
}

cv::Mat build_combined(const FeatureEntry &e) {
  // Normalise each block to unit L2 length so no single feature dominates.
  auto unit = [](const cv::Mat &m) {
    cv::Mat n;
    cv::normalize(m, n, 1.0, 0.0, cv::NORM_L2);
    return n;
  };
  std::vector<cv::Mat> parts = {unit(e.color_hist),  unit(e.correlogram),
                                unit(e.sift_mean),   unit(e.orb_mean),
                                unit(e.hu_moments),  unit(e.lbp_hist),
                                unit(e.hog_desc)};
  cv::Mat out;
  cv::hconcat(parts, out);
  return out;
}

void save_feature_db(const std::string &path,
                     const std::vector<FeatureEntry> &db) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::BASE64);
  if (!fs.isOpened())
    throw std::runtime_error("cannot open db for writing: " + path);
  fs << "n_entries" << (int)db.size();
  for (size_t i = 0; i < db.size(); i++) {
    fs << ("entry_" + std::to_string(i)) << "{";
    fs << "filename" << db[i].filename;
    fs << "color_hist" << db[i].color_hist;
    fs << "correlogram" << db[i].correlogram;
    fs << "sift_mean" << db[i].sift_mean;
    fs << "orb_mean" << db[i].orb_mean;
    fs << "hu_moments" << db[i].hu_moments;
    fs << "lbp_hist" << db[i].lbp_hist;
    fs << "hog_desc" << db[i].hog_desc;
    fs << "combined" << db[i].combined;
    fs << "}";
  }
  fs.release();
}

std::vector<FeatureEntry> load_feature_db(const std::string &path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened())
    throw std::runtime_error("cannot open db for reading: " + path);
  int n = (int)fs["n_entries"];
  std::vector<FeatureEntry> db;
  db.reserve(n);
  for (int i = 0; i < n; i++) {
    cv::FileNode node = fs["entry_" + std::to_string(i)];
    FeatureEntry e;
    node["filename"] >> e.filename;
    node["color_hist"] >> e.color_hist;
    node["correlogram"] >> e.correlogram;
    node["sift_mean"] >> e.sift_mean;
    node["orb_mean"] >> e.orb_mean;
    node["hu_moments"] >> e.hu_moments;
    node["lbp_hist"] >> e.lbp_hist;
    node["hog_desc"] >> e.hog_desc;
    node["combined"] >> e.combined;
    db.push_back(e);
  }
  fs.release();
  return db;
}

cv::Mat get_feature(const FeatureEntry &e, const std::string &feat_name) {
  if (feat_name == "color_hist")
    return e.color_hist;
  if (feat_name == "correlogram")
    return e.correlogram;
  if (feat_name == "sift")
    return e.sift_mean;
  if (feat_name == "orb")
    return e.orb_mean;
  if (feat_name == "hu")
    return e.hu_moments;
  if (feat_name == "lbp")
    return e.lbp_hist;
  if (feat_name == "hog")
    return e.hog_desc;
  if (feat_name == "combined")
    return e.combined;
  throw std::runtime_error("unknown feature name: " + feat_name);
}

float compute_distance(const std::string &feat_name, const cv::Mat &a,
                       const cv::Mat &b) {
  // Histogram-like features use chi-squared; descriptor means use L2.
  if (feat_name == "color_hist" || feat_name == "lbp")
    return dist_chi2(a, b);
  return dist_l2(a, b);
}

// Features fused by "combinedw" with their weights. Weights are each feature's
// standalone MAP@3 on the preprocessed database, so stronger features count for
// more. The weak features (SIFT/ORB/Hu/HOG) are deliberately excluded.
static const std::pair<const char *, float> kFusion[] = {
    {"color_hist", 0.3754f}, {"correlogram", 0.3190f}, {"lbp", 0.2819f}};

std::vector<float> fused_distances(const FeatureEntry &q,
                                   const std::vector<FeatureEntry> &db) {
  int n = (int)db.size();
  std::vector<float> fused(n, 0.0f);
  for (const auto &fw : kFusion) {
    std::string feat = fw.first;
    float w = fw.second;
    cv::Mat qf = get_feature(q, feat);
    // Per-feature distances to the whole db, then min-max to [0,1] so the
    // chi-squared and L2 scales become comparable before the weighted sum.
    std::vector<float> d(n);
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (int j = 0; j < n; j++) {
      d[j] = compute_distance(feat, qf, get_feature(db[j], feat));
      mn = std::min(mn, d[j]);
      mx = std::max(mx, d[j]);
    }
    float range = (mx - mn > 1e-12f) ? (mx - mn) : 1.0f;
    for (int j = 0; j < n; j++)
      fused[j] += w * (d[j] - mn) / range;
  }
  return fused;
}
