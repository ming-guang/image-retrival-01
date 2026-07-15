#include "flann_index.h"
#include "feature_db.h"
#include "image_features.h"
#include "io_utils.h"
#include "retrieve.h"
#include <opencv2/flann.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

static const std::string DEFAULT_IMG_DIR = "datasets/TMBuD/images";

// Stacks one feature column of the whole database into an N x D CV_32F matrix.
static cv::Mat stack_features(const std::vector<FeatureEntry> &db,
                              const std::string &feat) {
  cv::Mat data;
  for (const auto &e : db)
    data.push_back(get_feature(e, feat));
  return data; // already one row per entry
}

void cmd_build_flann_index(const std::vector<std::string> &args) {
  if (args.size() < 3) {
    std::cerr << "Usage: -buildFlannIndex <db_path> <feat_name> <index_out>\n";
    return;
  }
  std::vector<FeatureEntry> db = load_feature_db(args[0]);
  std::string feat = args[1], index_path = args[2];
  if (feat == "combinedw") {
    std::cerr << "combinedw fuses multiple metrics and has no single FLANN "
                 "index; use brute-force retrieval instead.\n";
    return;
  }
  cv::Mat data = stack_features(db, feat);
  // 4 randomised kd-trees: a good speed/accuracy trade-off for L2 features.
  cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(4));
  idx.save(index_path);
  std::cerr << "FLANN index saved to " << index_path << " (" << data.rows
            << " vectors, dim=" << data.cols << ")\n";
}

cv::Mat cmd_retrieve_fast(const cv::Mat &query,
                          const std::vector<std::string> &args) {
  if (args.size() < 4) {
    std::cerr << "Usage: -retrieveFast <query> <out> <db> <index> <feat> "
                 "<top_k> [image_dir]\n";
    return cv::Mat();
  }
  std::string db_path = args[0], index_path = args[1], feat = args[2];
  int top_k = std::stoi(args[3]);
  std::string img_dir = (args.size() > 4) ? args[4] : DEFAULT_IMG_DIR;

  std::vector<FeatureEntry> db = load_feature_db(db_path);
  cv::Mat proc = preprocess_image(query);
  FeatureEntry q = extract_all_features(proc, true);

  std::vector<std::pair<float, std::string>> ranked;
  std::string mode, hdr_label;
  double ms;

  if (feat == "combinedw") {
    // combinedw fuses several per-query metrics, so there is no single kd-tree
    // to search; fall back to the (still fast) brute-force fusion.
    double t0 = (double)cv::getTickCount();
    std::vector<float> fd = fused_distances(q, db);
    for (size_t j = 0; j < db.size(); j++)
      ranked.push_back({fd[j], db[j].filename});
    int k = std::min(top_k, (int)ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());
    ranked.resize(k);
    ms = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;
    mode = "brute-force";
    hdr_label = "brute-force fusion (FLANN N/A)";
  } else {
    // The saved index only stores the tree structure, so reattach the data.
    // Use build()/load() on one object (assigning a cv::flann::Index does not
    // transfer its internal tree and crashes on search).
    cv::Mat data = stack_features(db, feat);
    cv::flann::Index idx;
    std::ifstream probe(index_path.c_str(), std::ios::binary);
    bool have_index = probe.good();
    probe.close();
    bool loaded = false;
    if (have_index) {
      try {
        loaded = idx.load(data, index_path);
      } catch (const cv::Exception &) {
        loaded = false; // stale/corrupt/mismatched file -> rebuild below
      }
    }
    if (!loaded) {
      // No usable index yet: build one now and cache it for later queries.
      idx.build(data, cv::flann::KDTreeIndexParams(4));
      idx.save(index_path);
      std::cerr << "built FLANN index (" << data.rows << "x" << data.cols
                << ") -> " << index_path << "\n";
    }
    cv::Mat q_desc = get_feature(q, feat);
    if (q_desc.type() != CV_32F)
      q_desc.convertTo(q_desc, CV_32F);
    int k = std::min(top_k, data.rows);
    cv::Mat indices, dists;
    double t0 = (double)cv::getTickCount();
    idx.knnSearch(q_desc, indices, dists, k, cv::flann::SearchParams(64));
    ms = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;
    // FLANN returns squared L2 distances; sqrt matches the brute-force scale.
    for (int i = 0; i < k; i++) {
      int idx_i = indices.at<int>(0, i);
      if (idx_i < 0 || idx_i >= (int)db.size())
        continue;
      ranked.push_back({std::sqrt(dists.at<float>(0, i)), db[idx_i].filename});
    }
    mode = "flann";
    hdr_label = "FLANN kd-tree";
  }

  std::ostringstream hdr;
  hdr << hdr_label << " | " << feat << " | top=" << top_k << " | " << std::fixed
      << std::setprecision(3) << ms << " ms";
  if (stdout_is_tty()) {
    std::cout << hdr.str() << "\n";
    for (size_t i = 0; i < ranked.size(); i++)
      std::cout << "  #" << (i + 1) << " " << ranked[i].second
                << "  d=" << ranked[i].first << "\n";
  } else {
    print_results_json(mode, feat, top_k, ms, img_dir, ranked);
  }

  return build_result_grid(query, ranked, img_dir, hdr.str());
}
