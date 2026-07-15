#include "build_index.h"
#include "feature_db.h"
#include "image_features.h"
#include "io_utils.h"
#include "path_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <iostream>

void cmd_build_index(const std::vector<std::string> &args) {
  if (args.size() < 2) {
    std::cerr << "Usage: -buildIndex <image_dir> <db_out> [basic|all]\n";
    return;
  }
  std::string img_dir = args[0];
  std::string db_path = args[1];
  bool full = (args.size() > 2 && args[2] == "all");

  std::vector<cv::String> files;
  cv::glob(img_dir + "/*.png", files, false);
  if (files.empty()) {
    std::cerr << "No PNG images found in " << img_dir << "\n";
    return;
  }

  std::vector<FeatureEntry> db;
  db.reserve(files.size());
  for (size_t i = 0; i < files.size(); i++) {
    cv::Mat img = cv::imread(files[i]);
    if (img.empty()) {
      std::cerr << "  skip (unreadable): " << files[i] << "\n";
      continue;
    }
    // Preprocessing only in "all" mode, matching how the query is treated later.
    cv::Mat proc = full ? preprocess_image(img) : img;
    FeatureEntry e = extract_all_features(proc, full);
    e.filename = path_stem(files[i]);
    db.push_back(e);
    // Progress goes to stderr so stdout stays clean for machine consumers.
    if ((i + 1) % 50 == 0 || i + 1 == files.size())
      std::cerr << "  indexed " << (i + 1) << "/" << files.size() << "\r"
                << std::flush;
  }
  std::cerr << "\n";
  save_feature_db(db_path, db);
  if (stdout_is_tty()) {
    std::cout << "Saved " << db.size() << " entries to " << db_path
              << (full ? " (all features)" : " (basic features)") << "\n";
  } else {
    std::printf("{\"db\": \"%s\", \"entries\": %d, \"mode\": \"%s\"}\n",
                json_escape(db_path).c_str(), (int)db.size(),
                full ? "all" : "basic");
  }
}
