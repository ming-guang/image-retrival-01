#include "retrieve.h"
#include "feature_db.h"
#include "image_features.h"
#include "io_utils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

static const std::string DEFAULT_IMG_DIR = "datasets/TMBuD/images";

cv::Mat build_result_grid(
    const cv::Mat &query,
    const std::vector<std::pair<float, std::string>> &ranked,
    const std::string &img_dir, const std::string &header) {
  const int THUMB = 128;
  const int COLS = 7;          // result columns per row
  const int HEADER = 24;       // header strip height
  int k = (int)ranked.size();
  int result_rows = (k + COLS - 1) / COLS;
  int total_rows = std::max(1, result_rows);

  int grid_w = COLS * THUMB;
  int grid_h = HEADER + THUMB /*query row*/ + total_rows * THUMB;
  cv::Mat grid(grid_h, grid_w, CV_8UC3, cv::Scalar(30, 30, 30));

  // Header strip with feature + timing text.
  cv::putText(grid, header, cv::Point(6, 17), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

  // Query thumbnail (top-left of the query row).
  cv::Mat qthumb;
  cv::resize(query, qthumb, cv::Size(THUMB, THUMB));
  qthumb.copyTo(grid(cv::Rect(0, HEADER, THUMB, THUMB)));
  cv::putText(grid, "QUERY", cv::Point(4, HEADER + THUMB - 6),
              cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1,
              cv::LINE_AA);

  int y0 = HEADER + THUMB;
  for (int r = 0; r < k; r++) {
    int col = r % COLS, row = r / COLS;
    cv::Rect cell(col * THUMB, y0 + row * THUMB, THUMB, THUMB);
    cv::Mat img = cv::imread(img_dir + "/" + ranked[r].second + ".png");
    if (!img.empty()) {
      cv::Mat thumb;
      cv::resize(img, thumb, cv::Size(THUMB, THUMB));
      thumb.copyTo(grid(cell));
    }
    std::ostringstream lbl;
    lbl << "#" << (r + 1) << " " << std::fixed << std::setprecision(2)
        << ranked[r].first;
    cv::putText(grid, lbl.str(), cv::Point(cell.x + 3, cell.y + THUMB - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1,
                cv::LINE_AA);
  }

  // Show a window only for interactive use; piped/GUI callers get JSON instead.
  if (stdout_is_tty()) {
    try {
      cv::imshow("Retrieval Results", grid);
      cv::waitKey(0);
    } catch (const cv::Exception &) {
    }
  }
  return grid;
}

void print_results_json(
    const std::string &mode, const std::string &feature, int top_k,
    double time_ms, const std::string &img_dir,
    const std::vector<std::pair<float, std::string>> &ranked) {
  std::printf("{\n");
  std::printf("  \"mode\": \"%s\",\n", json_escape(mode).c_str());
  std::printf("  \"feature\": \"%s\",\n", json_escape(feature).c_str());
  std::printf("  \"top_k\": %d,\n", top_k);
  std::printf("  \"time_ms\": %.3f,\n", time_ms);
  std::printf("  \"count\": %d,\n", (int)ranked.size());
  std::printf("  \"results\": [\n");
  for (size_t i = 0; i < ranked.size(); i++) {
    std::string path = img_dir + "/" + ranked[i].second + ".png";
    std::printf(
        "    {\"rank\": %d, \"file\": \"%s\", \"path\": \"%s\", "
        "\"distance\": %.6f}%s\n",
        (int)i + 1, json_escape(ranked[i].second).c_str(),
        json_escape(path).c_str(), ranked[i].first,
        (i + 1 < ranked.size()) ? "," : "");
  }
  std::printf("  ]\n}\n");
}

cv::Mat cmd_retrieve(const cv::Mat &query,
                     const std::vector<std::string> &args) {
  if (args.size() < 3) {
    std::cerr << "Usage: -retrieve <query> <out> <db> <feat> <top_k> "
                 "[image_dir]\n";
    return cv::Mat();
  }
  std::string db_path = args[0];
  std::string feat = args[1];
  int top_k = std::stoi(args[2]);
  std::string img_dir = (args.size() > 3) ? args[3] : DEFAULT_IMG_DIR;

  std::vector<FeatureEntry> db = load_feature_db(db_path);
  // Preprocess + extract to match how "all" databases were built.
  cv::Mat proc = preprocess_image(query);
  FeatureEntry q = extract_all_features(proc, true);

  // Time only the brute-force scan + sort (the comparable "query time").
  double t0 = (double)cv::getTickCount();
  std::vector<std::pair<float, std::string>> scored;
  scored.reserve(db.size());
  if (feat == "combinedw") {
    // Weighted score-level fusion of the strong features.
    std::vector<float> fd = fused_distances(q, db);
    for (size_t j = 0; j < db.size(); j++)
      scored.push_back({fd[j], db[j].filename});
  } else {
    cv::Mat q_desc = get_feature(q, feat);
    for (const auto &e : db)
      scored.push_back({compute_distance(feat, q_desc, get_feature(e, feat)),
                        e.filename});
  }
  int k = std::min(top_k, (int)scored.size());
  std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
  scored.resize(k);
  double ms = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;

  std::ostringstream hdr;
  hdr << "brute-force | " << feat << " | top=" << top_k << " | "
      << std::fixed << std::setprecision(2) << ms << " ms";
  if (stdout_is_tty()) {
    std::cout << hdr.str() << "\n";
    for (int i = 0; i < k; i++)
      std::cout << "  #" << (i + 1) << " " << scored[i].second << "  d="
                << scored[i].first << "\n";
  } else {
    print_results_json("brute-force", feat, top_k, ms, img_dir, scored);
  }

  return build_result_grid(query, scored, img_dir, hdr.str());
}
