#include "eval_map.h"
#include "feature_db.h"
#include "io_utils.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

// Trims surrounding whitespace from a string.
static std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos)
    return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Reads DATASET SPLIT.csv and maps each image stem ("%05d") to its building name.
static std::unordered_map<std::string, std::string>
load_building_map(const std::string &csv_path) {
  std::unordered_map<std::string, std::string> out;
  std::ifstream in(csv_path);
  if (!in.is_open()) {
    std::cerr << "cannot open csv: " << csv_path << "\n";
    return out;
  }
  std::string line;
  std::getline(in, line); // skip header
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string pic, building;
    if (!std::getline(ss, pic, ','))
      continue;
    if (!std::getline(ss, building, ','))
      continue;
    pic = trim(pic);
    building = trim(building);
    if (pic.empty() || building.empty())
      continue;
    char stem[16];
    std::snprintf(stem, sizeof(stem), "%05d", std::atoi(pic.c_str()));
    out[stem] = building;
  }
  return out;
}

void cmd_eval_map(const std::vector<std::string> &args) {
  if (args.size() < 3) {
    std::cerr << "Usage: -evalMAP <db_path> <csv_path> <feat_name>\n";
    return;
  }
  std::string db_path = args[0], csv_path = args[1], feat = args[2];

  std::vector<FeatureEntry> db = load_feature_db(db_path);
  std::unordered_map<std::string, std::string> building = load_building_map(csv_path);
  if (building.empty())
    return;

  // Build the evaluation pool: db entries that have a known building, with the
  // building mapped to a small integer id for fast comparison. Keep the full
  // entries so the fused "combinedw" feature can read several fields per image.
  bool fused = (feat == "combinedw");
  std::unordered_map<std::string, int> name_to_id;
  std::vector<FeatureEntry> pool;
  std::vector<cv::Mat> feats;
  std::vector<int> cls;
  for (const auto &e : db) {
    auto it = building.find(e.filename);
    if (it == building.end())
      continue;
    int id;
    auto nit = name_to_id.find(it->second);
    if (nit == name_to_id.end())
      id = name_to_id[it->second] = (int)name_to_id.size();
    else
      id = nit->second;
    pool.push_back(e);
    feats.push_back(fused ? cv::Mat() : get_feature(e, feat));
    cls.push_back(id);
  }
  int n = (int)pool.size();
  if (n == 0) {
    std::cerr << "No labelled images matched the database.\n";
    return;
  }

  // Relevant-count per class (so R_q = class_count[cls] - 1 excluding the query).
  std::unordered_map<int, int> class_count;
  for (int c : cls)
    class_count[c]++;

  const int K[] = {3, 5, 11, 21};
  double map_sum[4] = {0, 0, 0, 0};
  int n_queries = 0; // queries that have at least one relevant match

  for (int q = 0; q < n; q++) {
    // For the fused feature, score the whole pool in one normalised pass.
    std::vector<float> fd;
    if (fused)
      fd = fused_distances(pool[q], pool);
    std::vector<std::pair<float, int>> scored; // (distance, class)
    scored.reserve(n - 1);
    for (int j = 0; j < n; j++) {
      if (j == q)
        continue;
      float dist = fused ? fd[j] : compute_distance(feat, feats[q], feats[j]);
      scored.push_back({dist, cls[j]});
    }
    std::sort(scored.begin(), scored.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                return a.first < b.first;
              });
    int relevant = class_count[cls[q]] - 1;
    if (relevant <= 0)
      continue; // singleton building contributes no AP
    n_queries++;

    // One sorted ranking serves every k; accumulate AP@k as we walk down it.
    for (int ki = 0; ki < 4; ki++) {
      int k = K[ki];
      int limit = std::min(k, (int)scored.size());
      int hits = 0;
      double ap = 0.0;
      for (int i = 0; i < limit; i++)
        if (scored[i].second == cls[q]) {
          hits++;
          ap += (double)hits / (i + 1);
        }
      map_sum[ki] += ap / std::min(relevant, k);
    }
  }

  int denom = std::max(1, n_queries);
  if (stdout_is_tty()) {
    std::cout << "MAP over " << n_queries << " queries (" << n << " images, "
              << name_to_id.size() << " buildings), feature [" << feat << "]:\n";
    for (int ki = 0; ki < 4; ki++)
      std::cout << "  MAP@" << K[ki] << "\t= " << (map_sum[ki] / denom) << "\n";
  } else {
    std::printf("{\n");
    std::printf("  \"feature\": \"%s\",\n", json_escape(feat).c_str());
    std::printf("  \"n_queries\": %d,\n", n_queries);
    std::printf("  \"n_images\": %d,\n", n);
    std::printf("  \"n_buildings\": %d,\n", (int)name_to_id.size());
    std::printf("  \"map\": {");
    for (int ki = 0; ki < 4; ki++)
      std::printf("\"%d\": %.6f%s", K[ki], map_sum[ki] / denom,
                  (ki < 3) ? ", " : "");
    std::printf("}\n}\n");
  }
}
