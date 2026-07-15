#ifndef H_REGISTRY
#define H_REGISTRY

#include <functional>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using CommandFunc =
    std::function<cv::Mat(const cv::Mat &, const std::vector<std::string> &)>;

// Dataset commands run on the whole dataset and take raw args, no input image.
using DatasetCommandFunc = std::function<void(const std::vector<std::string> &)>;

extern std::unordered_map<std::string, CommandFunc> registry;
extern std::unordered_map<std::string, DatasetCommandFunc> dataset_registry;

template <typename T>
void register_command(std::string name, cv::Mat (*func)(const cv::Mat &, T),
                      T (*parser)(const std::vector<std::string> &)) {
  registry[name] = [func, parser, name](const cv::Mat &img,
                                        const std::vector<std::string> &args) {
    try {
      T config = parser(args);
      return func(img, config);
    } catch (const std::exception &e) {
      std::cerr << "[Error] Parsing failed for '" << name << "': " << e.what()
                << std::endl;
      return cv::Mat();
    }
  };
}

inline void register_command(std::string name,
                             cv::Mat (*func)(const cv::Mat &)) {
  registry[name] = [func](const cv::Mat &img,
                          const std::vector<std::string> &) {
    return func(img);
  };
}

// Registers a command that parses its own args vector (e.g. retrieve).
inline void register_command(
    std::string name,
    cv::Mat (*func)(const cv::Mat &, const std::vector<std::string> &)) {
  registry[name] = func;
}

// Registers a dataset command that takes only args and returns nothing.
inline void register_dataset_command(
    std::string name, void (*func)(const std::vector<std::string> &)) {
  dataset_registry[name] = func;
}
#endif
