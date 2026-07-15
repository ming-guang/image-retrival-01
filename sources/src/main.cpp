#include "build_index.h"
#include "eval_map.h"
#include "flann_index.h"
#include "harris_corners.h"
#include "opt_harris.h"
#include "registry.h"
#include "retrieve.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

/*
 * Purpose: Entry point for the image processing command-line interface.
 * Registers available commands, parses command-line arguments,
 * loads the input image, executes the requested processing function,
 * and writes the result to the specified output file.
 * Input Parameters:
 * - argc: Integer representing the number of command-line arguments passed.
 * - argv: Array of C-style strings containing the command-line arguments.
 * Return Parameter:
 * - int: Returns 0 upon successful execution, or -1 if an error occurs
 * (e.g., missing arguments, unknown function, or file I/O failure).
 */
int main(int argc, char **argv) {
  /* Register available image processing commands and their argument parsers */
  register_command("harris", harris_corners, parse_harris_config_args);
  register_command("optHarris", opt_harris, parse_opt_harris_config_args);

  /* Image-retrieval commands that take a query image and return a result grid */
  register_command("retrieve", cmd_retrieve);
  register_command("retrieveFast", cmd_retrieve_fast);

  /* Dataset commands that operate on the whole dataset (no single input image) */
  register_dataset_command("buildIndex", cmd_build_index);
  register_dataset_command("buildFlannIndex", cmd_build_flann_index);
  register_dataset_command("evalMAP", cmd_eval_map);

  /* Need at least a function name and one argument */
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " -<func> <args...>" << std::endl;
    return -1;
  }

  /* Parse the function name, stripping the leading '-' if present */
  std::string func_name = argv[1];
  if (func_name.length() <= 0) {
    std::cerr << "invalid function name";
    return -1;
  }
  if (func_name[0] == '-') {
    func_name = func_name.substr(1);
  }

  /* Dataset commands: forward all remaining args, no image I/O here */
  if (dataset_registry.find(func_name) != dataset_registry.end()) {
    std::vector<std::string> args;
    for (int i = 2; i < argc; i++) {
      args.push_back(argv[i]);
    }
    dataset_registry[func_name](args);
    return 0;
  }

  /* Image commands require at least an input and output path */
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " -<func> <input> <output> [config...]"
              << std::endl;
    return -1;
  }

  /* Extract input and output file paths */
  std::string inp_path = argv[2];
  std::string out_path = argv[3];

  /* Collect any additional configuration arguments for the specific command */
  std::vector<std::string> config_args;
  for (int i = 4; i < argc; i++) {
    config_args.push_back(argv[i]);
  }

  /* Check if the requested command exists in the registry */
  if (registry.find(func_name) != registry.end()) {

    cv::Mat inp = cv::imread(inp_path);
    if (inp.empty()) {
      std::cerr << "Error: Could not open " << inp_path << std::endl;
      return -1;
    }

    /* Execute the registered command function via the registry map */
    cv::Mat out = registry[func_name](inp, config_args);

    /* Save the processed image if execution was successful */
    if (!out.empty()) {
      cv::imwrite(out_path, out);
      /* Status to stderr so stdout stays clean for machine-readable output */
      std::cerr << "Success: " << func_name << " applied. Saved to " << out_path
                << std::endl;
    }
  } else {
    std::cerr << "Error: Unknown function '" << func_name << "'" << std::endl;
    return -1;
  }

  return 0;
}
