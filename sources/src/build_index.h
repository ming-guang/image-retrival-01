#ifndef H_BUILD_INDEX
#define H_BUILD_INDEX

#include <string>
#include <vector>

// Extracts features from every PNG in a directory and writes a feature database.
// args: <image_dir> <db_out_path> [basic|all]  (default "basic").
void cmd_build_index(const std::vector<std::string> &args);

#endif
