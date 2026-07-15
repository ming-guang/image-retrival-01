#ifndef H_EVAL_MAP
#define H_EVAL_MAP

#include <string>
#include <vector>

// Computes Mean Average Precision at k = 3, 5, 11, 21 over the database, treating
// two images as relevant when they show the same building (per the dataset CSV).
// args: <db_path> <csv_path> <feat_name>
void cmd_eval_map(const std::vector<std::string> &args);

#endif
