#ifndef H_PATH_UTILS
#define H_PATH_UTILS

#include <string>

// Returns the filename without directory or extension, e.g. "a/b/00001.png" -> "00001".
inline std::string path_stem(const std::string &path) {
  size_t slash = path.find_last_of("/\\");
  size_t start = (slash == std::string::npos) ? 0 : slash + 1;
  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot < start)
    dot = path.size();
  return path.substr(start, dot - start);
}

#endif
