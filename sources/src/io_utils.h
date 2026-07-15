#ifndef H_IO_UTILS
#define H_IO_UTILS

#include <cstdio>
#include <string>
#include <unistd.h>

// True when stdout is a terminal; false when piped/redirected (machine consumer).
inline bool stdout_is_tty() { return isatty(fileno(stdout)) != 0; }

// Escapes a string so it is safe to embed inside a JSON double-quoted value.
inline std::string json_escape(const std::string &s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
    case '"': o += "\\\""; break;
    case '\\': o += "\\\\"; break;
    case '\n': o += "\\n"; break;
    case '\r': o += "\\r"; break;
    case '\t': o += "\\t"; break;
    default: o += c;
    }
  }
  return o;
}

#endif
