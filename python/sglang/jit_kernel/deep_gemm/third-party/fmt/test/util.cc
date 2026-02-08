// Formatting library for C++ - test utilities
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "util.h"

#include <cstring>

const char* const file_content = "Don't panic!";

fmt::buffered_file open_buffered_file(FILE** fp) {
#if FMT_USE_FCNTL
  auto pipe = fmt::pipe();
  pipe.write_end.write(file_content, std::strlen(file_content));
  pipe.write_end.close();
  fmt::buffered_file f = pipe.read_end.fdopen("r");
  if (fp) *fp = f.get();
#else
  fmt::buffered_file f("test-file", "w");
  fputs(file_content, f.get());
  if (fp) *fp = f.get();
#endif
  return f;
}

std::locale do_get_locale(const char* name) {
  try {
    return std::locale(name);
  } catch (const std::runtime_error&) {
  }
  return std::locale::classic();
}

std::locale get_locale(const char* name, const char* alt_name) {
  auto loc = do_get_locale(name);
  if (loc == std::locale::classic() && alt_name) loc = do_get_locale(alt_name);
#ifdef __OpenBSD__
  // Locales are not working in OpenBSD:
  // https://github.com/fmtlib/fmt/issues/3670.
  loc = std::locale::classic();
#endif
  if (loc == std::locale::classic())
    fmt::print(stderr, "{} locale is missing.\n", name);
  return loc;
}
