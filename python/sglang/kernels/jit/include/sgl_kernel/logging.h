#pragma once

#include <sgl_kernel/utils.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

namespace sglang::host {

enum class LogLevel : uint8_t {
  DEBUG_ = 0,
  INFO_ = 1,
  WARNING_ = 2,
  ERROR_ = 3,
  DEFAULT = WARNING_,
};

inline const LogLevel kLogLevel = [] {
  const auto env = std::getenv("SGLANG_JIT_LOG_LEVEL");
  if (!env) return LogLevel::DEFAULT;
  const auto str = std::string_view{env};
  if (str == "DEBUG" || str == "debug") return LogLevel::DEBUG_;
  if (str == "INFO" || str == "info") return LogLevel::INFO_;
  if (str == "WARNING" || str == "warning") return LogLevel::WARNING_;
  if (str == "ERROR" || str == "error") return LogLevel::ERROR_;
  std::clog << "[WARNING] unrecognized SGLANG_JIT_LOG_LEVEL value '" << str << "'; using default log level."
            << std::endl;
  return LogLevel::DEFAULT;
}();

struct Logger {
  std::ostringstream stream;
  explicit Logger(LogLevel level, DebugInfo location = {}) {
    static const std::string_view kPrefix[] = {"[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]"};
    stream << kPrefix[static_cast<size_t>(level)] << " [" << location.file_name() << ":" << location.line() << "] ";
  }
  ~Logger() {
    std::clog << std::move(stream).str() << std::endl;
  }
};

#define SGL_LOG(LEVEL)                                                                                   \
  if (const auto _log_level = ::sglang::host::LogLevel::LEVEL; _log_level < ::sglang::host::kLogLevel) { \
  } else                                                                                                 \
    ::sglang::host::Logger(_log_level).stream

#define SGL_LOG_DEBUG SGL_LOG(DEBUG_)
#define SGL_LOG_INFO SGL_LOG(INFO_)
#define SGL_LOG_WARNING SGL_LOG(WARNING_)

}  // namespace sglang::host
