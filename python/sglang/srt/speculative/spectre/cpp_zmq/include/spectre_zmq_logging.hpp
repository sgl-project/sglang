#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "spectre_protocol.hpp"

void spectre_set_log_level(int level);
int spectre_log_level();
bool spectre_info_enabled();
bool spectre_warn_enabled();
bool spectre_debug_enabled();
void spectre_enqueue_log(std::string msg);

template <typename... Args>
void spectre_log(bool enabled, const Args &...args) {
  if (!enabled) {
    return;
  }

  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::tm tm_buf;
#if defined(_WIN32)
  localtime_s(&tm_buf, &tt);
#else
  localtime_r(&tt, &tm_buf);
#endif

  std::ostringstream oss;
  oss << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "."
      << std::setw(3) << std::setfill('0') << ms.count() << "] ";
  (oss << ... << args);
  spectre_enqueue_log(oss.str());
}

template <typename... Args> void spectre_debug_log(const Args &...args) {
  spectre_log(spectre_debug_enabled(), args...);
}

template <typename... Args> void spectre_info_log(const Args &...args) {
  spectre_log(spectre_info_enabled(), args...);
}

template <typename... Args> void spectre_warn_log(const Args &...args) {
  spectre_log(spectre_warn_enabled(), args...);
}

inline long long
spectre_duration_us(const std::chrono::steady_clock::time_point &start,
                    const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

inline double spectre_now_us() {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

inline void stamp_request_batch(std::vector<spectre::SpectreRequest> &reqs,
                                double spectre::SpectreRequest::*field,
                                double timestamp_us = spectre_now_us()) {
  for (auto &req : reqs) {
    req.*field = timestamp_us;
  }
}
