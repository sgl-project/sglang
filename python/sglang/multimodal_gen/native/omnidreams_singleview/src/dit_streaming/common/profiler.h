// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// profiler.h -- Structured performance profiling accumulator for native.
//
// Collects named timing entries across multiple iterations, computes statistics,
// and serializes to JSON for baseline comparison. Header-only, zero external
// dependencies beyond the C++ standard library.
//
// Compile-time gating: all accumulator calls are no-ops unless OMNIDREAMS_SINGLEVIEW_PROFILE
// is defined. The standalone profiler (CMake build) always defines it; the Python
// extension build does not, ensuring zero overhead in production.
//
// This header is independent of profile_config.h, which provides the runtime
// g_wan_profile_level mechanism for the existing printf-based profiling path.
// Both can coexist: the printf path is controlled at runtime, while this
// accumulator path is controlled at compile time.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace omnidreams_singleview {

struct ProfileStats {
    float min_ms;
    float max_ms;
    float mean_ms;
    float median_ms;
    float p95_ms;
    float stddev_ms;
    int   count;
};

inline ProfileStats compute_stats(std::vector<float> vals) {
    ProfileStats s{};
    if (vals.empty()) return s;
    s.count = static_cast<int>(vals.size());
    std::sort(vals.begin(), vals.end());
    s.min_ms = vals.front();
    s.max_ms = vals.back();
    double sum = 0.0;
    for (float v : vals) sum += v;
    s.mean_ms = static_cast<float>(sum / vals.size());
    s.median_ms = (vals.size() % 2 == 1)
        ? vals[vals.size() / 2]
        : (vals[vals.size() / 2 - 1] + vals[vals.size() / 2]) * 0.5f;
    size_t p95_idx = static_cast<size_t>(std::ceil(vals.size() * 0.95)) - 1;
    if (p95_idx >= vals.size()) p95_idx = vals.size() - 1;
    s.p95_ms = vals[p95_idx];
    double var = 0.0;
    for (float v : vals) {
        double d = v - s.mean_ms;
        var += d * d;
    }
    s.stddev_ms = static_cast<float>(std::sqrt(var / vals.size()));
    return s;
}

// Category -> Component -> per-iteration ms values.
using TimingMap = std::map<std::string, std::map<std::string, std::vector<float>>>;

class ProfileAccumulator {
public:
    static ProfileAccumulator& instance() {
        static ProfileAccumulator inst;
        return inst;
    }

    void record(const std::string& category, const std::string& component, float ms) {
        std::lock_guard<std::mutex> lock(mu_);
        data_[category][component].push_back(ms);
    }

    void new_iteration() {
        std::lock_guard<std::mutex> lock(mu_);
        iteration_count_++;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        data_.clear();
        iteration_count_ = 0;
    }

    int iteration_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return iteration_count_;
    }

    TimingMap data() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_;
    }

    // Metadata fields -- set by the harness before dump.
    struct Metadata {
        std::string gpu_name;
        std::string cuda_version;
        int warmup_iters = 0;
        int timing_iters = 0;
        int profile_level = 0;
    };
    Metadata metadata;

    struct ModelConfig {
        std::string preset;
        int N = 0, T = 0, H = 0, W = 0;
        int M = 0, K = 0, H_heads = 0, D = 0, FF = 0;
        int num_layers = 0;
    };
    ModelConfig model_config;

    bool dump_json(const std::string& path) const {
        TimingMap data_snapshot = data();
        FILE* f = fopen(path.c_str(), "w");
        if (!f) return false;

        auto timestamp_str = [] {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char buf[64];
            std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
            return std::string(buf);
        };

        fprintf(f, "{\n");

        // Metadata
        fprintf(f, "  \"metadata\": {\n");
        fprintf(f, "    \"gpu\": \"%s\",\n", metadata.gpu_name.c_str());
        fprintf(f, "    \"cuda_version\": \"%s\",\n", metadata.cuda_version.c_str());
        fprintf(f, "    \"timestamp\": \"%s\",\n", timestamp_str().c_str());
        fprintf(f, "    \"warmup_iters\": %d,\n", metadata.warmup_iters);
        fprintf(f, "    \"timing_iters\": %d,\n", metadata.timing_iters);
        fprintf(f, "    \"profile_level\": %d\n", metadata.profile_level);
        fprintf(f, "  },\n");

        // Model config
        fprintf(f, "  \"model_config\": {\n");
        fprintf(f, "    \"preset\": \"%s\",\n", model_config.preset.c_str());
        fprintf(f, "    \"N\": %d, \"T\": %d, \"H\": %d, \"W\": %d,\n",
                model_config.N, model_config.T, model_config.H, model_config.W);
        fprintf(f, "    \"M\": %d, \"K\": %d, \"H_heads\": %d, \"D\": %d, \"FF\": %d,\n",
                model_config.M, model_config.K, model_config.H_heads, model_config.D, model_config.FF);
        fprintf(f, "    \"num_layers\": %d\n", model_config.num_layers);
        fprintf(f, "  },\n");

        // Timings
        fprintf(f, "  \"timings\": {\n");
        size_t cat_i = 0;
        for (auto& [cat, components] : data_snapshot) {
            fprintf(f, "    \"%s\": {\n", cat.c_str());
            size_t comp_i = 0;
            for (auto& [comp, vals] : components) {
                ProfileStats s = compute_stats(vals);
                fprintf(f, "      \"%s\": {\"mean\": %.4f, \"min\": %.4f, \"max\": %.4f, "
                        "\"median\": %.4f, \"p95\": %.4f, \"stddev\": %.4f, \"count\": %d}",
                        comp.c_str(), s.mean_ms, s.min_ms, s.max_ms,
                        s.median_ms, s.p95_ms, s.stddev_ms, s.count);
                fprintf(f, "%s\n", (++comp_i < components.size()) ? "," : "");
            }
            fprintf(f, "    }%s\n", (++cat_i < data_snapshot.size()) ? "," : "");
        }
        fprintf(f, "  }\n");

        fprintf(f, "}\n");
        fclose(f);
        return true;
    }

    void print_summary() const {
        TimingMap data_snapshot = data();
        std::printf("\n===== native Performance Summary =====\n");
        std::printf("GPU: %s | CUDA: %s | Warmup: %d | Iters: %d | Profile Level: %d\n",
                    metadata.gpu_name.c_str(), metadata.cuda_version.c_str(),
                    metadata.warmup_iters, metadata.timing_iters, metadata.profile_level);
        std::printf("Config: %s  N=%d T=%d H=%d W=%d  M=%d K=%d H=%d D=%d FF=%d  layers=%d\n",
                    model_config.preset.c_str(),
                    model_config.N, model_config.T, model_config.H, model_config.W,
                    model_config.M, model_config.K, model_config.H_heads, model_config.D,
                    model_config.FF, model_config.num_layers);
        std::printf("-----------------------------------------\n");
        for (auto& [cat, components] : data_snapshot) {
            std::printf("[%s]\n", cat.c_str());
            for (auto& [comp, vals] : components) {
                ProfileStats s = compute_stats(vals);
                std::printf("  %-24s  mean=%8.3f  min=%8.3f  max=%8.3f  median=%8.3f  p95=%8.3f  stddev=%6.3f ms  (n=%d)\n",
                            comp.c_str(), s.mean_ms, s.min_ms, s.max_ms,
                            s.median_ms, s.p95_ms, s.stddev_ms, s.count);
            }
        }
        std::printf("=========================================\n\n");
    }

private:
    ProfileAccumulator() = default;
    mutable std::mutex mu_;
    TimingMap data_;
    int iteration_count_ = 0;
};

// Compile-time gated macros. No-ops when OMNIDREAMS_SINGLEVIEW_PROFILE is not defined.
#ifdef OMNIDREAMS_SINGLEVIEW_PROFILE

#define OMNIDREAMS_SINGLEVIEW_PROF_RECORD(category, component, ms_val) \
    omnidreams_singleview::ProfileAccumulator::instance().record(category, component, ms_val)

#define OMNIDREAMS_SINGLEVIEW_PROF_NEW_ITER() \
    omnidreams_singleview::ProfileAccumulator::instance().new_iteration()

#define OMNIDREAMS_SINGLEVIEW_PROF_CLEAR() \
    omnidreams_singleview::ProfileAccumulator::instance().clear()

#else

#define OMNIDREAMS_SINGLEVIEW_PROF_RECORD(category, component, ms_val) ((void)0)
#define OMNIDREAMS_SINGLEVIEW_PROF_NEW_ITER()                          ((void)0)
#define OMNIDREAMS_SINGLEVIEW_PROF_CLEAR()                             ((void)0)

#endif

} // namespace omnidreams_singleview
