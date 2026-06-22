// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

namespace omnidreams_singleview {

// Profiling levels:
//  0: off
//  1: coarse per-forward CUDA-event breakdown (transformer_forward phases)
//  2+: reserved for finer-grained breakdowns (e.g. per-block sections)
inline std::atomic<int> g_wan_profile_level{0};

// Print every N forward calls (each CFG step calls forward twice).
inline std::atomic<int> g_wan_profile_print_every{1};

// Internal call counter for rate limiting.
inline std::atomic<long long> g_wan_profile_call_idx{0};

} // namespace omnidreams_singleview
