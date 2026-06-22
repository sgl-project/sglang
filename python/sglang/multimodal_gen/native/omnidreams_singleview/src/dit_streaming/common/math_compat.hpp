// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <math.h>

// This header re-introduces the C99 single-precision math symbols inside std::
// so that third-party headers that expect std::xxx f() continue to compile
// under C++20/libstdc++ where those aliases were removed.

#if defined(__cplusplus)
namespace std {
using ::ceilf;
using ::cosf;
using ::expf;
using ::exp2f;
using ::floorf;
using ::logf;
using ::log10f;
using ::log2f;
using ::powf;
using ::rintf;
using ::sinf;
using ::sqrtf;
using ::tanhf;
using ::truncf;
} // namespace std
#endif
