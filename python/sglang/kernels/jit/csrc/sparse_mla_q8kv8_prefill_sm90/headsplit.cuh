/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "etap.cuh"

namespace sglang {
template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
using SparseMlaQ8Kv8HeadSplitKernel = SparseMlaQ8Kv8EtapKernel<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK, 32>;
}
