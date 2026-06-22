// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/python.h>

// SageAttention-3's FP4 quantization kernels (scaled_fp4_quant / _permute /
// _trans) now live in sgl-kernel's sage3_ops library. This file used to #include
// the upstream fp4_quantization_4d.cu to compile them into the OmniDreams
// native extension directly. It now just pulls the symbols in via a cross-.so
// link (sgl_sage3_shim.cuh declares the exported ::scaled_fp4_quant_*); the
// singleview_loader links sage3_ops.so.
#include "sgl_sage3_shim.cuh"
