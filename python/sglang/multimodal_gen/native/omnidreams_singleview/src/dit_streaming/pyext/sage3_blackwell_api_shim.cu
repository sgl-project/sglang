// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/python.h>

// SageAttention-3's Blackwell FP4 attention (mha_fwd) now lives in sgl-kernel's
// sage3_ops library (contributed there; see sgl-kernel/csrc/attention/sage3.cu).
// This file used to #include the upstream api.cu to compile mha_fwd into the
// OmniDreams native extension directly. It now just pulls the symbol in via a
// cross-.so link (sgl_sage3_shim.cuh declares the exported ::mha_fwd); the
// singleview_loader links sage3_ops.so.
#include "sgl_sage3_shim.cuh"
