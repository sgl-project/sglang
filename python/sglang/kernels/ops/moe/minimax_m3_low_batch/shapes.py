# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/minimax_m3/nvidia/specialized_decode/__init__.py @ddd43a54
"""What the MiniMax-M3.1-Flash low-batch FFN kernels are built for, importable
without cutlass: one MoE layer of one TP4 or TP8 rank, 1-32 rows."""

HIDDEN = 6144
EXPERTS = 128
TOP_K = 4
MAX_ROWS = 32
# Intermediate columns per rank: TP4, TP8.
COLUMNS = (768, 384)
# The SwiGLU-OAI constants FC1's epilogue and the shared activation are
# compiled with; callers check the layer's against them.
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
SWIGLU_BETA = 1.0
