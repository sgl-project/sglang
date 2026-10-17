# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 mono decode layer: one sparse MoE layer of a decode / linear-verify
step in one persistent FlyDSL launch, both TP all-reduces in-kernel (SOURCE.md).
The host side is ``sglang.srt.models.minimax_m3_mono``."""
