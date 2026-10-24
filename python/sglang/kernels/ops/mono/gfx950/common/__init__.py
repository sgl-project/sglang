# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/__init__.py
"""Mono decode: a model's decode step as one persistent kernel per layer, with the
tensor-parallel reductions inside the kernel.

This package holds what every mono kernel family shares: ``plan`` (plain Python:
the execution model, region layouts, the hand-off trace) and ``device`` (traced inside
kernels: the tagged mailbox, wave / math primitives). The host side (TP consensus,
peer memory) is ``sglang.srt.layers.mono.runtime``. The kernels, their layouts and
schedules stay in the family's own package. It imports no model and holds no model
constant.
"""
