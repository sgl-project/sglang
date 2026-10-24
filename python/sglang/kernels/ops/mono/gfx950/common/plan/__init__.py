# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/plan/__init__.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build-time plain Python of a mono kernel (no torch, no FlyDSL): the execution
model, region layouts, and ``trace``, which records a kernel's mailbox
accesses while it is traced."""
