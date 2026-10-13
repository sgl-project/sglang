# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/compare.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A check step's comparison of a mono tensor against its reference."""

import torch


def compare(got: torch.Tensor, ref: torch.Tensor) -> str:
    """How far ``got`` is from ``ref``: the fraction of elements that differ, the
    largest absolute difference and the relative norm of the difference."""
    got, ref = got.float(), ref.float()
    d = (got - ref).abs()
    differ = (d != 0).float().mean().item()
    rel = (d.norm() / ref.norm().clamp_min(1e-30)).item()
    return f"differ {differ:.4f} max {d.max().item():.3e} rel {rel:.1e}"
