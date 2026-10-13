# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/plan/__init__.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build-time contracts of a mono kernel, plain Python (no torch, no FlyDSL):
``trace`` records a kernel's mailbox accesses while it is traced, ``check`` proves
the recorded hand-offs against the regions' declarations."""
