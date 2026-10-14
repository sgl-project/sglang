# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/__init__.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side mechanisms of a mono runner, one module each (imported directly, so
the ones without an AITER dependency load without it): ``consensus`` (TP-uniform
decisions) and ``peer_memory`` (symmetric peer buffers)."""
