# SPDX-License-Identifier: Apache-2.0
"""Shared device primitives of the HIP monokernels (RFC #43949): a resident grid
with one workgroup per CU (``plan``), wave ops and buffer access (``ops``), MX
helpers (``mx``), device-incremented epoch tags (``epoch``), the tagged mailbox
between workgroups and GPUs (``sync``), and the deadlines and error words every
wait runs under (``guard``).

Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6 (MIT License,
Copyright (C) 2026, Advanced Micro Devices, Inc.) by way of vLLM's DeepSeek-V4.1
mono decode; each file names its origin.
"""
