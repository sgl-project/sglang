# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm-project/vllm @ 193922d6ff: vllm/models/deepseek_v41/amd/mono/common/__init__.py
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/__init__.py
"""The mono kernels' shared mechanisms: ``ops`` (wave / math primitives, the TP
group's peer buffers), ``sync`` (the tagged mailbox), ``mx`` (FP8 / MX
arithmetic), ``plan`` (execution model, layouts, build keys) and
``peer_memory`` (symmetric peer buffers)."""
