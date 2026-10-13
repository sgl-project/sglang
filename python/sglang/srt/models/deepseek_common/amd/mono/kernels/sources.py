# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm-project/vllm @ 193922d6ff: vllm/models/deepseek_v41/amd/mono/sources.py
"""The mono layer's kernel sources, in every build's JIT cache key
(``common.plan.key_tuple``): an edit anywhere in this package rebuilds."""

from sglang.srt.models.deepseek_common.amd.mono.kernels.common.plan import source_digest

SOURCES = source_digest(".")
