# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

from sglang.kernels.ops.mono.common_hip.plan import source_digest

SOURCES = source_digest("common_hip", "deepseek_v41_hip")
