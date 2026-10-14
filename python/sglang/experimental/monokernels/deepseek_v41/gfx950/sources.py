# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

from sglang.experimental.monokernels.common.hip.plan import source_digest

SOURCES = source_digest("common/hip", "deepseek_v41/gfx950")
