# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel sources in every build's cache key: an edit in this package rebuilds."""

from sglang.experimental.monokernels.hip.common.plan import source_digest

SOURCES = source_digest(".")
