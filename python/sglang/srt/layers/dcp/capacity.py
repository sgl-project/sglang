# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Logical KV-capacity semantics for decode context parallelism.

Most DCP backends use the virtual paged allocator: every rank stores one
physical shard while the allocator exposes ``dcp_size`` times as many logical
token slots.  DeepSeek-V4 is currently different.  Its read-only mode keeps the
whole pool replicated, and its experimental physical mode shards unified-KV
rows without widening the SWA allocator or the DSV4 pool configurator.  Both
DSV4 modes therefore expose only the profiled logical capacity, not
``dcp_size`` times that capacity.

Keep this distinction centralized.  Communication parallelism and logical KV
capacity are separate properties and must not both be inferred from
``dcp_size``.
"""

from __future__ import annotations

from typing import Optional


def dcp_kv_capacity_multiplier(
    *, dcp_size: int, attention_backend: Optional[str]
) -> int:
    """Return the logical-capacity multiplier exposed by the allocator."""
    if dcp_size <= 1 or attention_backend == "dsv4":
        return 1
    return dcp_size


def aggregate_dcp_kv_capacity(
    per_rank_capacity: int,
    *,
    dcp_size: int,
    attention_backend: Optional[str],
) -> int:
    """Convert a profiled per-rank capacity to the allocator-visible total."""
    return per_rank_capacity * dcp_kv_capacity_multiplier(
        dcp_size=dcp_size,
        attention_backend=attention_backend,
    )
