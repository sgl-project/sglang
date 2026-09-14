# SPDX-License-Identifier: Apache-2.0
"""Per-pool HiCache host-tier occupancy, torch-free.

The anchor gauges ``sglang:hicache_host_used_tokens`` / ``_total_tokens`` read
one host pool. A hybrid model's host tier is a ``HostPoolGroup`` of several
pools (kv, mamba, swa, indexer, ...), each with its own capacity and eviction
pressure; these helpers walk the group so the scheduler can publish
``sglang:hicache_host_pool_used_tokens{pool}`` and friends next to them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# Component identities (str(ComponentType) is the lower-cased name) mapped to
# the host pool they free into; unmapped components keep their own name.
DEFAULT_HOST_POOL_LABELS: Mapping[str, str] = {
    "full": "kv",
    "swa": "swa",
    "mamba": "mamba",
}


def pool_label(name: Any) -> str:
    """Label value for a pool name: the ``PoolName`` value or the plain string."""
    return str(name.value) if isinstance(name, Enum) else str(name)


def collect_host_pool_stats(
    host_pool_group: Any,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return ``({pool: used_tokens}, {pool: total_tokens})`` for a host pool group.

    ``total`` is the pool's ``logical_size``; ``used`` is that minus what
    ``available_size()`` reports, floored at zero. For the mamba pool a token is one checkpoint
    slot. ``None`` or a group without entries yields two empty dicts.
    """
    used: Dict[str, int] = {}
    total: Dict[str, int] = {}
    if host_pool_group is None:
        return used, total
    for entry in host_pool_group.entries:
        host_pool = entry.host_pool
        label = pool_label(entry.name)
        capacity = int(host_pool.logical_size)
        available = int(host_pool.available_size())
        used[label] = max(capacity - available, 0)
        total[label] = capacity
    return used, total


def host_pool_eviction_counts(
    host_frees: Mapping[Any, Iterable[Any]],
    labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, int]:
    """Sum the freed host slots per pool label from a tree step's ``host_frees``.

    ``host_frees`` maps a component to the index tensors (or any sized
    sequences) it released; each entry's length is its slot count.
    """
    labels = DEFAULT_HOST_POOL_LABELS if labels is None else labels
    counts: Dict[str, int] = {}
    for component, values in host_frees.items():
        key = str(component)
        label = labels.get(key, key)
        freed = sum(len(value) for value in values)
        if freed <= 0:
            continue
        counts[label] = counts.get(label, 0) + int(freed)
    return counts
