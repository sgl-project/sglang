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
"""Bounded diagnostics for DSA buffers recorded by full CUDA graphs.

CUDA graphs retain device addresses, not Python tensor owners.  The probe mode
records addresses without changing lifetimes; retain mode additionally keeps
the captured tensors alive so the two behaviours can be compared using one
immutable image.

Workflow: set SGLANG_DSA_GRAPH_BUFFER_LIFETIME=probe for a suspect boot.  When
a CUDA-graph replay faults on a buffer that has since been freed or rewritten,
call describe_graph_buffer_overlap(tensor) at the fault site — it reports
whether the pointer lands inside (logical-overlap / storage-overlap) or nearest
to a buffer captured during a DSA graph.  If flipping to retain makes the fault
disappear, the fault is a lifetime bug: something released storage the captured
graph still references.  off (the default) records nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ENV = "SGLANG_DSA_GRAPH_BUFFER_LIFETIME"
_VALID_MODES = frozenset(("off", "probe", "retain"))
_MAX_RECORDS = 4096


@dataclass(frozen=True)
class GraphBufferRecord:
    label: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    data_start: int
    data_end: int
    storage_start: int
    storage_end: int


_records: list[GraphBufferRecord] = []
_owners: list[torch.Tensor] = []


def _mode() -> str:
    mode = (envs.SGLANG_DSA_GRAPH_BUFFER_LIFETIME.get() or "off").lower()
    if mode not in _VALID_MODES:
        raise RuntimeError(
            f"{_ENV} must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    return mode


def _logical_end(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return tensor.data_ptr()
    max_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return tensor.data_ptr() + (max_offset + 1) * tensor.element_size()


def _record(tensor: torch.Tensor, label: str) -> GraphBufferRecord:
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    return GraphBufferRecord(
        label=label,
        device=str(tensor.device),
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        data_start=tensor.data_ptr(),
        data_end=_logical_end(tensor),
        storage_start=storage_start,
        storage_end=storage_start + storage.nbytes(),
    )


_cap_warned = False


def _warn_record_cap_once() -> None:
    global _cap_warned
    if _cap_warned:
        return
    _cap_warned = True
    logger.warning(
        "%s hit its %d-record cap mid-capture; later buffers are not recorded "
        "(diagnostics only — capture continues)",
        _ENV,
        _MAX_RECORDS,
    )


def register_graph_buffer(tensor: torch.Tensor, label: str) -> None:
    """Record a DSA intermediate only during the real CUDA graph capture."""
    mode = _mode()
    if mode == "off" or not tensor.is_cuda:
        return
    if not torch.cuda.is_current_stream_capturing():
        return
    if len(_records) >= _MAX_RECORDS:
        # Diagnostics must never kill a capture: stop recording and warn once
        # instead of raising mid-capture.
        _warn_record_cap_once()
        return
    _records.append(_record(tensor, label))
    if mode == "retain":
        _owners.append(tensor)


def _range_gap(start: int, end: int, other_start: int, other_end: int) -> int:
    if max(start, other_start) < min(end, other_end):
        return -1
    if end <= other_start:
        return other_start - end
    return start - other_end


def describe_graph_buffer_overlap(tensor: torch.Tensor) -> str:
    """Describe exact overlaps and nearest captured DSA buffer ranges."""
    if not _records:
        return "none-recorded"
    target = _record(tensor, "target")
    ranked: list[tuple[int, int, GraphBufferRecord]] = []
    for record in _records:
        logical_gap = _range_gap(
            target.data_start, target.data_end, record.data_start, record.data_end
        )
        storage_gap = _range_gap(
            target.data_start,
            target.data_end,
            record.storage_start,
            record.storage_end,
        )
        relation = 0 if logical_gap < 0 else 1 if storage_gap < 0 else 2
        gap = logical_gap if relation == 0 else storage_gap
        ranked.append((relation, gap, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    parts = []
    for relation, gap, record in ranked[:8]:
        relation_name = ("logical-overlap", "storage-overlap", "nearest")[relation]
        parts.append(
            f"{relation_name}:{record.label}:gap={gap}:"
            f"data=[{record.data_start},{record.data_end}):"
            f"storage=[{record.storage_start},{record.storage_end}):"
            f"dtype={record.dtype}:shape={record.shape}:stride={record.stride}"
        )
    return " | ".join(parts)


def _reset_for_test() -> None:
    global _cap_warned
    _cap_warned = False
    _records.clear()
    _owners.clear()


def _state_for_test() -> tuple[list[GraphBufferRecord], list[Any]]:
    return list(_records), list(_owners)
