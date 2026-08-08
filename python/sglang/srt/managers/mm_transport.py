from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    unwrap_from_pickle,
)
from sglang.srt.utils import broadcast_pyobj

# Keep small model-specific tensors in the metadata stream so one request does
# not launch many tiny collectives. Image/audio payloads normally exceed this
# conservative threshold by orders of magnitude.
_DIRECT_BROADCAST_MIN_BYTES = 1 << 20
# A tensor collective has a fixed rendezvous cost. Require enough aggregate
# payload in one scheduler broadcast to amortize it; otherwise retain the
# original pickle path so low-rate single-request traffic does not regress.
_DIRECT_BROADCAST_MIN_TOTAL_BYTES = 8 << 20


@dataclass(frozen=True)
class _CpuTensorPlaceholder:
    shape: tuple[int, ...]
    dtype: torch.dtype


def _iter_tokenized_reqs(data: Optional[List[Any]]) -> Iterator[Any]:
    for req in data or []:
        if isinstance(
            req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
        ):
            yield from _iter_tokenized_reqs(req.batch)
        elif isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
            yield req


def _unwrap_mm_inputs(data: Optional[List[Any]]) -> None:
    for req in _iter_tokenized_reqs(data):
        req.mm_inputs = unwrap_from_pickle(req.mm_inputs)


def _replace_large_cpu_tensors(value: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        if (
            value.device.type == "cpu"
            and value.layout == torch.strided
            and value.numel() * value.element_size() >= _DIRECT_BROADCAST_MIN_BYTES
        ):
            tensors.append(value)
            return _CpuTensorPlaceholder(tuple(value.shape), value.dtype)
        return value
    if isinstance(value, list):
        return [_replace_large_cpu_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_large_cpu_tensors(item, tensors) for item in value)
    if isinstance(value, dict):
        return {
            key: _replace_large_cpu_tensors(item, tensors)
            for key, item in value.items()
        }
    return value


def _collect_large_cpu_tensors(value: Any, tensors: List[torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        if (
            value.device.type == "cpu"
            and value.layout == torch.strided
            and value.numel() * value.element_size() >= _DIRECT_BROADCAST_MIN_BYTES
        ):
            tensors.append(value)
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            _collect_large_cpu_tensors(item, tensors)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_large_cpu_tensors(item, tensors)


def _allocate_cpu_tensors(value: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(value, _CpuTensorPlaceholder):
        tensor = torch.empty(value.shape, dtype=value.dtype, device="cpu")
        tensors.append(tensor)
        return tensor
    if isinstance(value, list):
        return [_allocate_cpu_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_allocate_cpu_tensors(item, tensors) for item in value)
    if isinstance(value, dict):
        return {
            key: _allocate_cpu_tensors(item, tensors) for key, item in value.items()
        }
    return value


def _detach_mm_tensors(data: Optional[List[Any]]) -> tuple[List[torch.Tensor], List]:
    candidates: List[torch.Tensor] = []
    for req in _iter_tokenized_reqs(data):
        if req.mm_inputs is None:
            continue
        for item in req.mm_inputs.mm_items:
            for field in ("feature", "precomputed_embeddings"):
                _collect_large_cpu_tensors(getattr(item, field), candidates)

    if sum(t.numel() * t.element_size() for t in candidates) < (
        _DIRECT_BROADCAST_MIN_TOTAL_BYTES
    ):
        return [], []

    tensors: List[torch.Tensor] = []
    originals = []
    for req in _iter_tokenized_reqs(data):
        if req.mm_inputs is None:
            continue
        for item in req.mm_inputs.mm_items:
            for field in ("feature", "precomputed_embeddings"):
                value = getattr(item, field)
                replaced = _replace_large_cpu_tensors(value, tensors)
                if replaced is not value:
                    originals.append((item, field, value))
                    setattr(item, field, replaced)
    return tensors, originals


def _restore_mm_tensors(originals: List) -> None:
    for item, field, value in originals:
        setattr(item, field, value)


def _allocate_mm_tensors(data: Optional[List[Any]]) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for req in _iter_tokenized_reqs(data):
        if req.mm_inputs is None:
            continue
        for item in req.mm_inputs.mm_items:
            item.feature = _allocate_cpu_tensors(item.feature, tensors)
            item.precomputed_embeddings = _allocate_cpu_tensors(
                item.precomputed_embeddings, tensors
            )
    return tensors


def broadcast_mm_cpu_tensors(
    data: Optional[List[Any]],
    rank: int,
    dist_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
) -> List[Any]:
    """Broadcast work requests without putting large MM tensors in pickle.

    Request metadata and small values still use ``broadcast_pyobj``. Large CPU
    feature tensors use the Gloo tensor collective directly, avoiding a second
    tensor -> pickle bytes -> tensor round trip on multi-node serving.
    """
    is_src = rank == src
    source_tensors: List[torch.Tensor] = []
    originals = []
    if is_src:
        _unwrap_mm_inputs(data)
        source_tensors, originals = _detach_mm_tensors(data)

    try:
        result = broadcast_pyobj(data, rank, dist_group, src=src)
    finally:
        if is_src:
            _restore_mm_tensors(originals)

    tensors = source_tensors if is_src else _allocate_mm_tensors(result)
    pending: List[tuple[Any, torch.Tensor]] = []
    try:
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            payload = tensor if tensor.is_contiguous() else tensor.contiguous()
            handle = dist.broadcast(payload, src=src, group=dist_group, async_op=True)
            # Keep the payload alive until its asynchronous collective finishes.
            pending.append((handle, payload))
    finally:
        for handle, _payload in pending:
            handle.wait()
    return result
