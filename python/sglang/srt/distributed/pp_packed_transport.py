"""Packed pipeline-parallel tensor transport.

Reduces per-tensor send/recv overhead by packing multiple proxy tensors
into two contiguous buffers (data + control) and sending them as single
tensors.  Behind the SGLANG_PP_PACKED_TRANSPORT feature flag.

Protocol v1:
  data_buffer:   contiguous BF16/FP16 tensor for hidden_states, residual, aux
  control_buffer: contiguous int32/int64 tensor for topk_indices, token IDs, etc.
  header:         small CPU tensor with schema_id, active_rows, presence bitmask

The header is sent via send_object (CPU/Gloo), same as the existing metadata
path.  The data and control buffers are sent via torch.distributed.send/irecv
on the device group.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# Presence bitmask bits
BIT_HIDDEN = 1 << 0
BIT_RESIDUAL = 1 << 1
BIT_AUX = 1 << 2
BIT_TOPK = 1 << 3
BIT_NEXT_TOKEN_IDS = 1 << 4
BIT_ACCEPT_LENS = 1 << 5
BIT_NEW_SEQ_LENS = 1 << 6
BIT_BONUS_TOKENS = 1 << 7

# Maximum number of schema cache entries before eviction
SCHEMA_CACHE_MAX = 64


@dataclass
class PPSchemaKey:
    """Cache key for a packed-transport schema."""
    presence_mask: int
    hidden_size: int
    capture_layers: int
    topk_size: int
    dtype_id: int
    max_rows_bucket: int

    def to_tuple(self) -> Tuple:
        return (self.presence_mask, self.hidden_size, self.capture_layers,
                self.topk_size, self.dtype_id, self.max_rows_bucket)


@dataclass
class PPSchemaEntry:
    """A resolved schema entry with buffer layout."""
    schema_id: int
    key: PPSchemaKey
    data_num_elements: int
    control_num_elements: int
    data_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    control_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)


class PPSchemaCache:
    """Bounded LRU schema cache.

    On the first occurrence of a schema, the full layout is sent.
    On subsequent occurrences, only the schema_id and dynamic fields are sent.
    """
    def __init__(self, max_entries: int = SCHEMA_CACHE_MAX):
        self._max = max_entries
        self._cache: Dict[Tuple, PPSchemaEntry] = {}
        self._next_id = 0
        # Counters
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.full_metadata_bytes = 0
        self.cached_metadata_bytes = 0

    def lookup(self, key: PPSchemaKey) -> Optional[PPSchemaEntry]:
        tup = key.to_tuple()
        entry = self._cache.get(tup)
        if entry is not None:
            self.hits += 1
            self.cached_metadata_bytes += 16  # schema_id + active_rows + bitmask + pad
        else:
            self.misses += 1
        return entry

    def register(self, key: PPSchemaKey,
                 data_num_elements: int, control_num_elements: int,
                 data_offsets: Dict, control_offsets: Dict) -> PPSchemaEntry:
        tup = key.to_tuple()
        if len(self._cache) >= self._max:
            # Evict oldest (FIFO — simple and deterministic)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.evictions += 1
        entry = PPSchemaEntry(
            schema_id=self._next_id,
            key=key,
            data_num_elements=data_num_elements,
            control_num_elements=control_num_elements,
            data_offsets=dict(data_offsets),
            control_offsets=dict(control_offsets),
        )
        self._next_id += 1
        self._cache[tup] = entry
        self.full_metadata_bytes += 256  # approximate full metadata size
        return entry

    def get_by_id(self, schema_id: int) -> Optional[PPSchemaEntry]:
        for entry in self._cache.values():
            if entry.schema_id == schema_id:
                return entry
        return None

    @property
    def size(self) -> int:
        return len(self._cache)


# Global schema caches (per-process, shared across PP stages)
_send_schema_cache = PPSchemaCache()
_recv_schema_cache = PPSchemaCache()


def _dtype_to_id(dtype: torch.dtype) -> int:
    """Map torch dtype to a stable integer ID."""
    _map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.float8_e4m3fn: 5,
    }
    return _map.get(dtype, -1)


def _id_to_dtype(dtype_id: int) -> torch.dtype:
    _map = {
        0: torch.float32,
        1: torch.float16,
        2: torch.bfloat16,
        3: torch.int32,
        4: torch.int64,
        5: torch.float8_e4m3fn,
    }
    return _map.get(dtype_id, torch.bfloat16)


def calculate_pp_buffer_layout(
    tensor_dict: Dict[str, torch.Tensor],
    hidden_size: int,
    capture_layers: int,
    topk_size: int,
    max_rows_bucket: int,
) -> Tuple[PPSchemaKey, int, int, Dict, Dict]:
    """Calculate buffer layout for packed transport.

    Returns:
        schema_key, data_num_elements, control_num_elements,
        data_offsets, control_offsets
    """
    presence = 0
    data_offsets: Dict[str, Tuple[int, int]] = {}
    control_offsets: Dict[str, Tuple[int, int]] = {}
    data_offset = 0
    control_offset = 0

    # Data buffer: float-type tensors (hidden, residual, aux)
    if "hidden_states" in tensor_dict:
        presence |= BIT_HIDDEN
        numel = max_rows_bucket * hidden_size
        data_offsets["hidden_states"] = (data_offset, data_offset + numel)
        data_offset += numel

    if "residual" in tensor_dict:
        presence |= BIT_RESIDUAL
        numel = max_rows_bucket * hidden_size
        data_offsets["residual"] = (data_offset, data_offset + numel)
        data_offset += numel

    aux_key = "glm52_eagle3_aux_hidden_states"
    if aux_key in tensor_dict:
        presence |= BIT_AUX
        numel = max_rows_bucket * capture_layers * hidden_size
        data_offsets[aux_key] = (data_offset, data_offset + numel)
        data_offset += numel

    # Control buffer: integer-type tensors (topk, token IDs, etc.)
    if "topk_indices" in tensor_dict:
        presence |= BIT_TOPK
        numel = max_rows_bucket * topk_size
        control_offsets["topk_indices"] = (control_offset, control_offset + numel)
        control_offset += numel

    if "next_token_ids" in tensor_dict:
        presence |= BIT_NEXT_TOKEN_IDS
        numel = max_rows_bucket
        control_offsets["next_token_ids"] = (control_offset, control_offset + numel)
        control_offset += numel

    if "accept_lens" in tensor_dict:
        presence |= BIT_ACCEPT_LENS
        numel = max_rows_bucket
        control_offsets["accept_lens"] = (control_offset, control_offset + numel)
        control_offset += numel

    if "new_seq_lens" in tensor_dict:
        presence |= BIT_NEW_SEQ_LENS
        numel = max_rows_bucket
        control_offsets["new_seq_lens"] = (control_offset, control_offset + numel)
        control_offset += numel

    if "bonus_tokens" in tensor_dict:
        presence |= BIT_BONUS_TOKENS
        numel = max_rows_bucket
        control_offsets["bonus_tokens"] = (control_offset, control_offset + numel)
        control_offset += numel

    # Determine dtype from hidden_states
    data_dtype = tensor_dict.get("hidden_states", torch.empty(0, dtype=torch.bfloat16)).dtype

    schema_key = PPSchemaKey(
        presence_mask=presence,
        hidden_size=hidden_size,
        capture_layers=capture_layers,
        topk_size=topk_size,
        dtype_id=_dtype_to_id(data_dtype),
        max_rows_bucket=max_rows_bucket,
    )

    return schema_key, data_offset, control_offset, data_offsets, control_offsets


def pack_pp_proxy_tensors(
    tensor_dict: Dict[str, torch.Tensor],
    data_buffer: torch.Tensor,
    control_buffer: torch.Tensor,
    data_offsets: Dict[str, Tuple[int, int]],
    control_offsets: Dict[str, Tuple[int, int]],
    active_rows: int,
) -> None:
    """Pack tensors into preallocated contiguous buffers.

    Copies tensor data into the appropriate offsets. Only the first
    ``active_rows`` rows are copied; the rest of the buffer is untouched.
    """
    for key, (start, end) in data_offsets.items():
        if key not in tensor_dict:
            continue
        src = tensor_dict[key]
        dst_view = data_buffer[start:end].view(-1)
        src_flat = src[:active_rows].reshape(-1).to(dst_view.dtype)
        n = min(src_flat.numel(), dst_view.numel())
        dst_view[:n].copy_(src_flat[:n])

    for key, (start, end) in control_offsets.items():
        if key not in tensor_dict:
            continue
        src = tensor_dict[key]
        dst_view = control_buffer[start:end].view(-1)
        src_flat = src[:active_rows].reshape(-1).to(dst_view.dtype)
        n = min(src_flat.numel(), dst_view.numel())
        dst_view[:n].copy_(src_flat[:n])


def unpack_pp_proxy_tensors(
    data_buffer: torch.Tensor,
    control_buffer: torch.Tensor,
    schema_entry: PPSchemaEntry,
    active_rows: int,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    capture_layers: int,
    topk_size: int,
) -> Dict[str, torch.Tensor]:
    """Unpack tensors from contiguous buffers.

    Returns a dict matching the original tensor_dict structure.
    Only ``active_rows`` rows are extracted.
    """
    result: Dict[str, torch.Tensor] = {}
    data_dtype = _id_to_dtype(schema_entry.key.dtype_id)
    bucket = schema_entry.key.max_rows_bucket

    for key, (start, end) in schema_entry.data_offsets.items():
        total_elems = end - start
        # Calculate per-row element count for this tensor
        if key == "hidden_states":
            per_row = hidden_size
            view = data_buffer[start:start + active_rows * per_row].view(active_rows, per_row)
            result[key] = view.contiguous().to(data_dtype)
        elif key == "residual":
            per_row = hidden_size
            view = data_buffer[start:start + active_rows * per_row].view(active_rows, per_row)
            result[key] = view.contiguous().to(data_dtype)
        elif key == "glm52_eagle3_aux_hidden_states":
            per_row = capture_layers * hidden_size
            view = data_buffer[start:start + active_rows * per_row].view(active_rows, capture_layers, hidden_size)
            result[key] = view.contiguous().to(data_dtype)

    for key, (start, end) in schema_entry.control_offsets.items():
        if key == "topk_indices":
            per_row = topk_size
            view = control_buffer[start:start + active_rows * per_row].view(active_rows, per_row)
            result[key] = view.contiguous().to(torch.int32)
        elif key in ("next_token_ids", "bonus_tokens"):
            view = control_buffer[start:start + active_rows]
            result[key] = view.contiguous().to(torch.int64)
        elif key in ("accept_lens", "new_seq_lens"):
            view = control_buffer[start:start + active_rows]
            result[key] = view.contiguous().to(torch.int32)

    return result


def validate_pp_packed_header(
    schema_id: int,
    active_rows: int,
    presence_mask: int,
    expected_max_rows: int,
    recv_schema_cache: PPSchemaCache,
) -> PPSchemaEntry:
    """Validate a received packed-transport header.

    Returns the schema entry if valid. Raises on any mismatch.
    """
    entry = recv_schema_cache.get_by_id(schema_id)
    if entry is None:
        raise RuntimeError(
            f"PP packed transport: unknown schema_id={schema_id}. "
            f"Recv cache size={recv_schema_cache.size}. "
            f"This may indicate a schema-cache eviction or version mismatch."
        )

    if active_rows < 0:
        raise RuntimeError(
            f"PP packed transport: negative active_rows={active_rows}"
        )

    if active_rows > expected_max_rows:
        raise RuntimeError(
            f"PP packed transport: active_rows={active_rows} exceeds "
            f"expected_max_rows={expected_max_rows}"
        )

    if presence_mask != entry.key.presence_mask:
        raise RuntimeError(
            f"PP packed transport: presence bitmask mismatch. "
            f"Header={presence_mask:#x}, schema={entry.key.presence_mask:#x}"
        )

    return entry


def validate_pp_buffer_capacity(
    data_buffer: torch.Tensor,
    control_buffer: torch.Tensor,
    schema_entry: PPSchemaEntry,
) -> None:
    """Validate that buffers have sufficient capacity for the schema."""
    if data_buffer.numel() < schema_entry.data_num_elements:
        raise RuntimeError(
            f"PP packed transport: data buffer capacity={data_buffer.numel()} "
            f"< required={schema_entry.data_num_elements}"
        )
    if control_buffer.numel() < schema_entry.control_num_elements:
        raise RuntimeError(
            f"PP packed transport: control buffer capacity={control_buffer.numel()} "
            f"< required={schema_entry.control_num_elements}"
        )


class PPStaticBufferRegistry:
    """Registry for runner-owned static send/receive buffers.

    Buffers are allocated once and reused across iterations.
    Bucket-based: the buffer is sized for the maximum expected token rows
    in a bucket, avoiding per-iteration allocation.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self._data_buffers: Dict[int, torch.Tensor] = {}  # bucket -> buffer
        self._control_buffers: Dict[int, torch.Tensor] = {}
        self._max_bucket = 0
        self.allocation_count = 0

    def get_or_allocate(
        self,
        bucket: int,
        data_num_elements: int,
        control_num_elements: int,
        data_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get existing buffer or allocate new one for the given bucket."""
        if bucket not in self._data_buffers:
            self._data_buffers[bucket] = torch.zeros(
                max(data_num_elements, 1), dtype=data_dtype, device=self.device
            )
            self._control_buffers[bucket] = torch.zeros(
                max(control_num_elements, 1), dtype=torch.int32, device=self.device
            )
            self._max_bucket = max(self._max_bucket, bucket)
            self.allocation_count += 1
            logger.debug(
                "PP static buffer: allocated bucket=%d, data=%d, control=%d",
                bucket, data_num_elements, control_num_elements,
            )
        return self._data_buffers[bucket], self._control_buffers[bucket]

    @property
    def size(self) -> int:
        return len(self._data_buffers)

    def reset(self):
        """Free all buffers."""
        self._data_buffers.clear()
        self._control_buffers.clear()
        self._max_bucket = 0


def get_send_schema_cache() -> PPSchemaCache:
    return _send_schema_cache


def get_recv_schema_cache() -> PPSchemaCache:
    return _recv_schema_cache
