from dataclasses import dataclass
from typing import Literal

import torch

# DeepSeek DSA FP8 KV rows use 512 E4M3 values, four float32 group scales,
# and 64 BF16 RoPE values: 512 + 4 * 4 + 64 * 2 = 656 bytes.  The BF16
# FlashMLA workspace expands the 512 + 64 values to 576 * 2 = 1152 bytes.
KV_BYTES_PER_ROW = 656
KV_DEQUANTIZED_BYTES_PER_ROW = 576 * 2
INDEX_BYTES = torch.tensor([], dtype=torch.int32).element_size()
DENSE_DEDUP_BLOCK_SIZE = 256
# A 1xH20 crossover sweep with a 65,536-row pool measured the dense-epoch
# path at 0.28x, 0.53x, 0.80x, and 1.07x of cached full dequant for 8K, 16K,
# 24K, and 32K prefixes.  The first material win was 1.31x at 40K.  The byte
# model below cannot represent the fixed six-kernel launch/scan floor, so keep
# the experimental path fail-closed below that measured boundary.
DENSE_EPOCH_MIN_PREFIX_ROWS = 40 * 1024


def _validate_extent(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _capacity_bucket(extent: int) -> int:
    return 0 if extent == 0 else 1 << (extent - 1).bit_length()


@dataclass(frozen=True)
class DenseEpochDedupBuffers:
    slot_epoch: torch.Tensor
    slot_to_compact: torch.Tensor
    selected_physical_slots: torch.Tensor
    remapped_topk: torch.Tensor
    block_offsets: torch.Tensor
    epoch: torch.Tensor
    num_unique: torch.Tensor


@dataclass(frozen=True)
class DenseEpochSelection:
    selected_physical_slots: torch.Tensor
    remapped_topk: torch.Tensor
    num_unique: torch.Tensor
    capacity: int


SelectiveKVMode = Literal["off", "no_dedup", "dense_epoch"]


@dataclass(frozen=True)
class SelectiveKVDequantResult:
    kv_cache: torch.Tensor
    remapped_topk: torch.Tensor
    mode: SelectiveKVMode


def resolve_selective_kv_mode(
    no_dedup_enabled: bool,
    dense_epoch_enabled: bool,
) -> SelectiveKVMode:
    """Resolve the two temporary experiment gates to one explicit mode."""
    if no_dedup_enabled and dense_epoch_enabled:
        raise ValueError(
            "SGLANG_EXPERIMENTAL_DSA_SELECTIVE_KV_NO_DEDUP and "
            "SGLANG_EXPERIMENTAL_DSA_SELECTIVE_KV_DENSE_EPOCH are mutually exclusive"
        )
    if dense_epoch_enabled:
        return "dense_epoch"
    if no_dedup_enabled:
        return "no_dedup"
    return "off"


class SelectiveKVWorkspace:
    """Grow-only buffers for selective DSA prefill preparation.

    Calls on one attention backend execute serially on the same CUDA stream.
    Reusing bucketed storage therefore gives stable addresses for repeated or
    smaller shapes without retaining per-layer outputs.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._bf16: torch.Tensor | None = None
        self._physical_slots: torch.Tensor | None = None
        self._remapped_topk: torch.Tensor | None = None
        self._slot_epoch: torch.Tensor | None = None
        self._slot_to_compact: torch.Tensor | None = None
        self._selected_physical_slots: torch.Tensor | None = None
        self._dedup_remapped_topk: torch.Tensor | None = None
        self._block_offsets: torch.Tensor | None = None
        self._epoch: torch.Tensor | None = None
        self._num_unique: torch.Tensor | None = None

    @property
    def bf16_capacity(self) -> int:
        return 0 if self._bf16 is None else self._bf16.shape[0]

    @property
    def occurrence_capacity(self) -> int:
        return 0 if self._physical_slots is None else self._physical_slots.shape[0]

    def get_bf16(self, num_rows: int) -> torch.Tensor:
        _validate_extent("num_rows", num_rows)
        if num_rows > self.bf16_capacity:
            self._bf16 = torch.empty(
                (_capacity_bucket(num_rows), 1, 576),
                dtype=torch.bfloat16,
                device=self.device,
            )
        if self._bf16 is None:
            return torch.empty((0, 1, 576), dtype=torch.bfloat16, device=self.device)
        return self._bf16[:num_rows]

    def get_occurrence_metadata(
        self, num_occurrences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_extent("num_occurrences", num_occurrences)
        if num_occurrences > self.occurrence_capacity:
            capacity = _capacity_bucket(num_occurrences)
            self._physical_slots = torch.empty(
                capacity, dtype=torch.int32, device=self.device
            )
            self._remapped_topk = torch.empty(
                capacity, dtype=torch.int32, device=self.device
            )
        if self._physical_slots is None:
            empty = torch.empty(0, dtype=torch.int32, device=self.device)
            return empty, empty.clone()
        return (
            self._physical_slots[:num_occurrences],
            self._remapped_topk[:num_occurrences],
        )

    def get_dense_dedup_buffers(
        self,
        num_pool_rows: int,
        selection_capacity: int,
        num_occurrences: int,
    ) -> DenseEpochDedupBuffers:
        for name, value in (
            ("num_pool_rows", num_pool_rows),
            ("selection_capacity", selection_capacity),
            ("num_occurrences", num_occurrences),
        ):
            _validate_extent(name, value)
        if selection_capacity > num_pool_rows:
            raise ValueError(
                "selection_capacity cannot exceed num_pool_rows: "
                f"{selection_capacity} > {num_pool_rows}"
            )

        current_pool_capacity = (
            0 if self._slot_epoch is None else self._slot_epoch.shape[0]
        )
        if num_pool_rows > current_pool_capacity:
            pool_capacity = _capacity_bucket(num_pool_rows)
            self._slot_epoch = torch.full(
                (pool_capacity,), -1, dtype=torch.int64, device=self.device
            )
            self._slot_to_compact = torch.empty(
                pool_capacity, dtype=torch.int32, device=self.device
            )

        num_scan_blocks = (
            num_pool_rows + DENSE_DEDUP_BLOCK_SIZE - 1
        ) // DENSE_DEDUP_BLOCK_SIZE
        current_block_capacity = (
            0 if self._block_offsets is None else self._block_offsets.shape[0]
        )
        if num_scan_blocks > current_block_capacity:
            self._block_offsets = torch.empty(
                _capacity_bucket(num_scan_blocks),
                dtype=torch.int32,
                device=self.device,
            )

        current_selection_capacity = (
            0
            if self._selected_physical_slots is None
            else self._selected_physical_slots.shape[0]
        )
        if selection_capacity > current_selection_capacity:
            selected_capacity = _capacity_bucket(selection_capacity)
            self._selected_physical_slots = torch.empty(
                selected_capacity, dtype=torch.int32, device=self.device
            )

        current_occurrence_capacity = (
            0
            if self._dedup_remapped_topk is None
            else self._dedup_remapped_topk.shape[0]
        )
        if num_occurrences > current_occurrence_capacity:
            occurrence_capacity = _capacity_bucket(num_occurrences)
            self._dedup_remapped_topk = torch.empty(
                occurrence_capacity, dtype=torch.int32, device=self.device
            )

        if self._epoch is None:
            self._epoch = torch.zeros(1, dtype=torch.int64, device=self.device)
            self._num_unique = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Zero-sized calls are not useful in production, but returning valid
        # empty tensors keeps the workspace contract total and easy to test.
        if self._slot_epoch is None:
            self._slot_epoch = torch.empty(0, dtype=torch.int64, device=self.device)
            self._slot_to_compact = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
        if self._selected_physical_slots is None:
            self._selected_physical_slots = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
        if self._dedup_remapped_topk is None:
            self._dedup_remapped_topk = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
        if self._block_offsets is None:
            self._block_offsets = torch.empty(0, dtype=torch.int32, device=self.device)

        return DenseEpochDedupBuffers(
            slot_epoch=self._slot_epoch,
            slot_to_compact=self._slot_to_compact,
            selected_physical_slots=self._selected_physical_slots,
            remapped_topk=self._dedup_remapped_topk,
            block_offsets=self._block_offsets,
            epoch=self._epoch,
            num_unique=self._num_unique,
        )


@dataclass(frozen=True)
class SelectiveKVRemap:
    """CPU oracle for the DSA logical -> physical -> compact mapping."""

    physical_slots: torch.Tensor
    remapped_topk: torch.Tensor
    num_valid: int
    num_unique: int


@dataclass(frozen=True)
class NoDedupSelectiveKVRemap:
    """Fixed-shape selective mapping used by the no-dedup GPU probe."""

    physical_slots: torch.Tensor
    remapped_topk: torch.Tensor


@dataclass(frozen=True)
class SelectiveKVTrafficEstimate:
    """Byte model used to gate the selective path conservatively."""

    full_kv_bytes: int
    selective_kv_bytes: int
    metadata_bytes: int
    full_workspace_bytes: int
    selective_workspace_bytes: int

    @property
    def selective_total_bytes(self) -> int:
        return self.selective_kv_bytes + self.metadata_bytes


def prepare_dense_epoch_selection(
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
    *,
    num_pool_rows: int,
    workspace: SelectiveKVWorkspace,
    deduplicate_fn=None,
) -> DenseEpochSelection:
    """Prepare fixed-capacity compact DSA indices without reading device state.

    ``deduplicate_fn`` is injectable only so the standalone kernel test can
    load this orchestration module without importing the complete SGLang
    package. Production callers use the real kernel by default.
    """
    if deduplicate_fn is None:
        from sglang.kernels.ops.attention.dsa.selective_kv_dequant import (
            deduplicate_kv_slots_dense_epoch,
        )

        deduplicate_fn = deduplicate_kv_slots_dense_epoch

    capacity = min(page_table_1_flattened.numel(), flat_topk_indices.numel())
    buffers = workspace.get_dense_dedup_buffers(
        num_pool_rows=num_pool_rows,
        selection_capacity=capacity,
        num_occurrences=flat_topk_indices.numel(),
    )
    selected_physical_slots = buffers.selected_physical_slots[:capacity]
    remapped_topk = buffers.remapped_topk[: flat_topk_indices.numel()].view_as(
        flat_topk_indices
    )
    selected_physical_slots, remapped_topk, num_unique = deduplicate_fn(
        page_table_1_flattened,
        flat_topk_indices,
        slot_epoch=buffers.slot_epoch,
        slot_to_compact=buffers.slot_to_compact,
        selected_physical_slots=selected_physical_slots,
        remapped_topk=remapped_topk,
        block_offsets=buffers.block_offsets,
        epoch=buffers.epoch,
        num_unique=buffers.num_unique,
        num_pool_rows=num_pool_rows,
        selection_capacity=capacity,
    )
    return DenseEpochSelection(
        selected_physical_slots=selected_physical_slots,
        remapped_topk=remapped_topk,
        num_unique=num_unique,
        capacity=capacity,
    )


def dequantize_dsa_prefix_kv_selective(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
    *,
    num_pool_rows: int,
    mode: SelectiveKVMode,
    workspace: SelectiveKVWorkspace,
    dequantize_fn=None,
    deduplicate_fn=None,
) -> SelectiveKVDequantResult:
    """Prepare BF16 prefix KV for FlashMLA with an opt-in selective path.

    The shape-only traffic gate runs before either experimental path.  It
    intentionally assumes every top-k occurrence is unique, so enabling an
    experiment cannot make a clearly dense workload pay the dedup overhead.
    ``dense_epoch`` keeps the data-dependent active row count on device;
    ``no_dedup`` dequantizes every occurrence because padding entries need not
    form a contiguous suffix.
    """
    if mode not in ("off", "no_dedup", "dense_epoch"):
        raise ValueError(f"unsupported selective KV mode: {mode!r}")
    if dequantize_fn is None:
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )

        dequantize_fn = dequantize_k_cache_paged

    def full_dequantize() -> SelectiveKVDequantResult:
        return SelectiveKVDequantResult(
            kv_cache=dequantize_fn(quant_k_cache, page_table_1_flattened),
            remapped_topk=flat_topk_indices,
            mode="off",
        )

    if mode == "off":
        return full_dequantize()
    if flat_topk_indices.ndim != 2:
        raise ValueError(
            "flat_topk_indices must have shape (query_tokens, topk), got "
            f"{tuple(flat_topk_indices.shape)}"
        )

    query_tokens, topk = flat_topk_indices.shape
    if mode == "dense_epoch":
        use_selective = should_use_dense_epoch_kv_dequant(
            prefix_rows=page_table_1_flattened.numel(),
            query_tokens=query_tokens,
            topk=topk,
            num_pool_rows=num_pool_rows,
        )
    else:
        use_selective = should_use_selective_kv_dequant(
            prefix_rows=page_table_1_flattened.numel(),
            query_tokens=query_tokens,
            topk=topk,
        )
    if not use_selective:
        return full_dequantize()

    if mode == "no_dedup":
        physical_slots, remapped_topk = workspace.get_occurrence_metadata(
            flat_topk_indices.numel()
        )
        remapped_topk = remapped_topk.view_as(flat_topk_indices)
        selection = build_selective_kv_no_dedup(
            page_table_1_flattened,
            flat_topk_indices,
            physical_slots_out=physical_slots,
            remapped_topk_out=remapped_topk,
        )
        out = workspace.get_bf16(selection.physical_slots.numel())
        return SelectiveKVDequantResult(
            kv_cache=dequantize_fn(
                quant_k_cache,
                selection.physical_slots,
                out=out,
            ),
            remapped_topk=selection.remapped_topk,
            mode="no_dedup",
        )

    selection = prepare_dense_epoch_selection(
        page_table_1_flattened,
        flat_topk_indices,
        num_pool_rows=num_pool_rows,
        workspace=workspace,
        deduplicate_fn=deduplicate_fn,
    )
    out = workspace.get_bf16(selection.capacity)
    return SelectiveKVDequantResult(
        kv_cache=dequantize_fn(
            quant_k_cache,
            selection.selected_physical_slots,
            out=out,
            num_valid_rows=selection.num_unique,
        ),
        remapped_topk=selection.remapped_topk,
        mode="dense_epoch",
    )


def _require_index_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"{name} must be an integer index tensor (int32 or int64), "
            f"got {tensor.dtype}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for the reference implementation")


def build_selective_kv_remap_reference(
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
) -> SelectiveKVRemap:
    """Build a stable physical-slot deduplication oracle on CPU.

    ``flat_topk_indices`` contains indices into ``page_table_1_flattened``
    after the DSA request-local offsets have been applied.  The returned
    indices address a compact list of unique *physical* KV-cache slots.  This
    distinction is important when different requests share radix-cache rows.

    The first appearance of a physical slot determines its compact index.  A
    production GPU kernel does not need to preserve this order, but using a
    deterministic oracle makes exact remap tests and benchmark debugging much
    easier.
    """
    _require_index_tensor("page_table_1_flattened", page_table_1_flattened)
    _require_index_tensor("flat_topk_indices", flat_topk_indices)
    if page_table_1_flattened.ndim != 1:
        raise ValueError(
            "page_table_1_flattened must be one-dimensional, got "
            f"shape={tuple(page_table_1_flattened.shape)}"
        )

    flat_indices = flat_topk_indices.reshape(-1)
    if flat_indices.numel() and int(flat_indices.min().item()) < -1:
        raise ValueError("flat_topk_indices contains an index smaller than -1")

    valid = flat_indices >= 0
    valid_indices = flat_indices[valid]
    if valid_indices.numel():
        max_index = int(valid_indices.max().item())
        if max_index >= page_table_1_flattened.numel():
            raise ValueError(
                "flat_topk_indices contains an index outside "
                f"page_table_1_flattened: {max_index} >= "
                f"{page_table_1_flattened.numel()}"
            )

    remapped_flat = torch.full(
        flat_indices.shape,
        -1,
        dtype=torch.int32,
        device=flat_topk_indices.device,
    )
    compact_by_physical_slot = {}
    selected_physical_slots = []

    for position, logical_index in enumerate(flat_indices.tolist()):
        if logical_index == -1:
            continue
        physical_slot = int(page_table_1_flattened[logical_index].item())
        if physical_slot < 0:
            raise ValueError(
                "page_table_1_flattened contains a negative physical KV slot "
                f"at logical index {logical_index}: {physical_slot}"
            )
        compact_index = compact_by_physical_slot.get(physical_slot)
        if compact_index is None:
            compact_index = len(selected_physical_slots)
            compact_by_physical_slot[physical_slot] = compact_index
            selected_physical_slots.append(physical_slot)
        remapped_flat[position] = compact_index

    physical_slots = torch.tensor(
        selected_physical_slots,
        dtype=page_table_1_flattened.dtype,
        device=page_table_1_flattened.device,
    )
    return SelectiveKVRemap(
        physical_slots=physical_slots,
        remapped_topk=remapped_flat.reshape(flat_topk_indices.shape),
        num_valid=int(valid.sum().item()),
        num_unique=len(selected_physical_slots),
    )


def build_selective_kv_no_dedup(
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
    *,
    physical_slots_out: torch.Tensor | None = None,
    remapped_topk_out: torch.Tensor | None = None,
) -> NoDedupSelectiveKVRemap:
    """Build a fixed-shape selective mapping without deduplicating rows.

    This is the first GPU performance probe, not the final production policy.
    It performs no data-dependent compaction: every top-k occurrence owns one
    output row, while ``-1`` entries retain their sentinel and gather physical
    row zero as an ignored landing value.  Consequently tensor shapes and
    addresses depend only on the input shapes and are suitable for measuring
    capture behavior without a host synchronization.
    """
    if page_table_1_flattened.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "page_table_1_flattened must be an integer index tensor "
            f"(int32 or int64), got {page_table_1_flattened.dtype}"
        )
    if flat_topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "flat_topk_indices must be an integer index tensor "
            f"(int32 or int64), got {flat_topk_indices.dtype}"
        )
    if page_table_1_flattened.device != flat_topk_indices.device:
        raise ValueError(
            "page_table_1_flattened and flat_topk_indices must be on the same device"
        )
    if page_table_1_flattened.ndim != 1:
        raise ValueError(
            "page_table_1_flattened must be one-dimensional, got "
            f"shape={tuple(page_table_1_flattened.shape)}"
        )
    if (physical_slots_out is None) != (remapped_topk_out is None):
        raise ValueError(
            "physical_slots_out and remapped_topk_out must be provided together"
        )

    flat_indices = flat_topk_indices.reshape(-1)
    valid = flat_indices >= 0
    if page_table_1_flattened.numel() == 0:
        # An empty prefix is not a production selective-dequant case.  Support
        # the all-padding CPU oracle edge without manufacturing an invalid
        # physical row; reject any real index explicitly.
        if flat_indices.device.type != "cpu" or bool(valid.any().item()):
            raise ValueError(
                "non-padding top-k indices require a non-empty "
                "page_table_1_flattened"
            )
        return NoDedupSelectiveKVRemap(
            physical_slots=page_table_1_flattened.new_empty((0,)),
            remapped_topk=torch.full_like(flat_topk_indices, -1, dtype=torch.int32),
        )

    if physical_slots_out is None:
        safe_logical_indices = flat_indices.clamp_min(0)
        physical_slots = page_table_1_flattened[safe_logical_indices.long()]
        occurrence_rows = torch.arange(
            flat_indices.numel(), device=flat_indices.device, dtype=torch.int32
        )
        remapped = torch.where(
            valid,
            occurrence_rows,
            torch.full_like(occurrence_rows, -1),
        )
    else:
        if page_table_1_flattened.dtype != torch.int32:
            raise TypeError(
                "preallocated no-dedup metadata requires an int32 page table"
            )
        if physical_slots_out.shape != flat_indices.shape:
            raise ValueError(
                "physical_slots_out shape must match flattened top-k: "
                f"{tuple(physical_slots_out.shape)} != {tuple(flat_indices.shape)}"
            )
        if remapped_topk_out.shape != flat_topk_indices.shape:
            raise ValueError(
                "remapped_topk_out shape must match top-k: "
                f"{tuple(remapped_topk_out.shape)} != "
                f"{tuple(flat_topk_indices.shape)}"
            )
        for name, tensor in (
            ("physical_slots_out", physical_slots_out),
            ("remapped_topk_out", remapped_topk_out),
        ):
            if tensor.device != flat_topk_indices.device:
                raise ValueError(f"{name} must be on the top-k device")
            if tensor.dtype != torch.int32 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous int32")

        physical_slots = physical_slots_out
        remapped = remapped_topk_out.reshape(-1)
        # Reuse the remap destination as the safe logical-index staging
        # buffer, then overwrite it with compact occurrence rows.
        torch.clamp(flat_indices, min=0, out=remapped)
        torch.index_select(
            page_table_1_flattened,
            0,
            remapped,
            out=physical_slots,
        )
        torch.arange(
            flat_indices.numel(),
            device=flat_indices.device,
            dtype=torch.int32,
            out=remapped,
        )
        remapped.masked_fill_(~valid, -1)
    return NoDedupSelectiveKVRemap(
        physical_slots=physical_slots,
        remapped_topk=remapped.reshape(flat_topk_indices.shape),
    )


def estimate_selective_kv_traffic(
    *,
    prefix_rows: int,
    valid_topk_entries: int,
    unique_rows: int,
) -> SelectiveKVTrafficEstimate:
    """Estimate compulsory byte traffic before kernel-specific profiling.

    Metadata uses a transparent lower-order approximation per valid entry:
    read the logical top-k index, read its physical page-table entry, and write
    the compact index.  Each unique row also records one physical key and one
    compact value.  H20 measurements will replace or calibrate this model; it
    intentionally does not pretend to predict cache-line or atomic traffic.
    """
    for name, value in (
        ("prefix_rows", prefix_rows),
        ("valid_topk_entries", valid_topk_entries),
        ("unique_rows", unique_rows),
    ):
        _validate_extent(name, value)
    if unique_rows > valid_topk_entries:
        raise ValueError(
            "unique_rows cannot exceed valid_topk_entries: "
            f"{unique_rows} > {valid_topk_entries}"
        )
    if unique_rows > prefix_rows:
        raise ValueError(
            f"unique_rows cannot exceed prefix_rows: {unique_rows} > {prefix_rows}"
        )

    row_traffic = KV_BYTES_PER_ROW + KV_DEQUANTIZED_BYTES_PER_ROW
    metadata_bytes = valid_topk_entries * (3 * INDEX_BYTES) + unique_rows * (
        2 * INDEX_BYTES
    )
    return SelectiveKVTrafficEstimate(
        full_kv_bytes=prefix_rows * row_traffic,
        selective_kv_bytes=unique_rows * row_traffic,
        metadata_bytes=metadata_bytes,
        full_workspace_bytes=prefix_rows * KV_DEQUANTIZED_BYTES_PER_ROW,
        selective_workspace_bytes=unique_rows * KV_DEQUANTIZED_BYTES_PER_ROW,
    )


def should_use_selective_kv_dequant(
    *,
    prefix_rows: int,
    query_tokens: int,
    topk: int,
    safety_factor: float = 1.25,
) -> bool:
    """Return a conservative host-side pre-dedup decision.

    Without measuring overlap, ``query_tokens * topk`` is the maximum number
    of valid occurrences and ``min(prefix_rows, occurrences)`` is the maximum
    unique-row count.  The path is selected only if it still wins under that
    pessimistic bound.  A later device-side policy may use an observed unique
    ratio, but must retain this function as the fail-closed fallback.
    """
    if prefix_rows <= 0 or query_tokens <= 0 or topk <= 0:
        return False
    if safety_factor < 1.0:
        raise ValueError(f"safety_factor must be >= 1.0, got {safety_factor}")

    valid_upper_bound = query_tokens * topk
    unique_upper_bound = min(prefix_rows, valid_upper_bound)
    estimate = estimate_selective_kv_traffic(
        prefix_rows=prefix_rows,
        valid_topk_entries=valid_upper_bound,
        unique_rows=unique_upper_bound,
    )
    return estimate.selective_total_bytes * safety_factor < estimate.full_kv_bytes


def should_use_dense_epoch_kv_dequant(
    *,
    prefix_rows: int,
    query_tokens: int,
    topk: int,
    num_pool_rows: int,
    safety_factor: float = 1.25,
) -> bool:
    """Fail-closed byte gate including the persistent dense-table scan.

    The local-scan and finalize phases each read the int64 epoch table once.
    This upper-bound model also charges block scan/readback traffic and compact
    mapping publication. It deliberately double-counts some per-occurrence
    metadata already present in ``estimate_selective_kv_traffic``: the gate is
    a conservative pre-profile safeguard, not a throughput predictor.
    """
    if not isinstance(num_pool_rows, int) or isinstance(num_pool_rows, bool):
        raise ValueError(
            f"num_pool_rows must be a positive integer, got {num_pool_rows!r}"
        )
    if num_pool_rows <= 0:
        raise ValueError(
            f"num_pool_rows must be a positive integer, got {num_pool_rows!r}"
        )
    if prefix_rows <= 0 or query_tokens <= 0 or topk <= 0:
        return False
    if prefix_rows < DENSE_EPOCH_MIN_PREFIX_ROWS:
        return False
    if safety_factor < 1.0:
        raise ValueError(f"safety_factor must be >= 1.0, got {safety_factor}")

    valid_upper_bound = query_tokens * topk
    unique_upper_bound = min(prefix_rows, valid_upper_bound, num_pool_rows)
    estimate = estimate_selective_kv_traffic(
        prefix_rows=prefix_rows,
        valid_topk_entries=valid_upper_bound,
        unique_rows=unique_upper_bound,
    )
    num_scan_blocks = (
        num_pool_rows + DENSE_DEDUP_BLOCK_SIZE - 1
    ) // DENSE_DEDUP_BLOCK_SIZE
    dense_scan_bytes = (
        2 * num_pool_rows * 8 + num_scan_blocks * 12 + unique_upper_bound * 12
    )
    selective_bytes = estimate.selective_total_bytes + dense_scan_bytes
    return selective_bytes * safety_factor < estimate.full_kv_bytes


def maybe_build_selective_kv_no_dedup(
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
    *,
    enabled: bool,
    safety_factor: float = 1.25,
) -> NoDedupSelectiveKVRemap | None:
    """Plan the opt-in no-dedup probe without inspecting device data."""
    if not enabled:
        return None
    if flat_topk_indices.ndim != 2:
        raise ValueError(
            "flat_topk_indices must have shape (query_tokens, topk), got "
            f"{tuple(flat_topk_indices.shape)}"
        )
    query_tokens, topk = flat_topk_indices.shape
    if not should_use_selective_kv_dequant(
        prefix_rows=page_table_1_flattened.numel(),
        query_tokens=query_tokens,
        topk=topk,
        safety_factor=safety_factor,
    ):
        return None
    return build_selective_kv_no_dedup(
        page_table_1_flattened,
        flat_topk_indices,
    )
