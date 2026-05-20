"""KV cache canary verify kernel — host wrapper, dataclasses, enums.

This module defines the public API surface for the verify kernel:

- module-level constants (chain anchor, slot bytes, violation row width),
- IntEnums (CanaryLaunchTag, RealKvHashMode),
- the VerifyPlan dataclass consumed by canary_verify_step,
- the RealKvSource dataclass shared with canary_write_step,
- the canary_verify_step host wrapper itself.

The CUDA kernel is not wired up yet; the host wrapper raises NotImplementedError until that lands. Pin the
behaviour through canary_verify_step_torch_reference in kv_canary/verify_ref.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Maximum number of RealKvSource entries the C++ ABI supports per launch. Mirrors kMaxRealKvSources in
# canary_common.cuh. The host wrapper pads any shorter tuple up to this length with dummy entries and
# rejects longer tuples.
_MAX_REAL_KV_SOURCES: Final[int] = 4

# Chain hash anchor. Hardcoded at the jit_kernel layer; no runtime seed parameter exists. Mirrored in C++ as
# kCanaryChainAnchor; const-sync test pins them together.
CANARY_CHAIN_ANCHOR: Final[int] = 0xC0FFEE1234567890

# Bytes per canary slot. 4 int64 fields (token_id, position, prev_hash, real_kv_hash).
CANARY_SLOT_BYTES: Final[int] = 32

# Width of one violation_ring row in int64 fields.
VIOLATION_FIELDS: Final[int] = 8

# Violation-row field offsets. C++ counterpart (kViolationField*) lives in canary_common.cuh. These constants are
# private to this module + the torch reference; downstream readers should consume the schema through
# sglang.srt.kv_canary.state helpers, not by indexing positionally.
#
# Column 6 (_VIOLATION_FIELD_EXPECTED_AUX) is reason-agnostic: its interpretation depends on
# (kernel_kind, fail_reason_bits). Verify launches store ``expected_chain_hash`` there; write launches
# store ``expected_position`` (when pseudo-mode write detects a position mismatch). Callers must dispatch
# on the row's kernel_kind / fail_reason_bits before decoding this field.
_VIOLATION_FIELD_KERNEL_KIND: Final[int] = 0
_VIOLATION_FIELD_SLOT_IDX: Final[int] = 1
_VIOLATION_FIELD_POSITION: Final[int] = 2
_VIOLATION_FIELD_STORED_TOKEN: Final[int] = 3
_VIOLATION_FIELD_EXPECTED_TOKEN: Final[int] = 4
_VIOLATION_FIELD_STORED_CHAIN_HASH: Final[int] = 5
_VIOLATION_FIELD_EXPECTED_AUX: Final[int] = 6
_VIOLATION_FIELD_FAIL_REASON_BITS: Final[int] = 7


class CanaryLaunchTag(IntEnum):
    """Unique tag per (head | tail | sweep) × (K | V) × (FULL | SWA) launch.

    **Python-only — no C++ mirror.** Kernel just stamps the int value into every violation row's kernel_kind
    field; only the Python host reader decodes the tag to attribute violations back to launch slots. Runner
    assigns one tag per launch slot it owns.
    """

    HEAD_K_FULL = 0
    HEAD_V_FULL = 1
    TAIL_K_FULL = 2
    TAIL_V_FULL = 3
    SWEEP_K_FULL = 4
    SWEEP_V_FULL = 5
    HEAD_K_SWA = 6
    HEAD_V_SWA = 7
    TAIL_K_SWA = 8
    TAIL_V_SWA = 9
    SWEEP_K_SWA = 10
    SWEEP_V_SWA = 11


class RealKvHashMode(IntEnum):
    """Selector for the real-KV hash mixin.

    Mirrored in C++ (canary_common.cuh) as kRealKvHashMode*; value parity enforced by
    test_unit_const_sync.py.
    """

    OFF = 0  # mixin disabled; real_kv_hash field is always 0 in the canary slot.
    PARTIAL = 1  # splitmix64-fold first min(16, read_bytes) bytes (cheap: max 16B read, high entropy).
    ALL = 2  # splitmix64-fold all read_bytes (thorough, slower).


# Fail-reason bit positions. Bitfield (not enum) because the verify kernel may set multiple reasons on the
# same violation row.
_FAIL_REASON_BIT_CHAIN_HASH: Final[int] = 1 << 0
_FAIL_REASON_BIT_POSITION: Final[int] = 1 << 1
_FAIL_REASON_BIT_REAL_KV_HASH: Final[int] = 1 << 2


def _assert_contiguous(tensor: torch.Tensor, name: str) -> None:
    """Fail loud when a tensor whose data_ptr() flows into the canary CUDA ABI is non-contiguous.

    The CUDA kernels treat every input tensor as a raw byte/word buffer indexed by element offsets, so a
    non-contiguous input would silently corrupt the read/write. We refuse rather than silently call
    ``.contiguous()`` so the caller fixes the upstream allocation.
    """
    if not tensor.is_contiguous():
        raise ValueError(f"kv-canary: {name} must be contiguous")


@dataclass(frozen=True, slots=True, kw_only=True)
class RealKvSource:
    """One piece of real KV the canary folds into its fingerprint.

    Slot access invariant (must hold for every source, regardless of underlying layout) — for a given slot_idx,
    the canary reads exactly these bytes:

        tensor[
            slot_idx // page_size,
            (slot_idx % page_size) * num_bytes_per_token
            : ((slot_idx % page_size) + 1) * num_bytes_per_token
        ]

    Note that ``tensor`` may have "holes" in dim 1 — ``tensor.shape[1]`` can exceed ``page_size *
    num_bytes_per_token``. Trailing bytes of each row are ignored by the canary; this is exactly how the
    abstraction accommodates pools whose per-row layout interleaves canary-relevant bytes with other metadata
    (DSV4-style packed pools, layer-split storage, K/V interleaving, ...). When ``page_size == 1`` the pattern
    collapses to the simple ``tensor[slot_idx, :num_bytes_per_token]`` case.

    A pool may expose multiple RealKvSource instances per (canary buffer × K/V half) — canary_*_step iterates
    the source list and folds each into the running real_kv_hash via splitmix64 (one int64 fingerprint per slot,
    regardless of source count).

    Pool patchers construct sources by:
    - viewing / reshaping the underlying KV layer into the canonical [num_rows, dim1_bytes] form (no stage-copy
      needed when the underlying storage is already row-major contiguous on dim 0),
    - choosing ``page_size`` and ``num_bytes_per_token`` so that the access pattern above lands on the bytes
      the canary should fingerprint,
    - leaving any per-row padding / non-canary bytes in the trailing portion of each row (they will simply be
      skipped).

    Fields:
        tensor: The source tensor, any shape such that the access pattern above yields ``num_bytes_per_token``
            uint8 bytes per slot. Dtype is whatever the underlying pool uses; the canary views the relevant
            bytes via ``.view(torch.uint8)``.
        page_size: Number of slots packed into one row of dim 0. ``>= 1``.
        num_bytes_per_token: Bytes per slot in the dim-1 strip the canary reads.
        read_bytes: Leading bytes (out of ``num_bytes_per_token``) per slot folded into the fingerprint.
            ``0 <= read_bytes <= num_bytes_per_token``; ``0`` skips this source's contribution this call.
    """

    tensor: torch.Tensor
    page_size: int
    num_bytes_per_token: int
    read_bytes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyPlan:
    """Flat verify entries consumed by canary_verify_step.

    Each row is a self-contained (slot_idx, position, prev_slot_idx) triple, so the verify kernel makes no
    assumption about the entry's source — per-forward derivation, sweep over running reqs, and sweep over
    radix-cache orphan slots all populate the same schema. prev_slot_idx == -1 flags a chain-seed entry (kernel
    anchors on the hardcoded CANARY_CHAIN_ANCHOR constant instead of reading a predecessor; see §6.1).

    Sized to a cuda-graph-captured capacity; active prefix is verify_num_valid[0]. Padding tail entries are
    unspecified — kernel skips tid >= verify_num_valid[0].

    Fields:
        verify_slot_indices: Canary slot index per entry, shape [verify_capacity], int32. Already SWA-translated
            for the SWA group.
        verify_positions: Expected sequence position per entry, shape [verify_capacity], int32.
        verify_prev_slot_indices: Chain predecessor slot per entry, shape [verify_capacity], int32. -1 = chain
            head (anchor on CANARY_CHAIN_ANCHOR; §6.1). Explicit (not derived from verify_slot_indices[i-1])
            because chain heads, SWA window starts, cross-req boundaries, and radix-orphan extras break the
            "predecessor == previous array entry" assumption.
        verify_num_valid: Active entry count, shape [1], int32.
    """

    verify_slot_indices: torch.Tensor
    verify_positions: torch.Tensor
    verify_prev_slot_indices: torch.Tensor
    verify_num_valid: torch.Tensor

    @classmethod
    def allocate(cls, *, verify_capacity: int, device: torch.device) -> "VerifyPlan":
        """Allocate a fresh VerifyPlan, all zeros."""
        if verify_capacity <= 0:
            raise ValueError(
                f"kv-canary: VerifyPlan verify_capacity must be positive, got {verify_capacity}"
            )
        return cls(
            verify_slot_indices=torch.zeros(
                verify_capacity, dtype=torch.int32, device=device
            ),
            verify_positions=torch.zeros(
                verify_capacity, dtype=torch.int32, device=device
            ),
            verify_prev_slot_indices=torch.zeros(
                verify_capacity, dtype=torch.int32, device=device
            ),
            verify_num_valid=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def reset_to_skip_sentinel(self) -> None:
        """Zero verify_num_valid; other tensors are left as-is."""
        self.verify_num_valid.zero_()


def canary_verify_step(
    *,
    canary_buf: torch.Tensor,
    plan: VerifyPlan,
    kernel_kind: CanaryLaunchTag,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
) -> None:
    """Verify one canary buffer against a VerifyPlan.

    One CUDA thread per active verify entry. Each thread reads the slot's 4 stored int64 fields (token_id,
    position, prev_hash, real_kv_hash), recomputes the expected prev_hash from the predecessor slot (or from
    splitmix64(CANARY_CHAIN_ANCHOR) for chain heads, signaled by prev_slot_idx == -1), and atomically appends
    any mismatch (chain hash / position / real_kv_hash) to violation_ring. Read-only on canary_buf.

    Canary slot layout: each slot is canary_buf.shape[1] bytes holding 4 int64 fields (token_id, position,
    prev_hash, real_kv_hash). Chain link: next.prev_hash == splitmix64(this.prev_hash XOR this.token_id XOR
    this.position XOR this.real_kv_hash); chain head anchors on splitmix64(CANARY_CHAIN_ANCHOR), where
    CANARY_CHAIN_ANCHOR is a hardcoded module-level constant (§6.1; no runtime seed parameter — the canary is
    for bug detection, not adversarial security, so a fixed anchor is sufficient).

    Args:
        canary_buf: Canary buffer this launch verifies, shape [num_slots, slot_stride_bytes], uint8.
            slot_stride_bytes is read from canary_buf.shape[1].
        plan: Pre-allocated VerifyPlan; addresses baked into cuda-graph capture.
        kernel_kind: CanaryLaunchTag identifying which launch fired. Stamped (as int) into every violation row
            so host can attribute a violation back to its source launch.
        violation_ring: Global append-only sink, shape [ring_capacity, VIOLATION_FIELDS], int64. Shared across
            all canary launches; fill-once.
        violation_write_index: Global monotonic violation counter, shape [1], int32.
        slot_run_counter: Health counter, shape [1], int64. Incremented by the active entry count processed.
        kernel_run_counter: Health counter, shape [1], int64. Incremented by 1 per call (even for all-padding
            plans).
        real_kv_sources: Real KV pieces folded into each slot's real_kv_hash, as a tuple of RealKvSource. Empty
            tuple disables the mixin. Multiple sources are folded sequentially via splitmix64. Each source's
            buf is shape [num_slots, slot_stride_bytes], uint8.
        real_kv_hash_mode: RealKvHashMode (OFF / PARTIAL / ALL). Applies uniformly across all sources.

    Slot 0 is unconditionally skipped by the verify kernel — it is sglang's reserved padding sink per
    ``memory_pool.py:152``. All canary-attached pools MUST reserve slot 0 (free_slots starts at 1).

    Implementation:
        - CUDA __global__ `canary_verify_kernel`: 1-D grid `((verify_capacity + 127) / 128, 1, 1)` blocks ×
          `(128, 1, 1)` threads; tid = blockIdx.x * 128 + threadIdx.x = one verify entry. Early-exit on
          tid >= plan.verify_num_valid[0].
        - Per thread, gather:
          (a) self_slot = canary_buf[plan.verify_slot_indices[tid]] (32 B uint8 load, vectorized as 4× int64).
          (b) prev_slot = canary_buf[plan.verify_prev_slot_indices[tid]] when prev >= 0 (same shape); else
              expected_prev_hash = splitmix64(CANARY_CHAIN_ANCHOR).
          (c) For each src in real_kv_sources: read src.read_bytes leading bytes from src.tensor[...] (per the
              RealKvSource access invariant) and splitmix64-fold into running_real_kv_hash.
        - Compare expected vs stored (chain hash, position, real_kv_hash) and accumulate fail_reason bits; if
          non-zero → record_violation().
        - record_violation(): idx = atomicAdd(violation_write_index, 1); if idx < ring_capacity, atomic-write
          the 8 int64 fields to violation_ring[idx] (kernel_kind, slot_idx, position, stored vs expected
          fields, fail_reason).
        - Counters: warp-reduce per-thread "did I process an active entry?" via __ballot_sync + popc, then
          warp-leader atomicAdd to slot_run_counter. kernel_run_counter += 1: single thread (tid == 0) does an
          atomicAdd once per launch.

    Calling contract:
        - Pure side-effect; never raises. Host polls violation_write_index[0] > 0 for is_errored and
          violation_ring[0] for the first violation.
        - kernel_run_counter is bumped every call (canary-ran health signal).
        - Safe in cuda-graph capture; caller refills plan in-place before replay.

    Pinned by torch reference
    :func:`sglang.jit_kernel.kv_canary.verify_ref.canary_verify_step_torch_reference`; CUDA must match
    byte-for-byte.
    """
    if len(real_kv_sources) > _MAX_REAL_KV_SOURCES:
        raise ValueError(
            f"kv-canary: at most {_MAX_REAL_KV_SOURCES} RealKvSource entries supported by the CUDA ABI, "
            f"got {len(real_kv_sources)}"
        )

    _assert_contiguous(canary_buf, "canary_buf")
    _assert_contiguous(plan.verify_slot_indices, "plan.verify_slot_indices")
    _assert_contiguous(plan.verify_positions, "plan.verify_positions")
    _assert_contiguous(plan.verify_prev_slot_indices, "plan.verify_prev_slot_indices")
    _assert_contiguous(plan.verify_num_valid, "plan.verify_num_valid")
    _assert_contiguous(violation_ring, "violation_ring")
    _assert_contiguous(violation_write_index, "violation_write_index")
    _assert_contiguous(slot_run_counter, "slot_run_counter")
    _assert_contiguous(kernel_run_counter, "kernel_run_counter")

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    module = _jit_canary_verify_module()
    module.canary_verify_step_cuda(
        canary_buf,
        plan.verify_slot_indices,
        plan.verify_positions,
        plan.verify_prev_slot_indices,
        plan.verify_num_valid,
        int(kernel_kind),
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
        padded_bufs[0],
        padded_bufs[1],
        padded_bufs[2],
        padded_bufs[3],
        source_params,
        len(real_kv_sources),
        int(real_kv_hash_mode),
    )


@cache_once
def _jit_canary_verify_module() -> "Module":
    """Lazy-load the CUDA verify module via tvm-ffi."""
    return load_jit(
        "kv_canary_verify",
        cuda_files=["kv_canary/canary_verify.cuh"],
        cuda_wrappers=[
            ("canary_verify_step_cuda", "canary::canary_verify_step_cuda"),
        ],
    )


def _build_real_kv_source_abi(
    *,
    real_kv_sources: tuple[RealKvSource, ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pad a RealKvSource tuple up to _MAX_REAL_KV_SOURCES dummy entries and build the (bufs, params) ABI.

    Returns:
        padded_bufs: list of length _MAX_REAL_KV_SOURCES; uint8 2-D tensors on ``device``. Unused trailing
            slots are tiny 1-byte placeholders that the kernel never dereferences (their read_bytes is 0).
        source_params: int32 tensor on CPU, shape [_MAX_REAL_KV_SOURCES, 3]. Per row: (page_size,
            num_bytes_per_token, read_bytes). Padding rows are all zeros except page_size = 1 and
            num_bytes_per_token = 1 to keep the device-side address arithmetic well-defined.
    """
    padded_bufs: list[torch.Tensor] = []
    params = torch.zeros((_MAX_REAL_KV_SOURCES, 3), dtype=torch.int32, device="cpu")

    for i, source in enumerate(real_kv_sources):
        _assert_contiguous(source.tensor, f"real_kv_sources[{i}].tensor")
        source_u8 = source.tensor.view(torch.uint8)
        if source_u8.dim() != 2:
            raise ValueError(
                f"kv-canary: real_kv_sources[{i}].tensor (viewed as uint8) must be 2-D, "
                f"got {source_u8.dim()}-D"
            )
        padded_bufs.append(source_u8)
        params[i, 0] = source.page_size
        params[i, 1] = source.num_bytes_per_token
        params[i, 2] = source.read_bytes

    dummy = torch.zeros((1, 1), dtype=torch.uint8, device=device)
    for i in range(len(real_kv_sources), _MAX_REAL_KV_SOURCES):
        padded_bufs.append(dummy)
        params[i, 0] = 1  # page_size
        params[i, 1] = 1  # num_bytes_per_token
        params[i, 2] = 0  # read_bytes -> kernel skips this slot

    return padded_bufs, params
