from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Bytes per canary slot = CANARY_FIELDS_PER_SLOT * 8.
CANARY_SLOT_BYTES: Final[int] = consts.CANARY_FIELDS_PER_SLOT * 8


class CanaryLaunchTag(IntEnum):
    """Unique tag per (head | tail | sweep) × (K | V) × (FULL | SWA) launch."""

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


def _assert_contiguous(tensor: torch.Tensor, name: str) -> None:
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
    (layer-split storage, K/V interleaving, ...). When ``page_size == 1`` the pattern
    collapses to the simple ``tensor[slot_idx, :num_bytes_per_token]`` case.

    A pool may expose multiple RealKvSource instances per (canary buffer × K/V half) — the launch wrappers
    iterate the source list and fold each into the running real_kv_hash via splitmix64 (one int64 fingerprint
    per slot, regardless of source count).

    Pool patchers construct sources by:
    - viewing / reshaping the underlying KV layer into the canonical [num_rows, dim1_bytes] form (no stage-copy
      needed when the underlying storage is already row-major contiguous on dim 0),
    - choosing ``page_size`` and ``num_bytes_per_token`` so that the access pattern above lands on the bytes
      the canary should fingerprint,
    - leaving any per-row padding / non-canary bytes in the trailing portion of each row (they will simply be
      skipped).

    16-byte alignment precondition: the CUDA fold kernel issues 128-bit aligned loads, so ``read_bytes``,
    ``num_bytes_per_token``, and the row stride (``tensor.shape[1]`` in bytes) must all be positive
    multiples of 16. There is no "skip this source" sentinel — callers omit the source from their
    ``real_kv_sources`` tuple entirely (factory helpers return an empty tuple in that case).

    Fields:
        tensor: The source tensor, any shape such that the access pattern above yields ``num_bytes_per_token``
            uint8 bytes per slot. Dtype is whatever the underlying pool uses; the canary views the relevant
            bytes via ``.view(torch.uint8)``.
        page_size: Number of slots packed into one row of dim 0. ``>= 1``.
        num_bytes_per_token: Bytes per slot in the dim-1 strip the canary reads. Must be a positive
            multiple of 16.
        read_bytes: Leading bytes (out of ``num_bytes_per_token``) per slot folded into the fingerprint.
            Must be a positive multiple of 16, ``<= num_bytes_per_token``.
    """

    tensor: torch.Tensor
    page_size: int
    num_bytes_per_token: int
    read_bytes: int

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError(
                f"kv-canary: RealKvSource.page_size must be >= 1, got {self.page_size}"
            )
        if self.num_bytes_per_token <= 0 or self.num_bytes_per_token % 16 != 0:
            raise ValueError(
                f"kv-canary: RealKvSource.num_bytes_per_token must be a positive multiple of 16, "
                f"got {self.num_bytes_per_token}"
            )
        if (
            self.read_bytes <= 0
            or self.read_bytes > self.num_bytes_per_token
            or self.read_bytes % 16 != 0
        ):
            raise ValueError(
                f"kv-canary: RealKvSource.read_bytes must be a positive multiple of 16 in "
                f"(0, num_bytes_per_token={self.num_bytes_per_token}], got {self.read_bytes}"
            )
        if self.tensor.ndim < 2:
            raise ValueError(
                f"kv-canary: RealKvSource.tensor must be at least 2-D, got shape {tuple(self.tensor.shape)}"
            )
        row_stride_bytes = int(self.tensor.shape[1]) * self.tensor.element_size()
        if row_stride_bytes % 16 != 0:
            raise ValueError(
                f"kv-canary: RealKvSource.tensor dim-1 byte width must be a multiple of 16, "
                f"got {row_stride_bytes} bytes (shape={tuple(self.tensor.shape)}, "
                f"dtype={self.tensor.dtype})"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyOrWriteContext:
    """Shared launch context for canary verify/write kernels.

    Fields:
        canary_buf: Canary buffer this launch verifies or writes, shape [num_slots, slot_stride_bytes], uint8.
            slot_stride_bytes is read from canary_buf.shape[1].
        kernel_kind: CanaryLaunchTag identifying which launch fired. Stamped (as int) into every violation row
            so host can attribute a violation back to its source launch.
        violation_ring: Global append-only sink, shape [ring_capacity, VIOLATION_FIELDS], int64. Shared across
            all canary launches; fill-once.
        violation_write_index: Global monotonic violation counter, shape [1], int32.
        slot_run_counter: Health counter, shape [1], int64. Verify increments by active entries processed;
            write increments by write entries processed.
        kernel_run_counter: Health counter, shape [1], int64. Incremented by 1 per call.
        real_kv_sources: Real KV pieces folded into each slot's real_kv_hash, as a tuple of RealKvSource. Empty
            tuple disables the mixin. Multiple sources are folded sequentially via splitmix64 to produce one
            int64 fingerprint per slot.
        real_kv_hash_mode: RealKvHashMode (NONE / PARTIAL / ALL). Applies uniformly across all sources.
        enable_chain_position_assert: int32 [1] device flag gating the write kernel's chain-step
            write_position assert. 0 during warmup / cuda-graph capture; flipped to 1 in
            CanaryManager.mark_init_finished().
    """

    canary_buf: torch.Tensor
    kernel_kind: CanaryLaunchTag
    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor
    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor
    real_kv_sources: tuple[RealKvSource, ...]
    real_kv_hash_mode: consts.RealKvHashMode
    enable_chain_position_assert: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyPlan:
    """Flat verify entries consumed by launch_canary_verify_kernel.

    Each row is a self-contained (slot_idx, position, prev_slot_idx) triple, so the verify kernel makes no
    assumption about the entry's source — per-forward derivation, sweep over running reqs, and sweep over
    radix-cache orphan slots all populate the same schema. prev_slot_idx == -1 flags a chain-seed entry (kernel
    anchors on the hardcoded CANARY_CHAIN_ANCHOR constant instead of reading a predecessor).

    Sized to a cuda-graph-captured capacity; active prefix is verify_num_valid[0]. Padding tail entries are
    unspecified — kernel skips tid >= verify_num_valid[0].

    Fields:
        verify_slot_indices: Canary slot index per entry, shape [verify_capacity], int64. Already SWA-translated
            for the SWA group.
        verify_expected_tokens: Source-of-truth token id per entry, shape [verify_capacity], int64.
            The plan-side entries kernel gathers from
            ``CanaryDeviceState.req_to_verify_expected_tokens[rp, position + kv_token_id_vs_position_offset]``;
            entries that fall outside the pool's row (e.g. EAGLE draft's last slot rotating in a bonus
            token, or padding beyond the per-req length) get the ``-1`` sentinel. The verify kernel
            compares against the stored canary token and skips when this is ``-1``.
        verify_expected_positions: Expected sequence position per entry, shape [verify_capacity], int64.
        verify_prev_slot_indices: Chain predecessor slot per entry, shape [verify_capacity], int64. -1 = chain
            head (anchor on CANARY_CHAIN_ANCHOR). Explicit (not derived from verify_slot_indices[i-1])
            because chain heads, SWA window starts, cross-req boundaries, and radix-orphan extras break the
            "predecessor == previous array entry" assumption.
        verify_num_valid: Active entry count, shape [1], int32. Clamped by the plan kernel to
            min(total_requested, verify_capacity) so the verify kernel grid never reads past the buffer.
        enable: Run-this-step flag, shape [1], int32. 1 = verify kernel runs as usual; 0 = the plan kernel
            detected overflow (requested > verify_capacity) and the entire verify launch is skipped this step.
            Allocated as 1 by default; the plan kernel rewrites it every step.
    """

    verify_slot_indices: torch.Tensor
    verify_expected_tokens: torch.Tensor
    verify_expected_positions: torch.Tensor
    verify_prev_slot_indices: torch.Tensor
    verify_num_valid: torch.Tensor
    enable: torch.Tensor

    @classmethod
    def allocate(cls, *, verify_capacity: int, device: torch.device) -> VerifyPlan:
        if verify_capacity <= 0:
            raise ValueError(
                f"kv-canary: VerifyPlan verify_capacity must be positive, got {verify_capacity}"
            )
        return cls(
            verify_slot_indices=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_expected_tokens=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_expected_positions=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_prev_slot_indices=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_num_valid=torch.empty(1, dtype=torch.int32, device=device),
            # enable defaults to 1 ("run verify") so test helpers that build a VerifyPlan
            # directly (no plan kernel) don't have to remember to set it. Plan kernel always
            # overwrites this so the default is observable only when no plan kernel runs.
            enable=torch.ones(1, dtype=torch.int32, device=device),
        )

    def zero_for_testing_(self) -> VerifyPlan:
        """WARN: ONLY use it when testing plan kernel. Do not use it when testing verify or
        write kernel to avoid hiding bugs."""
        self.verify_slot_indices.zero_()
        # Test helpers expect the "skip token check" sentinel after zero-out, matching
        # the verify-kernel contract.
        self.verify_expected_tokens.fill_(-1)
        self.verify_expected_positions.zero_()
        self.verify_prev_slot_indices.zero_()
        self.verify_num_valid.zero_()
        self.enable.zero_()
        return self


def launch_canary_verify_kernel(
    *,
    context: VerifyOrWriteContext,
    plan: VerifyPlan,
    check_verify_expected_token: bool,
) -> None:
    """Verify one canary buffer against a VerifyPlan.

    A fixed persistent grid of `kPersistentBlocks * kVerifyBlockSize` CUDA threads grid-strides over active
    verify entries. Each thread reads the slot's 4 stored int64 fields (token_id, position, prev_hash,
    real_kv_hash), recomputes the expected prev_hash from the predecessor slot (or from
    splitmix64(CANARY_CHAIN_ANCHOR) for chain heads, signaled by prev_slot_idx == -1), and atomically appends
    any mismatch (chain hash / position / real_kv_hash) to violation_ring. Read-only on canary_buf.

    Canary slot layout: each slot is canary_buf.shape[1] bytes holding 4 int64 fields (token_id, position,
    prev_hash, real_kv_hash). Chain link: next.prev_hash == splitmix64_mix3(this.prev_hash, this.token_id,
    this.position), where splitmix64_mix3 folds each input into a running accumulator
    via ``acc = splitmix64(acc ^ next)`` starting from ``splitmix64(prev_hash)``. ``real_kv_hash`` is NOT
    folded into the chain (see ``compute_slot_hash`` rationale: keeps chain content-only and immune to
    legitimate radix prefix folding). Chain head anchors on
    splitmix64(CANARY_CHAIN_ANCHOR), where CANARY_CHAIN_ANCHOR is a hardcoded module-level constant (no
    runtime seed parameter — the canary is for bug detection, not adversarial security, so a fixed anchor
    is sufficient).

    Args:
        context: Shared verify/write launch context, including canary buffer, launch tag, violation sink,
            health counters, and real KV fingerprint sources.
        plan: Pre-allocated VerifyPlan; addresses baked into cuda-graph capture.

    Token-to-KV slot 0 is unconditionally skipped by the verify kernel: SGLang's TokenToKVPoolAllocator
    reserves it for padded-token dummy writes, and zero-initialized req_to_token entries therefore point to
    a non-real KV slot. Canary-attached pools mirror that contract by reserving canary slot 0.

    Implementation:
        - CUDA __global__ `canary_verify_kernel`: fixed 1-D grid `(kPersistentBlocks=64, 1, 1)` blocks ×
          `(kVerifyBlockSize=512, 1, 1)` threads (= 32768 threads total). Each thread grid-strides over
          verify entries `entry_idx ∈ [tid, tid + grid_threads, ...)` until
          `min(plan.verify_num_valid[0], verify_capacity)`.
        - Per thread, gather:
          (a) self_slot fields: 4 separate ``canary_load_field`` int64 loads from
              canary_buf[plan.verify_slot_indices[tid]] for (token, position, prev_hash, real_kv_hash).
          (b) expected_prev_hash = compute_slot_hash(canary_buf, slot_stride_bytes, prev_slot_idx), which
              folds only (token, position, prev_hash) from canary_buf[plan.verify_prev_slot_indices[tid]];
              prev_slot_idx == -1 anchors at splitmix64(CANARY_CHAIN_ANCHOR).
          (c) For each src in real_kv_sources: read src.read_bytes leading bytes from src.tensor[...] (per the
              RealKvSource access invariant) and splitmix64-fold into running_real_kv_hash.
        - Compare expected vs stored (chain hash, position, real_kv_hash) and accumulate fail_reason bits; if
          non-zero → record_violation().
        - record_violation(): idx = atomicAdd(violation_write_index, 1); if idx < ring_capacity, atomic-write
          the 8 int64 fields to violation_ring[idx] (kernel_kind, slot_idx, position, stored vs expected
          fields, fail_reason).
        - Counters: each thread maintains a local count of active entries it processed, warp-reduces via
          ``__shfl_down_sync`` (offsets 16..1), then the warp leader (lane 0) does a single atomicAdd of the
          warp's summed count into slot_run_counter. kernel_run_counter += 1: single thread (tid == 0) does an
          atomicAdd once per launch.

    Calling contract:
        - Pure side-effect; never raises. Host polls violation_write_index[0] > 0 for is_errored and
          violation_ring[0] for the first violation.
        - kernel_run_counter is bumped every call (canary-ran health signal).
        - Safe in cuda-graph capture; caller refills plan in-place before replay.

    Pinned by torch reference
    :func:`sglang.jit_kernel.kv_canary.verify_ref.launch_canary_verify_kernel_torch_reference`; CUDA must match
    byte-for-byte.
    """
    canary_buf = context.canary_buf
    real_kv_sources = context.real_kv_sources
    if len(real_kv_sources) > consts.MAX_REAL_KV_SOURCES:
        raise ValueError(
            f"kv-canary: at most {consts.MAX_REAL_KV_SOURCES} RealKvSource entries supported by the CUDA ABI, "
            f"got {len(real_kv_sources)}"
        )

    _assert_contiguous(canary_buf, "canary_buf")
    _assert_contiguous(plan.verify_slot_indices, "plan.verify_slot_indices")
    _assert_contiguous(plan.verify_expected_tokens, "plan.verify_expected_tokens")
    _assert_contiguous(plan.verify_expected_positions, "plan.verify_expected_positions")
    _assert_contiguous(plan.verify_prev_slot_indices, "plan.verify_prev_slot_indices")
    _assert_contiguous(plan.verify_num_valid, "plan.verify_num_valid")
    _assert_contiguous(plan.enable, "plan.enable")
    _assert_contiguous(context.violation_ring, "violation_ring")
    _assert_contiguous(context.violation_write_index, "violation_write_index")
    _assert_contiguous(context.slot_run_counter, "slot_run_counter")
    _assert_contiguous(context.kernel_run_counter, "kernel_run_counter")

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    module = _jit_canary_verify_module(check_verify_expected_token)
    module.canary_verify_step_cuda(
        canary_buf,
        plan.verify_slot_indices,
        plan.verify_expected_tokens,
        plan.verify_expected_positions,
        plan.verify_prev_slot_indices,
        plan.verify_num_valid,
        plan.enable,
        int(context.kernel_kind),
        context.violation_ring,
        context.violation_write_index,
        context.slot_run_counter,
        context.kernel_run_counter,
        padded_bufs[0],
        padded_bufs[1],
        padded_bufs[2],
        padded_bufs[3],
        source_params,
        len(real_kv_sources),
        int(context.real_kv_hash_mode),
    )


@cache_once
def _jit_canary_verify_module(check_verify_expected_token: bool) -> Module:
    args = make_cpp_args(check_verify_expected_token)
    return load_jit(
        "kv_canary_verify",
        *args,
        cuda_files=["kv_canary/canary_verify.cuh"],
        cuda_wrappers=[
            (
                "canary_verify_step_cuda",
                f"canary::CanaryVerifyKernel<{args}>::run",
            ),
        ],
    )


def _build_real_kv_source_abi(
    *,
    real_kv_sources: tuple[RealKvSource, ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    padded_bufs: list[torch.Tensor] = []
    params = torch.zeros(
        (consts.MAX_REAL_KV_SOURCES, consts.REAL_KV_SOURCE_FIELDS_PER_ENTRY),
        dtype=torch.int32,
        device="cpu",
    )

    for i, source in enumerate(real_kv_sources):
        _assert_contiguous(source.tensor, f"real_kv_sources[{i}].tensor")
        source_u8 = source.tensor.view(torch.uint8)
        if source_u8.dim() != 2:
            raise ValueError(
                f"kv-canary: real_kv_sources[{i}].tensor (viewed as uint8) must be 2-D, "
                f"got {source_u8.dim()}-D"
            )
        padded_bufs.append(source_u8)
        params[i, consts.REAL_KV_SOURCE_FIELD_PAGE_SIZE] = source.page_size
        params[i, consts.REAL_KV_SOURCE_FIELD_NUM_BYTES_PER_TOKEN] = (
            source.num_bytes_per_token
        )
        params[i, consts.REAL_KV_SOURCE_FIELD_READ_BYTES] = source.read_bytes

    # Pad bufs (never read by the kernel — num_sources bounds the iteration); params already zero.
    dummy = torch.empty((1, 1), dtype=torch.uint8, device=device)
    for _ in range(len(real_kv_sources), consts.MAX_REAL_KV_SOURCES):
        padded_bufs.append(dummy)

    return padded_bufs, params
