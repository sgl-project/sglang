"""KV cache canary write kernel — host wrapper, dataclasses, enums.

This module defines the public API surface for the write kernel:

- CanaryPseudoMode IntEnum,
- the WritePlan dataclass consumed by canary_write_step,
- the canary_write_step host wrapper itself.

RealKvSource is owned by kv_canary/verify.py (the verify path defines it first); we re-import it here so
the write module's API surface is self-contained for callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _MAX_REAL_KV_SOURCES,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
    _assert_contiguous,
    _build_real_kv_source_abi,
)
from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


class CanaryPseudoMode(IntEnum):
    """Toggle for caller-driven (token, position) expectation checks in canary_write_step.

    The kernel itself knows no oracle. When ON, the caller supplies pseudo_expected_tokens /
    pseudo_expected_positions tensors (computed by whatever oracle the caller chose) and the kernel
    compares actuals against them per chain step; mismatch → violation, chain still advances on actuals.

    Mirrored in C++ as kCanaryPseudoMode*; value parity enforced by test_unit_const_sync.py.
    """

    OFF = 0  # No comparison; pseudo_expected_* tensors ignored (may be unallocated / cuda-graph dummies).
    ON = 1  # Compare fb_input_ids[i] / fb_positions[i] vs pseudo_expected_tokens[i] / pseudo_expected_positions[i].


@dataclass(frozen=True, slots=True, kw_only=True)
class WritePlan:
    """Write plan consumed by canary_write_step: per-token slot indices + per-req metadata.

    Fully per-req — no per-token tile. canary_write_step uses write_offsets to map each thread's (req, j) into a
    flat index i, then reads token-level data from ForwardBatch's input_ids / positions / out_cache_loc[i]
    directly. SWA translation of write slots is done inline in canary_write_step via full_to_swa_index_mapping;
    only the chain-seed slot (a per-req gather from req_to_token at plan time) is SWA-translated here.

    Req r's write entries occupy flat indices [write_offsets[r], write_offsets[r+1]). seed_slot_idx == -1 means
    K_req_old == 0 (anchor on CANARY_CHAIN_ANCHOR; §6.1).

    Sweep callers fill an "empty" WritePlan (write_num_valid_reqs = 0) and skip canary_write_step entirely —
    plan kernel always produces both VerifyPlan and WritePlan in one launch, so an unused WritePlan is the
    price of that fusion.

    Fields:
        write_offsets: Exclusive prefix-sum offsets indexing into ForwardBatch's input_ids / positions /
            out_cache_loc, shape [write_req_capacity + 1], int32. write_offsets[0] == 0;
            write_offsets[write_num_valid_reqs[0]] == total_write_entries.
        write_seed_slot_indices: Chain-seed slot per write req, shape [write_req_capacity], int32. Already
            SWA-translated. -1 = no prefix (chain anchors on CANARY_CHAIN_ANCHOR; §6.1).
        write_num_valid_reqs: Active write-req count, shape [1], int32. canary_write_step skips blocks with
            block_id >= write_num_valid_reqs[0].
    """

    write_offsets: torch.Tensor
    write_seed_slot_indices: torch.Tensor
    write_num_valid_reqs: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        write_req_capacity: int,
        device: torch.device,
    ) -> "WritePlan":
        """Allocate a fresh WritePlan, all zeros."""
        if write_req_capacity <= 0:
            raise ValueError(
                f"kv-canary: WritePlan write_req_capacity must be positive, got {write_req_capacity}"
            )
        return cls(
            write_offsets=torch.zeros(
                write_req_capacity + 1, dtype=torch.int32, device=device
            ),
            write_seed_slot_indices=torch.zeros(
                write_req_capacity, dtype=torch.int32, device=device
            ),
            write_num_valid_reqs=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def reset_to_skip_sentinel(self) -> None:
        """Zero write_num_valid_reqs; other tensors are left as-is."""
        self.write_num_valid_reqs.zero_()


# Write-launch fail-reason bits. Distinct from the verify-launch bits in
# kv_canary.verify (CHAIN_HASH=1<<0, POSITION=1<<1, REAL_KV_HASH=1<<2) because
# both launches share the same violation ring; a single bit must unambiguously
# identify the failing field across kernel kinds.
_FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH: int = 1 << 3
_FAIL_REASON_BIT_WRITE_POSITION_MISMATCH: int = 1 << 4


def canary_write_step(
    *,
    canary_buf: torch.Tensor,
    plan: WritePlan,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    kernel_kind: CanaryLaunchTag,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
) -> None:
    """Write canary fingerprints into one canary buffer per a WritePlan.

    Grid: one CUDA block per active write req, single thread per block (chain is intrinsically serial).
    Block r walks entries ``[plan.write_offsets[r], plan.write_offsets[r+1])``. Per chain step ``i``:

    - ``slot`` = ``full_to_swa_index_mapping[fb_out_cache_loc[i]]`` if mapping non-None else ``fb_out_cache_loc[i]``.
    - ``token / position`` = ``fb_input_ids[i] / fb_positions[i]``.
    - ``real_kv_hash`` = ``fold_real_kv_sources(real_kv_sources, slot)`` if ``real_kv_hash_mode != OFF`` else 0.
    - Store 4 int64s ``(token, position, running_prev_hash, real_kv_hash)`` into ``canary_buf[slot]``.
    - Advance ``running_prev_hash = splitmix64(prev XOR token XOR position XOR real_kv_hash)``.

    Initial ``running_prev_hash`` when ``seed_slot_idx >= 0``: load the 4 int64 fields from
    ``canary_buf[plan.write_seed_slot_indices[r]]`` and set
    ``running_prev_hash = splitmix64(seed.prev_hash XOR seed.token XOR seed.position XOR seed.real_kv_hash)``
    (i.e. apply the same advance step that produced ``seed``'s successor — this keeps slot[0]'s stored
    ``prev_hash`` consistent with §6.1's chain link). Else
    ``running_prev_hash = splitmix64(CANARY_CHAIN_ANCHOR)``. ``write_seed_slot_indices`` is already
    SWA-translated by the plan kernel; ``CANARY_CHAIN_ANCHOR`` is hardcoded module-level (§6.1, no runtime seed).

    Pseudo-mode (caller-driven, kernel is oracle-agnostic): when ``pseudo_mode == ON`` the kernel additionally
    compares ``fb_input_ids[i]`` against ``pseudo_expected_tokens[i]`` and ``fb_positions[i]`` against
    ``pseudo_expected_positions[i]``; mismatch on either field records a violation. The chain still advances on
    the actual values (not the expected ones) so a downstream verify won't cascade. Whoever produced the pseudo
    inputs is responsible for filling these expected tensors; the kernel runs no oracle internally.

    Write only writes canary_buf (reads only at seed slots). Block uses no shared memory.

    The ``fb_*`` arguments are passed through unchanged from the source ForwardBatch — canary does not transform
    them.

    Args:
        canary_buf: Canary buffer this launch writes into, shape [num_slots, slot_stride_bytes], uint8.
        plan: Pre-allocated WritePlan.
        fb_input_ids: ForwardBatch.input_ids; token ids being written, shape [num_tokens_padded], int32.
            Flattened across reqs in plan.write_offsets order; tail beyond
            plan.write_offsets[plan.write_num_valid_reqs[0]] is cuda-graph padding.
        fb_positions: ForwardBatch.positions; sequence positions of fb_input_ids, shape [num_tokens_padded], int32.
        fb_out_cache_loc: ForwardBatch.out_cache_loc; full-pool slot index per token, shape [num_tokens_padded],
            int32. SWA-translated inline by this kernel via full_to_swa_index_mapping.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int32, or None. Required (non-None) iff
            this canary is on the SWA group; the trailing -1 sentinel row maps out-of-window full-pool slots to
            a skip-this-entry marker.
        kernel_kind: CanaryLaunchTag stamped into violation rows (see canary_verify_step). Sweep callers do not
            invoke this kernel.
        pseudo_mode: CanaryPseudoMode toggle. OFF = real-mode, pseudo_expected_* tensors ignored. ON = compare
            each chain step's actual (token, position) against the caller-supplied expected tensors below.
        pseudo_expected_tokens: Expected token id per write entry, shape [num_tokens_padded], int32. Only read
            when pseudo_mode == ON; may be uninitialized / cuda-graph dummy when OFF. Layout mirrors fb_input_ids
            (flattened across reqs in plan.write_offsets order); padding tail is ignored. Filled by the caller
            from whichever oracle drives the pseudo input — the kernel knows no oracle.
        pseudo_expected_positions: Expected position per write entry, shape [num_tokens_padded], int32. Same
            shape/layout/lifetime rules as pseudo_expected_tokens.
        violation_ring: Global append-only sink, shape [ring_capacity, VIOLATION_FIELDS], int64. Shared with
            verify launches.
        violation_write_index: Global monotonic violation counter, shape [1], int32.
        slot_run_counter: Health counter, shape [1], int64. Incremented by the number of write entries processed.
        kernel_run_counter: Health counter, shape [1], int64. Incremented by 1 per call.
        real_kv_sources: Real KV pieces folded into each slot's real_kv_hash, as a tuple of RealKvSource. Empty
            tuple disables the mixin. Folded sequentially via splitmix64 to produce one int64 fingerprint per slot.
        real_kv_hash_mode: RealKvHashMode (OFF / BIT / ALL). Applies uniformly across all sources.

    Implementation:
        - CUDA __global__ `canary_write_kernel`: 1-D grid `(write_req_capacity, 1, 1)` blocks × `(1, 1, 1)` thread
          per block. block_id r = blockIdx.x = one write req; chains are intrinsically serial so a single thread
          per block is optimal (warp-level parallelism would idle 31 lanes).
        - Per block, early-exit on r >= plan.write_num_valid_reqs[0]. Else load entry_start = plan.write_offsets[r],
          entry_count = plan.write_offsets[r+1] - entry_start, seed_slot_idx = plan.write_seed_slot_indices[r] into
          registers.
        - Initialize running_prev_hash: if seed_slot_idx >= 0, load the 4 int64 fields from
          canary_buf[seed_slot_idx] and set running_prev_hash = splitmix64(prev_hash XOR token XOR position XOR
          real_kv_hash); else running_prev_hash = splitmix64(kCanaryChainAnchor).
        - Serial chain loop `for j in range(entry_count)`:
              i = entry_start + j;
              slot_full = fb_out_cache_loc[i];
              slot = (full_to_swa_index_mapping is None) ? slot_full : full_to_swa_index_mapping[slot_full];
              token = fb_input_ids[i]; position = fb_positions[i];
              real_kv_hash = (real_kv_hash_mode == OFF) ? 0 : fold_real_kv_sources(real_kv_sources, slot);
                  // applies RealKvSource access invariant
              if pseudo_mode == ON:
                  if token != pseudo_expected_tokens[i] or position != pseudo_expected_positions[i]:
                      record_violation();  // chain still advances on the ACTUAL (token, position) below
              store (token, position, running_prev_hash, real_kv_hash) to canary_buf[slot] as 4 int64 fields;
              running_prev_hash = splitmix64(running_prev_hash XOR token XOR position XOR real_kv_hash);
        - All chain state lives in the block's single thread's registers. No shared memory, no cross-block
          coordination.
        - record_violation() identical to verify (atomicAdd + atomic-write).
        - Counters: thread of block 0 does atomicAdd(kernel_run_counter, 1); each block accumulates its
          entry_count and atomicAdds to slot_run_counter once at exit.

    Calling contract:
        - Pure side-effect; never raises.
        - Pseudo-mode mismatch records violations but does NOT abort the chain.
        - kernel_run_counter is bumped every call.
        - Safe in cuda-graph capture; caller refills fb_input_ids / fb_positions / fb_out_cache_loc / plan
          in-place before replay.

    Pinned by torch reference
    :func:`sglang.jit_kernel.kv_canary.write_ref.canary_write_step_torch_reference`; CUDA must match
    byte-for-byte.
    """
    if len(real_kv_sources) > _MAX_REAL_KV_SOURCES:
        raise ValueError(
            f"kv-canary: at most {_MAX_REAL_KV_SOURCES} RealKvSource entries supported by the CUDA ABI, "
            f"got {len(real_kv_sources)}"
        )

    _assert_contiguous(canary_buf, "canary_buf")
    _assert_contiguous(plan.write_offsets, "plan.write_offsets")
    _assert_contiguous(plan.write_seed_slot_indices, "plan.write_seed_slot_indices")
    _assert_contiguous(plan.write_num_valid_reqs, "plan.write_num_valid_reqs")
    _assert_contiguous(fb_input_ids, "fb_input_ids")
    _assert_contiguous(fb_positions, "fb_positions")
    _assert_contiguous(fb_out_cache_loc, "fb_out_cache_loc")
    _assert_contiguous(pseudo_expected_tokens, "pseudo_expected_tokens")
    _assert_contiguous(pseudo_expected_positions, "pseudo_expected_positions")
    _assert_contiguous(violation_ring, "violation_ring")
    _assert_contiguous(violation_write_index, "violation_write_index")
    _assert_contiguous(slot_run_counter, "slot_run_counter")
    _assert_contiguous(kernel_run_counter, "kernel_run_counter")
    if full_to_swa_index_mapping is not None:
        _assert_contiguous(full_to_swa_index_mapping, "full_to_swa_index_mapping")

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    # The SWA LUT is presence-flagged via a separate int. When None, pass a tiny dummy tensor that the
    # kernel never dereferences (kSwaMappingAbsent disables the indexing path).
    if full_to_swa_index_mapping is None:
        swa_lut = torch.zeros(1, dtype=torch.int32, device=canary_buf.device)
        swa_present = 0
    else:
        swa_lut = full_to_swa_index_mapping
        swa_present = 1

    module = _jit_canary_write_module()
    module.canary_write_step_cuda(
        canary_buf,
        plan.write_offsets,
        plan.write_seed_slot_indices,
        plan.write_num_valid_reqs,
        fb_input_ids,
        fb_positions,
        fb_out_cache_loc,
        swa_lut,
        swa_present,
        int(kernel_kind),
        int(pseudo_mode),
        pseudo_expected_tokens,
        pseudo_expected_positions,
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
def _jit_canary_write_module() -> "Module":
    """Lazy-load the CUDA write module via tvm-ffi. Same JIT plumbing as the verify loader."""
    return load_jit(
        "kv_canary_write",
        cuda_files=["kv_canary/canary_write.cuh"],
        cuda_wrappers=[
            ("canary_write_step_cuda", "canary::canary_write_step_cuda"),
        ],
    )
