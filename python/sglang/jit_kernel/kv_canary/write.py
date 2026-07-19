from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.verify import (
    VerifyOrWriteContext,
    _assert_contiguous,
    _build_real_kv_source_abi,
)
from sglang.kernels.jit import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@dataclass(frozen=True, slots=True, kw_only=True)
class WritePlan:
    """Write plan consumed by launch_canary_write_kernel: per-token slot indices + per-req metadata.

    Fully per-req — no per-token tile. launch_canary_write_kernel uses write_offsets to map each thread's
    (req, j) into a flat index i, then reads token-level data from input_ids / positions /
    out_cache_loc[i] directly.
    SWA translation of per-token slots is done **host-side by the caller** (typically the endpoint) before
    invoking launch_canary_write_kernel — the kernel is SWA-agnostic and only understands "slot ≥ 0 ⇒ write;
    slot < 0 ⇒ skip this entry". Only the chain-seed slot (a per-req gather from req_to_token at plan time)
    is SWA-translated by the plan kernel and lives in write_seed_slot_indices.

    Req r's write entries occupy flat indices [write_offsets[r], write_offsets[r+1]). seed_slot_idx == -1 means
    K_req_old == 0 (anchor on CANARY_CHAIN_ANCHOR).

    Fields:
        write_offsets: Exclusive prefix-sum offsets indexing into ForwardBatch's input_ids / positions /
            out_cache_loc, shape [write_req_capacity + 1], int64. write_offsets[0] == 0;
            write_offsets[write_num_valid_reqs[0]] == total_write_entries.
        write_seed_slot_indices: Chain-seed slot per write req, shape [write_req_capacity], int64. Already
            SWA-translated. -1 = no prefix (chain anchors on CANARY_CHAIN_ANCHOR).
        write_num_valid_reqs: Active write-req count, shape [1], int32. launch_canary_write_kernel skips blocks
            with block_id >= write_num_valid_reqs[0].
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
    ) -> WritePlan:
        if write_req_capacity <= 0:
            raise ValueError(
                f"kv-canary: WritePlan write_req_capacity must be positive, got {write_req_capacity}"
            )
        return cls(
            write_offsets=torch.empty(
                write_req_capacity + 1, dtype=torch.int64, device=device
            ),
            write_seed_slot_indices=torch.empty(
                write_req_capacity, dtype=torch.int64, device=device
            ),
            write_num_valid_reqs=torch.empty(1, dtype=torch.int32, device=device),
        )

    def zero_for_testing_(self) -> WritePlan:
        """WARN: ONLY use it when testing plan kernel. Do not use it when testing verify or
        write kernel to avoid hiding bugs."""
        self.write_offsets.zero_()
        self.write_seed_slot_indices.zero_()
        self.write_num_valid_reqs.zero_()
        return self


def launch_canary_write_kernel(
    *,
    context: VerifyOrWriteContext,
    plan: WritePlan,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    enable_write_input_assert: bool,
    expected_input_tokens: torch.Tensor | None,
    expected_input_positions: torch.Tensor | None,
) -> None:
    """Write canary fingerprints into one canary buffer per a WritePlan.

    Grid: one CUDA block per active write req, single thread per block (chain is intrinsically serial).
    Block r walks entries ``[plan.write_offsets[r], plan.write_offsets[r+1])``. Per chain step ``i``:

    - ``slot`` = ``out_cache_loc[i]`` (caller-pre-translated for SWA groups; entries set to -1 are skipped).
    - ``token / position`` = ``input_ids[i] / positions[i]``.
    - ``real_kv_hash`` = ``real_kv_fold_sources(real_kv_sources, slot)`` if ``real_kv_hash_mode != NONE`` else 0.
    - Store 4 int64s ``(token, position, running_prev_hash, real_kv_hash)`` into ``canary_buf[slot]``.
    - Advance ``running_prev_hash = splitmix64_mix3(prev, token, position)``, where
      splitmix64_mix3 folds each input via ``acc = splitmix64(acc ^ next)`` starting from ``splitmix64(prev)``.
      ``real_kv_hash`` is intentionally not folded into the chain — see ``compute_slot_hash`` in
      ``csrc/kv_canary/canary_common.cuh`` for the radix-folding rationale.

    Initial ``running_prev_hash`` when ``seed_slot_idx >= 0``: load (token, position, prev_hash) from
    ``canary_buf[plan.write_seed_slot_indices[r]]`` and set
    ``running_prev_hash = splitmix64_mix3(seed.prev_hash, seed.token, seed.position)``
    (i.e. apply the same advance step that produced ``seed``'s successor — this keeps slot[0]'s stored
    ``prev_hash`` consistent with the chain link). Else
    ``running_prev_hash = splitmix64(CANARY_CHAIN_ANCHOR)``. ``write_seed_slot_indices`` is already
    SWA-translated by the plan kernel; ``CANARY_CHAIN_ANCHOR`` is hardcoded module-level (no runtime seed).

    Write-time input verification (caller-driven, kernel is oracle-agnostic): when
    ``enable_write_input_assert`` is True the kernel additionally compares ``input_ids[i]`` against
    ``expected_input_tokens[i]`` and ``positions[i]`` against ``expected_input_positions[i]``; mismatch
    on either field records a violation. The chain still advances on the actual values (not the expected
    ones) so a downstream verify won't cascade. Whoever produced the expected tensors is responsible for
    filling them; the kernel runs no oracle internally.

    Write only writes canary_buf (reads only at seed slots). Block uses no shared memory.

    The ForwardBatch-derived arguments are passed through unchanged from the source ForwardBatch — canary does not transform
    them.

    Args:
        context: Shared verify/write launch context, including canary buffer, launch tag, violation sink,
            health counters, and real KV fingerprint sources.
        plan: Pre-allocated WritePlan.
        input_ids: ForwardBatch.input_ids; token ids being written, shape [num_tokens_padded], int64.
            Flattened across reqs in plan.write_offsets order; tail beyond
            plan.write_offsets[plan.write_num_valid_reqs[0]] is cuda-graph padding.
        positions: ForwardBatch.positions; sequence positions of input_ids, shape [num_tokens_padded], int64.
        out_cache_loc: Per-token canary slot index, shape [num_tokens_padded], int64. The caller is
            responsible for translating ForwardBatch.out_cache_loc into the canary's index space for SWA
            groups (typically a host-side LUT gather in the endpoint); FULL groups pass it through
            unchanged. A -1 entry signals skip-this-token (used for SWA out-of-window slots or padding).
            The kernel does not consult any LUT.
        enable_write_input_assert: bool toggle. False = expected_input_* tensors must be None. True = compare
            each chain step's actual (token, position) against the caller-supplied expected tensors below.
        expected_input_tokens: Expected token id per write entry, shape [num_tokens_padded], int64. Only read
            when enable_write_input_assert is True; must be None when enable_write_input_assert is False.
            Layout mirrors input_ids (flattened across reqs in plan.write_offsets order); padding tail
            is ignored. Filled by the caller from whichever oracle produces expected inputs — the kernel
            knows no oracle.
        expected_input_positions: Expected position per write entry, shape [num_tokens_padded], int64, or None.
            Same shape/layout/lifetime rules as expected_input_tokens.

    Implementation:
        - CUDA __global__ `canary_write_kernel`: 1-D grid `(write_req_capacity, 1, 1)` blocks × `(1, 1, 1)` thread
          per block. block_id r = blockIdx.x = one write req; chains are intrinsically serial so a single thread
          per block is optimal (warp-level parallelism would idle 31 lanes).
        - Per block, early-exit on r >= plan.write_num_valid_reqs[0]. Else load entry_start = plan.write_offsets[r],
          entry_count = plan.write_offsets[r+1] - entry_start, seed_slot_idx = plan.write_seed_slot_indices[r] into
          registers.
        - Initialize running_prev_hash: if seed_slot_idx >= 0, load (token, position, prev_hash) from
          canary_buf[seed_slot_idx] and set running_prev_hash = splitmix64_mix3(prev_hash, token, position);
          else running_prev_hash = splitmix64(kCanaryChainAnchor).
        - Serial chain loop `for j in range(entry_count)`:
              i = entry_start + j;
              slot = out_cache_loc[i];  // caller-pre-translated; the kernel never consults a LUT
              if (slot < 0) continue;       // -1 sentinel = skip (SWA out-of-window or padding)
              token = input_ids[i]; position = positions[i];
              real_kv_hash = (real_kv_hash_mode == NONE) ? 0 : real_kv_fold_sources(real_kv_sources, slot);
                  // applies RealKvSource access invariant
              if enable_write_input_assert:
                  if token != expected_input_tokens[i] or position != expected_input_positions[i]:
                      record_violation();  // chain still advances on the ACTUAL (token, position) below
              store (token, position, running_prev_hash, real_kv_hash) to canary_buf[slot] as 4 int64 fields;
              running_prev_hash = splitmix64_mix3(running_prev_hash, token, position);
        - All chain state lives in the block's single thread's registers. No shared memory, no cross-block
          coordination.
        - record_violation() identical to verify (atomicAdd + atomic-write).
        - Counters: thread of block 0 does atomicAdd(kernel_run_counter, 1); each block accumulates its
          entry_count and atomicAdds to slot_run_counter once at exit.

    Calling contract:
        - Pure side-effect; never raises.
        - Input-verification mismatch records violations but does NOT abort the chain.
        - kernel_run_counter is bumped every call.
        - Safe in cuda-graph capture; caller refills input_ids / positions / out_cache_loc / plan
          in-place before replay.

    Pinned by torch reference
    :func:`sglang.jit_kernel.kv_canary.write_ref.launch_canary_write_kernel_torch_reference`; CUDA must match
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
    _assert_contiguous(plan.write_offsets, "plan.write_offsets")
    _assert_contiguous(plan.write_seed_slot_indices, "plan.write_seed_slot_indices")
    _assert_contiguous(plan.write_num_valid_reqs, "plan.write_num_valid_reqs")
    _assert_contiguous(input_ids, "input_ids")
    _assert_contiguous(positions, "positions")
    _assert_contiguous(out_cache_loc, "out_cache_loc")
    if enable_write_input_assert:
        if expected_input_tokens is None or expected_input_positions is None:
            raise ValueError(
                "kv-canary: expected input tensors are required when enable_write_input_assert=True"
            )
        _assert_contiguous(expected_input_tokens, "expected_input_tokens")
        _assert_contiguous(expected_input_positions, "expected_input_positions")
    else:
        if expected_input_tokens is not None or expected_input_positions is not None:
            raise ValueError(
                "kv-canary: expected input tensors must be None when enable_write_input_assert=False"
            )
    _assert_contiguous(context.violation_ring, "violation_ring")
    _assert_contiguous(context.violation_write_index, "violation_write_index")
    _assert_contiguous(context.slot_run_counter, "slot_run_counter")
    _assert_contiguous(context.kernel_run_counter, "kernel_run_counter")
    _assert_contiguous(
        context.enable_chain_position_assert, "enable_chain_position_assert"
    )

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    module = _jit_canary_write_module()
    module.canary_write_step_cuda(
        canary_buf,
        plan.write_offsets,
        plan.write_seed_slot_indices,
        plan.write_num_valid_reqs,
        input_ids,
        positions,
        out_cache_loc,
        int(context.kernel_kind),
        int(enable_write_input_assert),
        expected_input_tokens,
        expected_input_positions,
        context.violation_ring,
        context.violation_write_index,
        context.slot_run_counter,
        context.kernel_run_counter,
        context.enable_chain_position_assert,
        padded_bufs[0],
        padded_bufs[1],
        padded_bufs[2],
        padded_bufs[3],
        source_params,
        len(real_kv_sources),
        int(context.real_kv_hash_mode),
    )


@cache_once
def _jit_canary_write_module() -> Module:
    return load_jit(
        "kv_canary_write",
        cuda_files=["kv_canary/canary_write.cuh"],
        cuda_wrappers=[
            ("canary_write_step_cuda", "canary::canary_write_step_cuda"),
        ],
    )
