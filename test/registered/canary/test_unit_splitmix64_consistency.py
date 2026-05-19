"""Bit-wise consistency between Python and CUDA ``splitmix64_mix``.

The canary kernel computes the chain hash on device; the Python reference
in ``jit_kernel.kv_cache_canary_ref`` lives only to document the
algorithm and to back this consistency check. Any drift between the two
silently invalidates the canary's chain detection. We launch a single
canary write kernel with 1000 chained entries — each iteration mixes the
prior ``prev_hash`` with a random ``(token, position)`` pair — then read
the stored ``prev_hash`` field per slot and require byte-equality with
the Python recomputation.
"""

from __future__ import annotations

import random

import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_SLOT_BYTES,
    KERNEL_KIND_HEAD,
    VIOLATION_FIELDS,
    canary_step,
    to_signed_int64,
)
from sglang.jit_kernel.kv_cache_canary_ref import splitmix64_mix
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-large")

_SEED = 0xC0FFEE1234567890
_U64_MASK = (1 << 64) - 1


def _alloc_state(ring_capacity: int = 32) -> dict:
    return dict(
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device="cuda"
        ),
        violation_ring_valid=torch.zeros(
            ring_capacity, dtype=torch.int32, device="cuda"
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device="cuda"),
        first_violation=torch.zeros(VIOLATION_FIELDS, dtype=torch.int64, device="cuda"),
        first_violation_set=torch.zeros(1, dtype=torch.int32, device="cuda"),
        is_errored=torch.zeros(1, dtype=torch.int32, device="cuda"),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def test_python_and_cuda_splitmix64_chains_match_bitwise():
    rng = random.Random(20260518)
    n = 1000
    slot_stride = (
        CANARY_SLOT_BYTES * 2
    )  # over-aligned, kernel only touches the first 32

    tokens = [rng.randrange(1, 1 << 31) for _ in range(n)]
    positions = list(range(n))
    slot_indices = list(range(n))

    # Drive one write-req chain of length n through the kernel.
    buf = torch.zeros(n, slot_stride, dtype=torch.uint8, device="cuda")
    state = _alloc_state()

    canary_step(
        src_buf=buf.flatten(),
        dst_buf=buf.flatten(),
        slot_stride_bytes=slot_stride,
        verify_slot_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        verify_positions=torch.zeros(1, dtype=torch.int64, device="cuda"),
        verify_prev_slot_indices=torch.full((1,), -1, dtype=torch.int64, device="cuda"),
        verify_active_mask=torch.zeros(1, dtype=torch.int32, device="cuda"),
        write_slot_indices=torch.tensor(slot_indices, dtype=torch.int64, device="cuda"),
        write_token_ids=torch.tensor(tokens, dtype=torch.int64, device="cuda"),
        write_positions=torch.tensor(positions, dtype=torch.int64, device="cuda"),
        write_req_seed_slot_indices=torch.full(
            (1,), -1, dtype=torch.int64, device="cuda"
        ),
        write_req_entry_starts=torch.zeros(1, dtype=torch.int64, device="cuda"),
        write_req_entry_counts=torch.full((1,), n, dtype=torch.int64, device="cuda"),
        write_req_active_mask=torch.ones(1, dtype=torch.int32, device="cuda"),
        expected_write_token_ids=torch.full((n,), -1, dtype=torch.int64, device="cuda"),
        expected_write_positions=torch.full((n,), -1, dtype=torch.int64, device="cuda"),
        seed=_SEED,
        violation_ring=state["violation_ring"],
        violation_ring_valid=state["violation_ring_valid"],
        violation_write_index=state["violation_write_index"],
        first_violation=state["first_violation"],
        first_violation_set=state["first_violation_set"],
        is_errored=state["is_errored"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        kernel_kind=KERNEL_KIND_HEAD,
        real_kv_buf=torch.zeros(1, dtype=torch.uint8, device="cuda"),
        real_kv_slot_stride_bytes=0,
        real_kv_read_bytes=0,
        real_kv_hash_mode=0,
    )
    torch.cuda.synchronize()

    # Read prev_hash field (offset 16 bytes = field index 2) from each slot.
    stored = buf.view(torch.int64).view(n, slot_stride // 8)[:, 2].cpu().tolist()

    # Recompute the chain in Python.
    expected_prev = _SEED
    for i in range(n):
        assert (
            stored[i] & _U64_MASK == expected_prev
        ), f"chain mismatch at slot {i}: cuda={stored[i] & _U64_MASK:#x} python={expected_prev:#x}"
        # Cross-check via the helper used inside test_kv_cache_canary.py.
        assert stored[i] == to_signed_int64(expected_prev)
        expected_prev = splitmix64_mix(expected_prev, tokens[i], positions[i])

    # Counters reflect work done.
    assert int(state["slot_run_counter"].item()) == n
    assert int(state["is_errored"].item()) == 0


def test_python_splitmix64_mix_zero_inputs_are_zero():
    """splitmix64(0) == 0 (defining property of the algorithm)."""
    assert splitmix64_mix(0, 0, 0) == 0


def test_python_splitmix64_mix_distinct_inputs_give_distinct_outputs():
    """Random spot-check: 100 random triples must all hash to distinct values."""
    rng = random.Random(20260518)
    seen: set[int] = set()
    for _ in range(100):
        prev = rng.randrange(0, _U64_MASK + 1)
        tok = rng.randrange(0, 1 << 32)
        pos = rng.randrange(0, 1 << 32)
        h = splitmix64_mix(prev, tok, pos)
        assert 0 <= h <= _U64_MASK
        seen.add(h)
    # 100 random triples and 64-bit output -> collision probability is
    # vanishingly low; require strictly distinct results.
    assert len(seen) == 100
