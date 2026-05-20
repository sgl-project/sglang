"""Random differential fuzz tests: CUDA canary_verify_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random as _random

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _FAIL_REASON_BIT_CHAIN_HASH,
    _FAIL_REASON_BIT_POSITION,
    _FAIL_REASON_BIT_REAL_KV_HASH,
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    CANARY_CHAIN_ANCHOR,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
)
from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_and_assert_verify_state_equal as _run_both_and_assert_state_equal,
)
from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_verify as _run_both,
)
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_state_multiset_equal,
    chain_anchor_signed,
    make_canary_buf,
    make_real_kv_source,
    make_verify_plan,
    make_write_plan,
    read_slot_fields,
    splitmix64,
    splitmix64_mix4,
    to_signed_int64,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _build_random_verify_sources(
    rng: _random.Random,
    *,
    num_sources: int,
    num_slots: int,
    read_bytes_max: int = 16,
) -> tuple[tuple[RealKvSource, ...], tuple[RealKvSource, ...]]:
    sources_cuda: list[RealKvSource] = []
    sources_ref: list[RealKvSource] = []
    for _ in range(num_sources):
        rb = rng.randint(1, read_bytes_max)
        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=rb,
            page_size=1,
            read_bytes=rb,
            device=_DEVICE,
            fill=rng.randint(0, 255),
        )
        src.tensor.random_()
        ref_src = RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        )
        sources_cuda.append(src)
        sources_ref.append(ref_src)
    return tuple(sources_cuda), tuple(sources_ref)


def _stamp_random_chain(
    rng: _random.Random,
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    num_slots: int,
    chain_len: int,
) -> tuple[list[int], list[int], list[int]]:
    # Skip slot 0 — the verify kernel unconditionally treats it as the reserved padding sentinel.
    slot_pool = list(range(1, num_slots))
    rng.shuffle(slot_pool)
    slots = slot_pool[:chain_len]

    tokens: list[int] = [rng.randint(0, 0xFFFFFFFF) for _ in range(chain_len)]
    positions: list[int] = list(range(chain_len))
    # Stamp real_kv_hash=0 so the chain stays consistent with RealKvHashMode.OFF callers (expected=0).
    real_kv_hashes: list[int] = [0] * chain_len

    running = splitmix64(CANARY_CHAIN_ANCHOR)
    for slot_idx, token, position, rkv in zip(slots, tokens, positions, real_kv_hashes):
        signed_prev = to_signed_int64(running)
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=signed_prev,
                real_kv_hash=to_signed_int64(rkv),
            )
        running = splitmix64_mix4(running, token, position, rkv)

    prev_slots = [-1] + slots[:-1]
    return slots, positions, prev_slots


def test_verify_pure_random_fuzz_byte_equal() -> None:
    rng = _random.Random(0)
    for iteration in range(100):
        seed_snapshot = rng.getstate()
        num_slots = 200
        bs = rng.randint(1, 8)
        entries_per_req = [rng.randint(1, 16) for _ in range(bs)]
        while sum(entries_per_req) > 100:
            entries_per_req = [rng.randint(1, 8) for _ in range(bs)]
        total_entries = sum(entries_per_req)

        # Skip slot 0 — the verify kernel unconditionally treats it as the reserved padding sentinel.
        slot_pool = list(range(1, num_slots))
        rng.shuffle(slot_pool)
        all_slots = slot_pool[:total_entries]

        slot_indices: list[int] = []
        positions: list[int] = []
        prev_slot_indices: list[int] = []
        used_slots: list[int] = []
        offset = 0
        for req_idx in range(bs):
            n = entries_per_req[req_idx]
            req_slots = all_slots[offset : offset + n]
            req_positions = sorted(rng.sample(range(1024), n))
            for i, (slot_idx, pos) in enumerate(zip(req_slots, req_positions)):
                slot_indices.append(slot_idx)
                positions.append(pos)
                if used_slots and rng.random() < 0.5:
                    prev_slot_indices.append(rng.choice(used_slots))
                else:
                    prev_slot_indices.append(-1)
                used_slots.append(slot_idx)
            offset += n

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        for slot_idx in slot_indices:
            token = rng.randint(0, 0xFFFFFFFF)
            position = rng.randint(0, 0x7FFFFFFF)
            prev_hash = rng.randint(0, (1 << 64) - 1)
            real_kv_hash = rng.randint(0, (1 << 64) - 1)
            for buf in (cuda_buf, ref_buf):
                write_slot_fields(
                    canary_buf=buf,
                    slot_idx=slot_idx,
                    token=token,
                    position=position,
                    prev_hash=to_signed_int64(prev_hash),
                    real_kv_hash=to_signed_int64(real_kv_hash),
                )

        num_sources = rng.randint(0, 4)
        sources_cuda, sources_ref = _build_random_verify_sources(
            rng, num_sources=num_sources, num_slots=num_slots
        )
        mode = rng.choice([RealKvHashMode.OFF, RealKvHashMode.BIT, RealKvHashMode.ALL])
        tag = rng.choice(list(CanaryLaunchTag))

        plan_cuda = make_verify_plan(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=prev_slot_indices,
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=prev_slot_indices,
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(capacity=256, device=_DEVICE)
        ref_log = FakeViolationLog.allocate(capacity=256, device=_DEVICE)

        try:
            # Use the multiset variant: CUDA atomically assigns ring slots so the row order is
            # non-deterministic across warps, but the multiset of recorded violations must match ref.
            _run_both(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=sources_cuda,
                real_kv_sources_ref=sources_ref,
                real_kv_hash_mode=mode,
                kernel_kind=tag,
            )
            assert_canary_state_multiset_equal(log_a=cuda_log, log_b=ref_log)
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"bs={bs} total_entries={total_entries} mode={mode} tag={tag}: {e}"
            ) from e


def test_verify_random_clean_chain_no_violation() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(5, 50)
        num_slots = chain_len + 10
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        slots, positions, prev_slots = _stamp_random_chain(
            rng,
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            num_slots=num_slots,
            chain_len=chain_len,
        )
        plan_cuda = make_verify_plan(
            slot_indices=slots,
            positions=positions,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=slots,
            positions=positions,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            assert (
                int(cuda_log.write_index[0].item()) == 0
            ), f"expected no violation, got write_index={int(cuda_log.write_index[0].item())}"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len}: {e}"
            ) from e


def test_verify_random_token_corruption_reports_chain_bit() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(3, 20)
        num_slots = chain_len + 10
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        slots, positions, prev_slots = _stamp_random_chain(
            rng,
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            num_slots=num_slots,
            chain_len=chain_len,
        )
        corrupt_idx = rng.randint(0, chain_len - 2)
        corrupt_slot = slots[corrupt_idx]
        stored_token, stored_pos, stored_prev, stored_rkv = read_slot_fields(
            canary_buf=cuda_buf, slot_idx=corrupt_slot
        )
        new_token = (stored_token + rng.randint(1, 1000)) & 0xFFFFFFFF
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=corrupt_slot,
                token=new_token,
                position=stored_pos,
                prev_hash=stored_prev,
                real_kv_hash=stored_rkv,
            )
        downstream_idx = corrupt_idx + 1
        downstream_slot = slots[downstream_idx]
        plan_cuda = make_verify_plan(
            slot_indices=[downstream_slot],
            positions=[positions[downstream_idx]],
            prev_slot_indices=[corrupt_slot],
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=[downstream_slot],
            positions=[positions[downstream_idx]],
            prev_slot_indices=[corrupt_slot],
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert (
                write_idx >= 1
            ), "expected at least one violation after token corruption"
            found = any(
                int(cuda_log.ring[r, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
                & _FAIL_REASON_BIT_CHAIN_HASH
                for r in range(min(write_idx, cuda_log.ring.shape[0]))
            )
            assert found, "CHAIN_HASH bit not found after token corruption"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len} corrupt_idx={corrupt_idx}: {e}"
            ) from e


def test_verify_random_position_corruption_reports_position_bit() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(2, 20)
        num_slots = chain_len + 10
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        slots, positions, prev_slots = _stamp_random_chain(
            rng,
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            num_slots=num_slots,
            chain_len=chain_len,
        )
        corrupt_idx = rng.randint(0, chain_len - 1)
        corrupt_slot = slots[corrupt_idx]
        stored_token, stored_pos, stored_prev, stored_rkv = read_slot_fields(
            canary_buf=cuda_buf, slot_idx=corrupt_slot
        )
        new_pos = stored_pos + rng.randint(1, 100)
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=corrupt_slot,
                token=stored_token,
                position=new_pos,
                prev_hash=stored_prev,
                real_kv_hash=stored_rkv,
            )
        plan_cuda = make_verify_plan(
            slot_indices=[corrupt_slot],
            positions=[positions[corrupt_idx]],
            prev_slot_indices=[prev_slots[corrupt_idx]],
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=[corrupt_slot],
            positions=[positions[corrupt_idx]],
            prev_slot_indices=[prev_slots[corrupt_idx]],
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert (
                write_idx >= 1
            ), "expected at least one violation after position corruption"
            found = any(
                int(cuda_log.ring[r, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
                & _FAIL_REASON_BIT_POSITION
                for r in range(min(write_idx, cuda_log.ring.shape[0]))
            )
            assert found, "POSITION bit not found after position corruption"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len} corrupt_idx={corrupt_idx} new_pos={new_pos}: {e}"
            ) from e


def test_verify_random_prev_hash_corruption_reports_chain_bit() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(2, 20)
        num_slots = chain_len + 10
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        slots, positions, prev_slots = _stamp_random_chain(
            rng,
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            num_slots=num_slots,
            chain_len=chain_len,
        )
        corrupt_idx = rng.randint(0, chain_len - 1)
        corrupt_slot = slots[corrupt_idx]
        stored_token, stored_pos, stored_prev, stored_rkv = read_slot_fields(
            canary_buf=cuda_buf, slot_idx=corrupt_slot
        )
        new_prev = to_signed_int64(
            (stored_prev ^ rng.randint(1, (1 << 63))) & ((1 << 64) - 1)
        )
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=corrupt_slot,
                token=stored_token,
                position=stored_pos,
                prev_hash=new_prev,
                real_kv_hash=stored_rkv,
            )
        plan_cuda = make_verify_plan(
            slot_indices=[corrupt_slot],
            positions=[positions[corrupt_idx]],
            prev_slot_indices=[prev_slots[corrupt_idx]],
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=[corrupt_slot],
            positions=[positions[corrupt_idx]],
            prev_slot_indices=[prev_slots[corrupt_idx]],
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert (
                write_idx >= 1
            ), "expected at least one violation after prev_hash corruption"
            found = any(
                int(cuda_log.ring[r, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
                & _FAIL_REASON_BIT_CHAIN_HASH
                for r in range(min(write_idx, cuda_log.ring.shape[0]))
            )
            assert found, "CHAIN_HASH bit not found after prev_hash corruption"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len} corrupt_idx={corrupt_idx}: {e}"
            ) from e


def test_verify_random_real_kv_corruption_reports_real_kv_bit() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(2, 15)
        num_slots = chain_len + 10
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()

        read_bytes = rng.randint(1, 8)
        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=read_bytes,
            page_size=1,
            read_bytes=read_bytes,
            device=_DEVICE,
        )
        src.tensor.random_()
        mode = rng.choice([RealKvHashMode.BIT, RealKvHashMode.ALL])

        from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode
        from sglang.jit_kernel.kv_canary.write_ref import (
            canary_write_step_torch_reference,
        )

        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(chain_len)]
        positions_list = list(range(chain_len))
        slot_pool = list(range(num_slots))
        rng.shuffle(slot_pool)
        slot_list = slot_pool[:chain_len]

        write_plan = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens = torch.zeros(chain_len, dtype=torch.int32, device=_DEVICE)
        pseudo_positions = torch.zeros(chain_len, dtype=torch.int32, device=_DEVICE)
        write_log = FakeViolationLog.allocate(device=_DEVICE)
        canary_write_step_torch_reference(
            canary_buf=cuda_buf,
            plan=write_plan,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            pseudo_mode=CanaryPseudoMode.OFF,
            pseudo_expected_tokens=pseudo_tokens,
            pseudo_expected_positions=pseudo_positions,
            violation_ring=write_log.ring,
            violation_write_index=write_log.write_index,
            slot_run_counter=write_log.slot_run_counter,
            kernel_run_counter=write_log.kernel_run_counter,
            real_kv_sources=(src,),
            real_kv_hash_mode=mode,
        )
        ref_buf.copy_(cuda_buf)

        corrupt_idx = rng.randint(0, chain_len - 1)
        corrupt_slot = slot_list[corrupt_idx]
        src.tensor[corrupt_slot, 0] ^= 0xFF
        src_ref = RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        )

        prev_slots = [-1] + slot_list[:-1]
        plan_cuda = make_verify_plan(
            slot_indices=slot_list,
            positions=positions_list,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=slot_list,
            positions=positions_list,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(capacity=64, device=_DEVICE)
        ref_log = FakeViolationLog.allocate(capacity=64, device=_DEVICE)

        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(src,),
                real_kv_sources_ref=(src_ref,),
                real_kv_hash_mode=mode,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert (
                write_idx >= 1
            ), "expected at least one violation after real_kv corruption"
            found = any(
                int(cuda_log.ring[r, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
                & _FAIL_REASON_BIT_REAL_KV_HASH
                for r in range(min(write_idx, cuda_log.ring.shape[0]))
            )
            assert found, f"REAL_KV_HASH bit not found after corruption (mode={mode})"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len} corrupt_idx={corrupt_idx} mode={mode}: {e}"
            ) from e


def test_verify_random_ring_overflow_counter_consistent() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        ring_capacity = rng.randint(2, 8)
        n_violations = ring_capacity + rng.randint(1, ring_capacity + 2)
        num_slots = n_violations + 10
        anchor_signed = chain_anchor_signed()
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        # Skip slot 0 — the verify kernel unconditionally treats it as the reserved padding sentinel.
        slot_indices = list(range(1, n_violations + 1))
        for slot_idx in slot_indices:
            # Sample once per slot so cuda_buf and ref_buf receive identical bytes.
            slot_token = rng.randint(1, 0xFFFF)
            for buf in (cuda_buf, ref_buf):
                write_slot_fields(
                    canary_buf=buf,
                    slot_idx=slot_idx,
                    token=slot_token,
                    position=0,
                    prev_hash=anchor_signed,
                    real_kv_hash=0,
                )
        wrong_positions = [rng.randint(1, 1000) for _ in range(n_violations)]
        plan_cuda = make_verify_plan(
            slot_indices=slot_indices,
            positions=wrong_positions,
            prev_slot_indices=[-1] * n_violations,
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=slot_indices,
            positions=wrong_positions,
            prev_slot_indices=[-1] * n_violations,
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
        ref_log = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert (
                write_idx == n_violations
            ), f"expected write_index=={n_violations} got {write_idx}"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"ring_capacity={ring_capacity} n_violations={n_violations}: {e}"
            ) from e


def test_verify_random_page_size_gt_1_layout() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        page_size = rng.choice([2, 4, 8])
        chain_len = rng.randint(2, 12)
        num_slots = chain_len * page_size + page_size
        num_bytes_per_token = rng.choice([4, 8])

        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=num_bytes_per_token,
            page_size=page_size,
            read_bytes=num_bytes_per_token,
            device=_DEVICE,
        )
        src.tensor.random_()
        mode = rng.choice([RealKvHashMode.BIT, RealKvHashMode.ALL])

        from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode
        from sglang.jit_kernel.kv_canary.write_ref import (
            canary_write_step_torch_reference,
        )

        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(chain_len)]
        positions_list = list(range(chain_len))
        slot_list = [i * page_size for i in range(chain_len)]

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()

        write_plan = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens = torch.zeros(chain_len, dtype=torch.int32, device=_DEVICE)
        pseudo_positions = torch.zeros(chain_len, dtype=torch.int32, device=_DEVICE)
        write_log = FakeViolationLog.allocate(device=_DEVICE)
        canary_write_step_torch_reference(
            canary_buf=cuda_buf,
            plan=write_plan,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            pseudo_mode=CanaryPseudoMode.OFF,
            pseudo_expected_tokens=pseudo_tokens,
            pseudo_expected_positions=pseudo_positions,
            violation_ring=write_log.ring,
            violation_write_index=write_log.write_index,
            slot_run_counter=write_log.slot_run_counter,
            kernel_run_counter=write_log.kernel_run_counter,
            real_kv_sources=(src,),
            real_kv_hash_mode=mode,
        )
        ref_buf.copy_(cuda_buf)

        src_ref = RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        )
        prev_slots = [-1] + slot_list[:-1]
        plan_cuda = make_verify_plan(
            slot_indices=slot_list,
            positions=positions_list,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=slot_list,
            positions=positions_list,
            prev_slot_indices=prev_slots,
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(src,),
                real_kv_sources_ref=(src_ref,),
                real_kv_hash_mode=mode,
            )
            assert (
                int(cuda_log.write_index[0].item()) == 0
            ), "expected no violation on clean chain with page_size>1"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"page_size={page_size} chain_len={chain_len} mode={mode}: {e}"
            ) from e
