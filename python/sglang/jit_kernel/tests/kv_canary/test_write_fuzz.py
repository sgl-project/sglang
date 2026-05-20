"""Random differential fuzz tests: CUDA canary_write_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random as _random

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    RealKvHashMode,
    RealKvSource,
)
from sglang.jit_kernel.kv_canary.write import (
    _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH,
    _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH,
    CanaryPseudoMode,
)
from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_and_assert_write_buf_and_state_equal as _run_both_and_assert_buf_and_state_equal,
)
from sglang.jit_kernel.tests.kv_canary._differential import _run_both_write as _run_both
from sglang.jit_kernel.tests.kv_canary._fixtures import _dummy_pseudo_tensors
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_multiset_equal,
    make_canary_buf,
    make_real_kv_source,
    make_write_plan,
    to_signed_int64,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _build_random_write_sources(
    rng: _random.Random,
    *,
    num_sources: int,
    num_slots: int,
) -> tuple[tuple[RealKvSource, ...], tuple[RealKvSource, ...]]:
    sources_cuda: list[RealKvSource] = []
    sources_ref: list[RealKvSource] = []
    for _ in range(num_sources):
        rb = rng.randint(1, 16)
        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=rb,
            page_size=1,
            read_bytes=rb,
            device=_DEVICE,
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


def test_write_pure_random_fuzz_byte_equal() -> None:
    rng = _random.Random(0)
    for iteration in range(100):
        seed_snapshot = rng.getstate()
        num_slots = 200
        bs = rng.randint(1, 8)
        entries_per_req = [rng.randint(1, 16) for _ in range(bs)]
        total_entries = sum(entries_per_req)

        write_offsets = [0]
        for count in entries_per_req:
            write_offsets.append(write_offsets[-1] + count)

        # Partition slots into write targets and seed candidates so the seed reads can't race a
        # concurrent block's write to the same slot. Slot 0 is excluded — it's the reserved padding
        # sentinel for the verify kernel and used as the kv-pool padding sink by sglang.
        slot_pool = list(range(1, num_slots))
        rng.shuffle(slot_pool)
        disjoint_slots = slot_pool[:total_entries]
        seed_candidate_pool = slot_pool[total_entries:]

        seed_slot_indices: list[int] = []
        for _ in range(bs):
            if rng.random() < 0.5 or not seed_candidate_pool:
                seed_slot_indices.append(-1)
            else:
                seed_slot_indices.append(rng.choice(seed_candidate_pool))

        plan_cuda = make_write_plan(
            write_offsets=write_offsets,
            seed_slot_indices=seed_slot_indices,
            num_valid_reqs=bs,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=write_offsets,
            seed_slot_indices=seed_slot_indices,
            num_valid_reqs=bs,
            device=_DEVICE,
        )

        fb_input_ids_list = [rng.randint(0, 0xFFFF) for _ in range(total_entries)]
        fb_positions_list = [rng.randint(0, 0x7FFFFFFF) for _ in range(total_entries)]
        loc_choice = rng.random()
        if loc_choice < 0.5:
            fb_out_cache_loc_list = list(disjoint_slots)
        elif loc_choice < 0.75:
            fb_out_cache_loc_list = [
                -1 if rng.random() < 0.3 else s for s in disjoint_slots
            ]
        else:
            fb_out_cache_loc_list = [-1] * total_entries

        pseudo_mode = rng.choice([CanaryPseudoMode.OFF, CanaryPseudoMode.ON])
        if pseudo_mode == CanaryPseudoMode.ON:
            pseudo_tokens_list = [
                t if rng.random() < 0.5 else rng.randint(0, 0xFFFF)
                for t in fb_input_ids_list
            ]
            pseudo_positions_list = [
                p if rng.random() < 0.5 else rng.randint(0, 0x7FFFFFFF)
                for p in fb_positions_list
            ]
        else:
            pseudo_tokens_list = [0] * total_entries
            pseudo_positions_list = [0] * total_entries

        num_sources = rng.randint(0, 4)
        sources_cuda, sources_ref = _build_random_write_sources(
            rng, num_sources=num_sources, num_slots=num_slots
        )
        mode = rng.choice([RealKvHashMode.OFF, RealKvHashMode.BIT, RealKvHashMode.ALL])

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        fb_input_ids = torch.tensor(
            fb_input_ids_list, dtype=torch.int32, device=_DEVICE
        )
        fb_positions_t = torch.tensor(
            fb_positions_list, dtype=torch.int32, device=_DEVICE
        )
        fb_out_cache_loc = torch.tensor(
            fb_out_cache_loc_list, dtype=torch.int32, device=_DEVICE
        )
        pseudo_tokens = torch.tensor(
            pseudo_tokens_list, dtype=torch.int32, device=_DEVICE
        )
        pseudo_positions = torch.tensor(
            pseudo_positions_list, dtype=torch.int32, device=_DEVICE
        )
        cuda_log = FakeViolationLog.allocate(capacity=256, device=_DEVICE)
        ref_log = FakeViolationLog.allocate(capacity=256, device=_DEVICE)

        try:
            # Multiset compare: write kernel launches one block per req and atomicAdd-orders violations,
            # so ring rows are permuted vs the sequential ref. Canary buf and counters must still match.
            _run_both(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions_t,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=pseudo_mode,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=sources_cuda,
                real_kv_sources_ref=sources_ref,
                real_kv_hash_mode=mode,
            )
            assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
            assert_canary_state_multiset_equal(log_a=cuda_log, log_b=ref_log)
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"bs={bs} total_entries={total_entries} pseudo_mode={pseudo_mode} mode={mode}: {e}"
            ) from e


def test_write_random_clean_chain_from_head() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(5, 50)
        num_slots = chain_len + 10
        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(chain_len)]
        positions_list = list(range(chain_len))
        slot_list = list(range(chain_len))

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        plan_cuda = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(chain_len)
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.OFF,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len}: {e}"
            ) from e


def test_write_random_seed_slot_resume() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        chain_len = rng.randint(2, 20)
        num_slots = chain_len + 20
        seed_slot = rng.randint(chain_len, num_slots - 1)

        seed_token = rng.randint(1, 0xFFFF)
        seed_position = rng.randint(0, 50)
        seed_prev_signed = to_signed_int64(rng.randint(0, (1 << 64) - 1))

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=seed_slot,
                token=seed_token,
                position=seed_position,
                prev_hash=seed_prev_signed,
                real_kv_hash=0,
            )

        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(chain_len)]
        positions_list = [seed_position + 1 + i for i in range(chain_len)]
        slot_list = list(range(chain_len))

        plan_cuda = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[seed_slot],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, chain_len],
            seed_slot_indices=[seed_slot],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(chain_len)
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.OFF,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"chain_len={chain_len} seed_slot={seed_slot}: {e}"
            ) from e


def test_write_random_pseudo_match_no_violation() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        n = rng.randint(1, 20)
        tokens_list = [rng.randint(0, 0xFFFF) for _ in range(n)]
        positions_list = list(range(n))
        slot_list = list(range(n))

        cuda_buf = make_canary_buf(
            num_slots=n + 4, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        plan_cuda = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens = fb_input_ids.clone()
        pseudo_positions = fb_positions.clone()
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.ON,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            assert (
                int(cuda_log.write_index[0].item()) == 0
            ), "expected no violation when pseudo expected == actual"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} n={n}: {e}"
            ) from e


def test_write_random_pseudo_mismatch_reports_correct_bit() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        n = rng.randint(2, 20)
        tokens_list = [rng.randint(0, 0xFFFF) for _ in range(n)]
        positions_list = list(range(n))
        slot_list = list(range(n))
        corrupt_idx = rng.randint(0, n - 1)
        mismatch_type = rng.choice(["token", "position"])

        pseudo_tokens_list = list(tokens_list)
        pseudo_positions_list = list(positions_list)
        if mismatch_type == "token":
            pseudo_tokens_list[corrupt_idx] = (
                tokens_list[corrupt_idx] + rng.randint(1, 999)
            ) & 0xFFFF
            expected_bit = _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH
        else:
            pseudo_positions_list[corrupt_idx] = positions_list[
                corrupt_idx
            ] + rng.randint(1, 99)
            expected_bit = _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH

        cuda_buf = make_canary_buf(
            num_slots=n + 4, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        plan_cuda = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens = torch.tensor(
            pseudo_tokens_list, dtype=torch.int32, device=_DEVICE
        )
        pseudo_positions = torch.tensor(
            pseudo_positions_list, dtype=torch.int32, device=_DEVICE
        )
        cuda_log = FakeViolationLog.allocate(capacity=64, device=_DEVICE)
        ref_log = FakeViolationLog.allocate(capacity=64, device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.ON,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            write_idx = int(cuda_log.write_index[0].item())
            assert write_idx >= 1, "expected at least one violation"
            corrupt_slot = slot_list[corrupt_idx]
            found = any(
                (
                    int(cuda_log.ring[r, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
                    & expected_bit
                )
                and int(cuda_log.ring[r, 1].item()) == corrupt_slot
                for r in range(min(write_idx, cuda_log.ring.shape[0]))
            )
            assert (
                found
            ), f"expected bit {expected_bit:#x} at slot {corrupt_slot} not found"
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"n={n} corrupt_idx={corrupt_idx} mismatch_type={mismatch_type}: {e}"
            ) from e


def test_write_random_negative_slot_skip_no_op() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        n = rng.randint(2, 20)
        num_slots = n + 10
        skip_indices = set(rng.sample(range(n), rng.randint(1, max(1, n // 2))))
        slot_list = [
            -1 if i in skip_indices else rng.randint(0, num_slots - 1) for i in range(n)
        ]
        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(n)]
        positions_list = list(range(n))

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        plan_cuda = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(n)
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.OFF,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            expected_counter = n - len(skip_indices)
            actual_counter = int(cuda_log.slot_run_counter[0].item())
            assert actual_counter == expected_counter, (
                f"slot_run_counter={actual_counter} expected={expected_counter} "
                f"(n={n} skipped={len(skip_indices)})"
            )
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"n={n} skip_indices={sorted(skip_indices)}: {e}"
            ) from e


def test_write_random_real_kv_modes_byte_equal() -> None:
    rng = _random.Random(0)
    for iteration in range(25):
        seed_snapshot = rng.getstate()
        n = rng.randint(1, 15)
        num_slots = n + 10
        mode = rng.choice([RealKvHashMode.OFF, RealKvHashMode.BIT, RealKvHashMode.ALL])
        num_sources = rng.randint(1, 3)
        sources_cuda, sources_ref = _build_random_write_sources(
            rng, num_sources=num_sources, num_slots=num_slots
        )
        tokens_list = [rng.randint(1, 0xFFFF) for _ in range(n)]
        positions_list = list(range(n))
        slot_list = list(range(n))

        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        plan_cuda = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        plan_ref = make_write_plan(
            write_offsets=[0, n],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        fb_input_ids = torch.tensor(tokens_list, dtype=torch.int32, device=_DEVICE)
        fb_positions = torch.tensor(positions_list, dtype=torch.int32, device=_DEVICE)
        fb_out_cache_loc = torch.tensor(slot_list, dtype=torch.int32, device=_DEVICE)
        pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(n)
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        try:
            _run_both_and_assert_buf_and_state_equal(
                cuda_canary_buf=cuda_buf,
                ref_canary_buf=ref_buf,
                plan_cuda=plan_cuda,
                plan_ref=plan_ref,
                fb_input_ids=fb_input_ids,
                fb_positions=fb_positions,
                fb_out_cache_loc=fb_out_cache_loc,
                pseudo_mode=CanaryPseudoMode.OFF,
                pseudo_expected_tokens=pseudo_tokens,
                pseudo_expected_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=sources_cuda,
                real_kv_sources_ref=sources_ref,
                real_kv_hash_mode=mode,
            )
        except AssertionError as e:
            raise AssertionError(
                f"iteration={iteration} rng_seed_state_first_int={seed_snapshot[1][0]} "
                f"n={n} mode={mode} num_sources={num_sources}: {e}"
            ) from e
