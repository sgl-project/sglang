from __future__ import annotations

import random

import pytest
import torch

from sglang.jit_kernel.kv_canary.plan import launch_canary_plan_kernels
from sglang.jit_kernel.kv_canary.plan_ref import (
    launch_canary_plan_kernels_torch_reference,
)
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.jit_kernel.tests.kv_canary._differential import run_plan_diff
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    allocate_plan_pair,
    derive_plan_capacity,
    empty_extras,
    make_lut,
    make_req_to_token,
)
from sglang.jit_kernel.tests.kv_canary._invariants import PlanInvariants
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")


def _tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=_DEVICE)


def _plan_pair(
    *, verify_capacity: int, write_req_capacity: int
) -> tuple[tuple[VerifyPlan, WritePlan], tuple[VerifyPlan, WritePlan]]:
    triton_v, triton_w, ref_v, ref_w = allocate_plan_pair(
        verify_capacity=verify_capacity, write_req_capacity=write_req_capacity
    )
    return (triton_v, triton_w), (ref_v, ref_w)


def _alloc_for_inputs(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extras_count: int,
    swa_window_size: int,
) -> tuple[int, int]:
    bs = int(req_pool_indices.shape[0])
    rpi_cpu = req_pool_indices.detach().cpu().tolist()
    pfx_cpu = prefix_lens.detach().cpu().tolist()
    ext_cpu = extend_seq_lens.detach().cpu().tolist()
    total_verify = 0
    for rpi, pfx in zip(rpi_cpu, pfx_cpu):
        if rpi == 0:
            continue
        if swa_window_size > 0:
            window_start = max(0, pfx - swa_window_size)
            total_verify += max(0, pfx - window_start)
        else:
            total_verify += max(0, pfx)
    return derive_plan_capacity(
        kind="loose", total_verify=total_verify, extras_count=extras_count, bs=bs
    )


def _run_label(
    *,
    label: str,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: torch.Tensor | None,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan]:
    _ = extras
    verify_plan = VerifyPlan.allocate(
        verify_capacity=verify_capacity, device=_DEVICE
    ).zero_for_testing_()
    write_plan = WritePlan.allocate(
        write_req_capacity=write_req_capacity, device=_DEVICE
    ).zero_for_testing_()
    runner = (
        launch_canary_plan_kernels
        if label == "real"
        else launch_canary_plan_kernels_torch_reference
    )
    runner(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=verify_capacity,
        req_to_verify_expected_tokens=None,
        req_to_verify_expected_tokens_valid_lens=None,
        kv_token_id_vs_position_offset=0,
    )
    torch.cuda.synchronize()
    return verify_plan, write_plan


class TestBasicShape:
    def test_single_req_extend_basic(self) -> None:
        """bs=1, prefix=0, extend=5 → verify entries empty; write_offsets[0:2] = [0, 5]; seed = -1."""
        # Step 1: build a one-req batch with no prefix and 5 extend tokens.
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([3]),
            prefix_lens=_tensor([0]),
            extend_seq_lens=_tensor([5]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # Step 2: prefix=0 → no verify entries; seed = -1 because prefix==0.
        assert int(triton_v.verify_num_valid[0].item()) == 0
        assert int(triton_w.write_num_valid_reqs[0].item()) == 1
        assert int(triton_w.write_offsets[0].item()) == 0
        assert int(triton_w.write_offsets[1].item()) == 5
        assert int(triton_w.write_seed_slot_indices[0].item()) == -1

    def test_single_req_decode(self) -> None:
        """extend=1, prefix=K → write_seed_slot = req_to_token[rp, K-1]; verify covers all K prefix tokens."""
        max_seq_len = 16
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([2]),
            prefix_lens=_tensor([7]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # Step: verify covers positions [0..7); seed slot for req rp=2 is at position 6 = rp * max_seq_len + 6.
        assert int(triton_v.verify_num_valid[0].item()) == 7
        assert int(triton_w.write_seed_slot_indices[0].item()) == 2 * max_seq_len + 6

    def test_multi_req_mixed_extend_decode(self) -> None:
        """bs=3 mixed extend/decode → write_offsets cumsum is byte-equal across Triton + ref."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=8)
        # req0: prefill extend=8; req1: decode extend=1; req2: decode extend=1.
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 3]),
            prefix_lens=_tensor([0, 4, 10]),
            extend_seq_lens=_tensor([8, 1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # Step: write_offsets exclusive cumsum on extend_seq_lens.
        expected_write_offsets = [0, 8, 9, 10]
        for i, value in enumerate(expected_write_offsets):
            assert int(triton_w.write_offsets[i].item()) == value
        # Verify count = 0 + 4 + 10 = 14.
        assert int(triton_v.verify_num_valid[0].item()) == 14


class TestSeedSlot:
    def test_prefix_zero_seed_is_minus_one(self) -> None:
        """prefix=0 → seed_slot_idx = -1 (no predecessor to anchor on)."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([0]),
            extend_seq_lens=_tensor([3]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        assert int(plans[0][1].write_seed_slot_indices[0].item()) == -1

    def test_prev_slot_minus_one_at_chain_head(self) -> None:
        """pos=0 entry → verify_prev_slot_indices == -1 (chain head)."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        # First entry has pos=0 → prev_slot = -1.
        assert int(plans[0][0].verify_prev_slot_indices[0].item()) == -1

    def test_prev_slot_is_self_minus_one(self) -> None:
        """pos>0 entry → prev = req_to_token[rp, pos-1] (SWA-translated when SWA enabled)."""
        max_seq_len = 16
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([2]),
            prefix_lens=_tensor([4]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v = plans[0][0]
        # entry[1] is pos=1: prev_slot = req_to_token[2, 0] = 2 * max_seq_len + 0 = 32.
        assert int(triton_v.verify_prev_slot_indices[1].item()) == 2 * max_seq_len + 0
        # entry[2] is pos=2: prev_slot = req_to_token[2, 1] = 2 * max_seq_len + 1 = 33.
        assert int(triton_v.verify_prev_slot_indices[2].item()) == 2 * max_seq_len + 1

    def test_seed_translated_through_permuted_lut(self) -> None:
        """Permuted LUT: seed slot is the LUT-lookup of req_to_token[rp, prefix-1], NOT identity."""
        rng = random.Random(42)
        max_seq_len = 16
        max_reqs = 4
        pool_size = max_reqs * max_seq_len
        lut = make_lut(kind="permutation", pool_size=pool_size, device=_DEVICE, rng=rng)
        rtt = make_req_to_token(
            kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
        )

        rp = 2
        prefix = 5
        req_pool_indices = _tensor([rp])
        prefix_lens = _tensor([prefix])
        extend_seq_lens = _tensor([1])

        full_seed_slot = rp * max_seq_len + (prefix - 1)
        expected_seed = int(lut[full_seed_slot].item())

        extras = empty_extras()
        verify_capacity, write_req_capacity = _alloc_for_inputs(
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            extras_count=0,
            swa_window_size=max_seq_len,
        )
        for label in ("real", "ref"):
            _, w_plan = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=max_seq_len,
                full_to_swa_index_mapping=lut,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            actual_seed = int(w_plan.write_seed_slot_indices[0].item())
            assert (
                actual_seed == expected_seed
            ), f"[{label}] permuted-LUT seed expected {expected_seed} got {actual_seed}"

    def test_swa_window_head_prev_slot_is_real_predecessor(self) -> None:
        """SWA window with non-zero window_start: head entry's prev_slot != -1; it is the real predecessor."""
        rng = random.Random(13)
        max_seq_len = 256
        max_reqs = 2
        pool_size = max_reqs * max_seq_len
        swa_window_size = 128
        prefix = 200
        rp = 1
        lut = make_lut(kind="permutation", pool_size=pool_size, device=_DEVICE, rng=rng)
        rtt = make_req_to_token(
            kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
        )

        req_pool_indices = _tensor([rp])
        prefix_lens = _tensor([prefix])
        extend_seq_lens = _tensor([1])
        extras = empty_extras()

        window_start = prefix - swa_window_size
        full_prev_slot = int(rtt[rp, window_start - 1].item())
        expected_prev = int(lut[full_prev_slot].item())

        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="loose", total_verify=swa_window_size, extras_count=0, bs=1
        )

        for label in ("real", "ref"):
            v_plan, _ = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=swa_window_size,
                full_to_swa_index_mapping=lut,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            actual_prev = int(v_plan.verify_prev_slot_indices[0].item())
            assert (
                actual_prev != -1
            ), f"[{label}] SWA window head must have real predecessor, got -1"
            assert (
                actual_prev == expected_prev
            ), f"[{label}] expected prev={expected_prev} got {actual_prev}"


class TestPadding:
    def test_padding_rows_contribute_zero(self) -> None:
        """``req_pool_indices[r] == 0`` rows → no verify entry, no write entry, seed = -1."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        # Step: bs=3 with row 1 marked as padding (rpi=0).
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 0, 2]),
            prefix_lens=_tensor([5, 99, 3]),
            extend_seq_lens=_tensor([1, 99, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # verify count = 5 (req0) + 0 (padding) + 3 (req2) = 8.
        assert int(triton_v.verify_num_valid[0].item()) == 8
        # write_offsets cumsum: [0, 1, 1, 2] — padding row contributes 0.
        expected_write_offsets = [0, 1, 1, 2]
        for i, value in enumerate(expected_write_offsets):
            assert int(triton_w.write_offsets[i].item()) == value
        # Padding row's seed must be -1.
        assert int(triton_w.write_seed_slot_indices[1].item()) == -1

    def test_per_req_slot_when_req_to_token_is_sparse(self) -> None:
        """sparse_permuted rtt: verify_slot_indices read directly from the constructed table."""
        rng = random.Random(7)
        max_seq_len = 8
        max_reqs = 3
        rtt = make_req_to_token(
            kind="sparse_permuted",
            max_reqs=max_reqs,
            max_seq_len=max_seq_len,
            device=_DEVICE,
            rng=rng,
        )
        rp = 1
        prefix = 4
        req_pool_indices = _tensor([rp])
        prefix_lens = _tensor([prefix])
        extend_seq_lens = _tensor([1])
        extras = empty_extras()
        verify_capacity, write_req_capacity = _alloc_for_inputs(
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            extras_count=0,
            swa_window_size=0,
        )

        expected_slots = [int(rtt[rp, pos].item()) for pos in range(prefix)]

        for label in ("real", "ref"):
            v_plan, _ = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            actual_slots = v_plan.verify_slot_indices[:prefix].detach().cpu().tolist()
            assert (
                actual_slots == expected_slots
            ), f"[{label}] sparse-rtt slots expected {expected_slots} got {actual_slots}"

    def test_padding_row_with_garbage_prefix_does_not_oob(self) -> None:
        """rpi==0 padding row with absurd prefix_lens must not OOB-read req_to_token (row is skipped)."""
        req_pool_indices = _tensor([1, 0, 2])
        prefix_lens = _tensor([5, 99999, 3])
        extend_seq_lens = _tensor([1, 99999, 1])
        rtt = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        extras = empty_extras()
        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="loose", total_verify=8, extras_count=0, bs=3
        )

        for label in ("real", "ref"):
            v_plan, w_plan = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            assert int(v_plan.verify_num_valid[0].item()) == 8, label
            assert (
                int(w_plan.write_seed_slot_indices[1].item()) == -1
            ), f"[{label}] padding row seed must be -1"
            PlanInvariants.assert_all(
                verify_plan=v_plan,
                write_plan=w_plan,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                swa_window_size=0,
                extras_slot_indices=extras[0],
                extras_positions=extras[1],
                extras_prev_slot_indices=extras[2],
                extras_count=0,
            )


class TestSwa:
    def test_swa_window_clip_prefix_less_than_window(self) -> None:
        """SWA: prefix=3 < window=128 → window_start=0, verify covers 3 entries (no clip)."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=512, device=_DEVICE
        )
        # Identity LUT keeps slot indices unchanged after SWA translation.
        full_pool_size = 4 * 512
        lut = torch.arange(full_pool_size + 1, dtype=torch.int64, device=_DEVICE)
        plans = _plan_pair(verify_capacity=256, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=128,
            full_to_swa_index_mapping=lut,
        )

        assert int(plans[0][0].verify_num_valid[0].item()) == 3

    def test_swa_window_clip_prefix_gt_window(self) -> None:
        """SWA: prefix=200 > window=128 → window_start=72, verify covers 128 entries."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=512, device=_DEVICE
        )
        full_pool_size = 4 * 512
        lut = torch.arange(full_pool_size + 1, dtype=torch.int64, device=_DEVICE)
        plans = _plan_pair(verify_capacity=512, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([200]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=128,
            full_to_swa_index_mapping=lut,
        )

        triton_v = plans[0][0]
        assert int(triton_v.verify_num_valid[0].item()) == 128
        # First verify entry should be at position 72.
        assert int(triton_v.verify_expected_positions[0].item()) == 72

    def test_swa_lut_translates_verify_slots(self) -> None:
        """FULL slot → SWA slot translation is performed inside the plan kernel for verify_slot_indices."""
        max_seq_len = 16
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        full_pool_size = 4 * max_seq_len
        # Build a LUT that maps FULL slot S → SWA slot (S + 100) for every S; chosen so we can distinguish a
        # translated value from a raw full slot.
        lut = (
            torch.arange(full_pool_size + 1, dtype=torch.int64, device=_DEVICE) + 100
        ).contiguous()
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=128,
            full_to_swa_index_mapping=lut,
        )

        triton_v = plans[0][0]
        # FULL slot for (rp=1, pos=0) = 1 * max_seq_len + 0 = 16; expected SWA slot = 16 + 100 = 116.
        assert int(triton_v.verify_slot_indices[0].item()) == 1 * max_seq_len + 0 + 100
        assert int(triton_v.verify_slot_indices[1].item()) == 1 * max_seq_len + 1 + 100
        assert int(triton_v.verify_slot_indices[2].item()) == 1 * max_seq_len + 2 + 100

    def test_swa_lut_translates_seed_slot(self) -> None:
        """write_seed_slot_indices is also SWA-translated inside the plan kernel."""
        max_seq_len = 16
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        full_pool_size = 4 * max_seq_len
        lut = (
            torch.arange(full_pool_size + 1, dtype=torch.int64, device=_DEVICE) + 100
        ).contiguous()
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=128,
            full_to_swa_index_mapping=lut,
        )

        triton_w = plans[0][1]
        # FULL slot at (rp=1, pos=2) = 1 * max_seq_len + 2 = 18; expected SWA seed = 18 + 100 = 118.
        assert (
            int(triton_w.write_seed_slot_indices[0].item()) == 1 * max_seq_len + 2 + 100
        )

    def test_verify_covers_all_tokens_in_swa_window(self) -> None:
        """SWA group with window=128 + bs=4 → verify_num_valid == Σ min(prefix_lens[r], 128)."""
        window = 128
        prefix_values = [50, 128, 200, 1024]
        max_seq_len = 2048
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        full_pool_size = 4 * max_seq_len
        lut = torch.arange(full_pool_size + 1, dtype=torch.int64, device=_DEVICE)
        plans = _plan_pair(verify_capacity=1024, write_req_capacity=8)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 3, 1]),
            prefix_lens=_tensor(prefix_values),
            extend_seq_lens=_tensor([1, 1, 1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=window,
            full_to_swa_index_mapping=lut,
        )

        expected_total = sum(min(p, window) for p in prefix_values)
        assert int(plans[0][0].verify_num_valid[0].item()) == expected_total


class TestNoExtras:
    def test_plan_num_valid_counts_only_per_req_entries(self) -> None:
        """Sweep extras are written directly into VerifyPlan; plan kernel does not append them."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            extend_seq_lens=_tensor([1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v = plans[0][0]
        assert int(triton_v.verify_num_valid[0].item()) == 3

    def test_zero_prefix_has_no_verify_entries(self) -> None:
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([0]),
            extend_seq_lens=_tensor([5]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        assert int(plans[0][0].verify_num_valid[0].item()) == 0

    def test_verify_capacity_just_fits_per_req_entries(self) -> None:
        rp = 1
        prefix = 4
        req_pool_indices = _tensor([rp])
        prefix_lens = _tensor([prefix])
        extend_seq_lens = _tensor([1])
        rtt = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )

        total_verify = prefix
        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="tight_match",
            total_verify=total_verify,
            extras_count=0,
            bs=1,
        )

        for label in ("real", "ref"):
            v_plan, _ = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            n = int(v_plan.verify_num_valid[0].item())
            assert n == total_verify, f"[{label}] num_valid {n}"
            assert int(v_plan.enable[0].item()) == 1

    def test_verify_capacity_undershoot_by_one(self) -> None:
        rp = 1
        prefix = 3
        req_pool_indices = _tensor([rp])
        prefix_lens = _tensor([prefix])
        extend_seq_lens = _tensor([1])
        rtt = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )

        total_verify = prefix
        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="under_by_one",
            total_verify=total_verify,
            extras_count=0,
            bs=1,
        )

        real_v, _ = _run_label(
            label="real",
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=rtt,
            extras=empty_extras(),
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        ref_v, _ = _run_label(
            label="ref",
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=rtt,
            extras=empty_extras(),
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        n_real = int(real_v.verify_num_valid[0].item())
        n_ref = int(ref_v.verify_num_valid[0].item())
        assert n_real == n_ref, f"real {n_real} vs ref {n_ref} diverged under cap"
        assert n_real == verify_capacity
        assert int(real_v.enable[0].item()) == 0
        assert int(ref_v.enable[0].item()) == 0


class TestMisc:
    def test_zero_extend_writes_empty_write_plan(self) -> None:
        """``extend_seq_lens`` all zero → write offsets stay zero; VerifyPlan still populated."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([4, 6]),
            extend_seq_lens=_tensor([0, 0]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # VerifyPlan covers 4+6 = 10 entries; write offsets are zero.
        assert int(triton_v.verify_num_valid[0].item()) == 10
        # write_offsets cumsum of zeros stays zero across the active prefix.
        assert int(triton_w.write_offsets[0].item()) == 0
        assert int(triton_w.write_offsets[1].item()) == 0
        assert int(triton_w.write_offsets[2].item()) == 0
        # Seeds for write-empty reqs must be -1 per plan semantics.
        assert int(triton_w.write_seed_slot_indices[0].item()) == -1
        assert int(triton_w.write_seed_slot_indices[1].item()) == -1

    def test_replay_same_inputs_yields_same_outputs(self) -> None:
        """Two consecutive runs on identical inputs produce byte-equal plans (kernel is pure)."""
        req_pool_indices = _tensor([1, 2, 3])
        prefix_lens = _tensor([4, 7, 2])
        extend_seq_lens = _tensor([2, 1, 3])
        rtt = make_req_to_token(
            kind="linear", max_reqs=8, max_seq_len=16, device=_DEVICE
        )
        extras = empty_extras()
        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="loose", total_verify=13, extras_count=0, bs=3
        )

        for label in ("real", "ref"):
            run1_v, run1_w = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            run2_v, run2_w = _run_label(
                label=label,
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=rtt,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                write_req_capacity=write_req_capacity,
            )
            assert torch.equal(
                run1_v.verify_slot_indices, run2_v.verify_slot_indices
            ), label
            assert torch.equal(
                run1_v.verify_expected_positions, run2_v.verify_expected_positions
            ), label
            assert torch.equal(
                run1_v.verify_prev_slot_indices, run2_v.verify_prev_slot_indices
            ), label
            assert torch.equal(run1_v.verify_num_valid, run2_v.verify_num_valid), label
            assert torch.equal(run1_w.write_offsets, run2_w.write_offsets), label
            assert torch.equal(
                run1_w.write_seed_slot_indices, run2_w.write_seed_slot_indices
            ), label
            assert torch.equal(
                run1_w.write_num_valid_reqs, run2_w.write_num_valid_reqs
            ), label

    def test_shrink_bs_clears_stale_write_offsets(self) -> None:
        """Reusing a WritePlan with smaller bs: write_offsets beyond new bs must be zeroed by the kernel."""
        rtt = make_req_to_token(
            kind="linear", max_reqs=16, max_seq_len=16, device=_DEVICE
        )
        verify_capacity, write_req_capacity = derive_plan_capacity(
            kind="loose", total_verify=80, extras_count=0, bs=8
        )

        for label in ("real", "ref"):
            big_rpi = _tensor([1, 2, 3, 4, 5, 6, 7, 8])
            big_pfx = _tensor([10] * 8)
            big_ext = _tensor([1] * 8)
            small_rpi = _tensor([1, 2, 3])
            small_pfx = _tensor([5, 5, 5])
            small_ext = _tensor([1, 1, 1])

            verify_plan = VerifyPlan.allocate(
                verify_capacity=verify_capacity, device=_DEVICE
            ).zero_for_testing_()
            write_plan = WritePlan.allocate(
                write_req_capacity=write_req_capacity, device=_DEVICE
            ).zero_for_testing_()
            runner = (
                launch_canary_plan_kernels
                if label == "real"
                else launch_canary_plan_kernels_torch_reference
            )
            runner(
                verify_plan_out=verify_plan,
                write_plan_out=write_plan,
                req_pool_indices=big_rpi,
                prefix_lens=big_pfx,
                extend_seq_lens=big_ext,
                req_to_token=rtt,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                req_to_verify_expected_tokens=None,
                req_to_verify_expected_tokens_valid_lens=None,
                kv_token_id_vs_position_offset=0,
            )
            torch.cuda.synchronize()
            runner(
                verify_plan_out=verify_plan,
                write_plan_out=write_plan,
                req_pool_indices=small_rpi,
                prefix_lens=small_pfx,
                extend_seq_lens=small_ext,
                req_to_token=rtt,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=verify_capacity,
                req_to_verify_expected_tokens=None,
                req_to_verify_expected_tokens_valid_lens=None,
                kv_token_id_vs_position_offset=0,
            )
            torch.cuda.synchronize()
            n_active = int(write_plan.write_num_valid_reqs[0].item())
            tail_offsets = (
                write_plan.write_offsets[n_active + 1 : 8].detach().cpu().tolist()
            )
            assert all(
                v == 0 for v in tail_offsets
            ), f"[{label}] stale write_offsets tail not cleared: {tail_offsets}"


class TestVerifyContent:
    def test_verify_num_valid_aggregate(self) -> None:
        """``verify_num_valid == sum(per-req verify_count)``."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 3]),
            prefix_lens=_tensor([2, 5, 4]),
            extend_seq_lens=_tensor([1, 1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        assert int(plans[0][0].verify_num_valid[0].item()) == 11

    def test_verify_covers_all_tokens_no_skip(self) -> None:
        """FULL group + bs=4 → verify_num_valid == Σ(prefix_lens) — every prefix token verified."""
        # Step: 4 reqs with mixed prefix and extend; FULL group means no SWA window clip.
        prefix_values = [0, 3, 7, 12]
        extend_values = [4, 1, 1, 1]
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=32, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=128, write_req_capacity=8)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 3, 1]),
            prefix_lens=_tensor(prefix_values),
            extend_seq_lens=_tensor(extend_values),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        assert int(plans[0][0].verify_num_valid[0].item()) == sum(prefix_values)

    def test_plan_verify_expected_positions_strictly_increment_per_req(self) -> None:
        """Per req, verify_expected_positions[verify_offsets[r]:verify_offsets[r+1]] == [window_start..prefix-1]."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=32, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([5, 8]),
            extend_seq_lens=_tensor([1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v = plans[0][0]
        # Req 0: positions [0..5); Req 1: positions [0..8).
        req0_positions = triton_v.verify_expected_positions[:5].cpu().tolist()
        req1_positions = triton_v.verify_expected_positions[5:13].cpu().tolist()
        assert req0_positions == [0, 1, 2, 3, 4]
        assert req1_positions == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_write_num_valid_reqs_excludes_padding(self) -> None:
        """Padding rows (rpi == 0) contribute zero to write_offsets so the write kernel does no work for them, but they ARE included in write_num_valid_reqs, which equals bs (the full batch size including padding)."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=16, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=8)
        # bs=4, last two rows are padding.
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 0, 0]),
            prefix_lens=_tensor([3, 5, 99, 99]),
            extend_seq_lens=_tensor([1, 1, 99, 99]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_w = plans[0][1]
        # Padding rows must contribute 0 to write_offsets cumsum: [0, 1, 2, 2, 2].
        expected_write_offsets = [0, 1, 2, 2, 2]
        for i, value in enumerate(expected_write_offsets):
            assert int(triton_w.write_offsets[i].item()) == value


class TestByteEqual:
    def test_byte_equal_python_reference(self) -> None:
        """End-to-end Triton vs Python ref byte-equal across a representative bs=4 case (no SWA)."""
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=32, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=128, write_req_capacity=8)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2, 3, 1]),
            prefix_lens=_tensor([0, 3, 8, 15]),
            extend_seq_lens=_tensor([4, 1, 1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

    def test_byte_equal_python_reference_hardcoded(self) -> None:
        """bs=3, three prefix combinations → hand-computed verify_offsets / write_offsets / seed slots."""
        # Step 1: pin (prefix, extend) per req.
        prefixes = [0, 4, 7]
        extends = [3, 1, 1]
        rps = [1, 2, 3]
        max_seq_len = 16
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor(rps),
            prefix_lens=_tensor(prefixes),
            extend_seq_lens=_tensor(extends),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

        triton_v, triton_w = plans[0]
        # Step 2: hand-compute expected write_offsets (exclusive cumsum of extends) and verify_num_valid.
        expected_write_offsets = [0, 3, 4, 5]
        expected_verify_num_valid = sum(prefixes)
        expected_seeds = [
            -1,  # prefix=0 → no predecessor
            rps[1] * max_seq_len + (prefixes[1] - 1),
            rps[2] * max_seq_len + (prefixes[2] - 1),
        ]

        for i, value in enumerate(expected_write_offsets):
            assert (
                int(triton_w.write_offsets[i].item()) == value
            ), f"write_offsets[{i}] expected {value} got {int(triton_w.write_offsets[i].item())}"
        assert (
            int(triton_v.verify_num_valid[0].item()) == expected_verify_num_valid
        ), f"verify_num_valid expected {expected_verify_num_valid}"
        for i, expected_seed in enumerate(expected_seeds):
            assert (
                int(triton_w.write_seed_slot_indices[i].item()) == expected_seed
            ), f"write_seed_slot_indices[{i}] expected {expected_seed}"


class TestBoundarySweep:
    @pytest.mark.parametrize("bs", [1, 31, 32, 33, 128])
    def test_bs_boundary_byte_equal_sweep(self, bs: int) -> None:
        """Sweep bs boundary values around Triton block boundaries; assert Triton vs ref byte-equal."""
        req_pool_indices = list(range(1, bs + 1))
        prefix_lens = [10] * bs
        extend_seq_lens = [1] * bs
        max_seq_len = 32
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=bs + 1, max_seq_len=max_seq_len, device=_DEVICE
        )

        total_verify = sum(min(p, max_seq_len) for p in prefix_lens)
        plans = _plan_pair(
            verify_capacity=max(total_verify + 64, 256), write_req_capacity=bs + 4
        )
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor(req_pool_indices),
            prefix_lens=_tensor(prefix_lens),
            extend_seq_lens=_tensor(extend_seq_lens),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

    @pytest.mark.parametrize("prefix_val", [0, 1, 127, 128, 129, 4096])
    def test_prefix_lens_boundary_byte_equal_sweep(self, prefix_val: int) -> None:
        """Sweep prefix_lens boundary values; assert Triton vs ref byte-equal."""
        max_seq_len = max(prefix_val + 4, 256)
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        total_verify = prefix_val + 10
        plans = _plan_pair(
            verify_capacity=max(total_verify + 64, 256), write_req_capacity=8
        )
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([prefix_val, 10]),
            extend_seq_lens=_tensor([1, 1]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )

    @pytest.mark.parametrize("extend_val", [1, 128, 4096])
    def test_extend_seq_lens_boundary_byte_equal_sweep(self, extend_val: int) -> None:
        """Sweep extend_seq_lens boundary values; assert Triton vs ref byte-equal."""
        max_seq_len = max(extend_val + 4, 64)
        req_to_token = make_req_to_token(
            kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
        )
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([0]),
            extend_seq_lens=_tensor([extend_val]),
            req_to_token=req_to_token,
            extras=empty_extras(),
        )


class TestExpectedTokenPool:
    """Cover the optional ``req_to_verify_expected_tokens`` pool input and
    ``kv_token_id_vs_position_offset`` shift in the plan_entries kernel.

    Pool dtype is int32 with layout ``[max_reqs, pool_max_context_len]``.
    For each verify entry the kernel gathers ``expected_input_id =
    req_to_verify_expected_tokens[rp, position + offset]`` and writes ``-1`` as
    a sentinel when the pool is absent or the gather index is out of range.
    """

    _MAX_SEQ_LEN = 16
    _MAX_REQS = 4
    _POOL_COLS = 16
    _DEFAULT = object()

    def setup_method(self) -> None:
        self.req_to_token = make_req_to_token(
            kind="linear",
            max_reqs=self._MAX_REQS,
            max_seq_len=self._MAX_SEQ_LEN,
            device=_DEVICE,
        )
        self.pool = self._make_pool(fill_fn=lambda rp, pos: rp * 1000 + pos)

    def _make_pool(
        self, *, fill_fn, pool_max_context_len: int | None = None
    ) -> torch.Tensor:
        cols = self._POOL_COLS if pool_max_context_len is None else pool_max_context_len
        pool = torch.full(
            (self._MAX_REQS, cols), -999, dtype=torch.int32, device=_DEVICE
        )
        for rp in range(self._MAX_REQS):
            for pos in range(cols):
                pool[rp, pos] = fill_fn(rp, pos)
        return pool

    def _run(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        pool: object = _DEFAULT,
        offset: int = 0,
        swa_window_size: int = 0,
        full_to_swa_index_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run plan_diff with the per-class fixed inputs; return the expected_tokens slice up to verify_num_valid."""
        bs = int(req_pool_indices.shape[0])
        pool_arg = self.pool if pool is self._DEFAULT else pool
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=_tensor([1] * bs),
            req_to_token=self.req_to_token,
            extras=empty_extras(),
            swa_window_size=swa_window_size,
            full_to_swa_index_mapping=full_to_swa_index_mapping,
            req_to_verify_expected_tokens=pool_arg,
            kv_token_id_vs_position_offset=offset,
        )
        triton_v = plans[0][0]
        n_valid = int(triton_v.verify_num_valid[0].item())
        return triton_v.verify_expected_tokens[:n_valid]

    @staticmethod
    def _expected(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=_DEVICE)

    def test_pool_disabled_writes_minus_one_sentinel(self) -> None:
        """pool=None default path: every verify entry's expected_token slot is -1."""
        got = self._run(
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([3, 5]),
            pool=None,
        )
        assert got.shape[0] == 8
        assert torch.equal(got, self._expected([-1] * 8))

    def test_pool_enabled_target_offset_0_byte_equal(self) -> None:
        """offset=0 (target pool): expected_token[i] == pool[rp, position[i]]."""
        got = self._run(
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([3, 5]),
        )
        assert torch.equal(
            got,
            self._expected(
                [
                    rp * 1000 + pos
                    for rp, plen in [(1, 3), (2, 5)]
                    for pos in range(plen)
                ]
            ),
        )

    def test_pool_enabled_eagle_offset_plus_1_byte_equal(self) -> None:
        """offset=+1 (EAGLE draft): expected_token[i] == pool[rp, position[i] + 1]."""
        got = self._run(
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([3, 5]),
            offset=1,
        )
        assert torch.equal(
            got,
            self._expected(
                [
                    rp * 1000 + pos + 1
                    for rp, plen in [(1, 3), (2, 5)]
                    for pos in range(plen)
                ]
            ),
        )

    def test_pool_oob_above_size0_writes_sentinel(self) -> None:
        """positions whose ``position + offset`` exceed pool_max_context_len get -1; in-range slots stay correct."""
        pool_cols = 4
        small_pool = self._make_pool(
            fill_fn=lambda rp, pos: rp * 1000 + pos, pool_max_context_len=pool_cols
        )
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([6]),
            pool=small_pool,
        )
        assert torch.equal(
            got,
            self._expected([1000 + pos if pos < pool_cols else -1 for pos in range(6)]),
        )

    def test_pool_oob_offset_plus_1_byte_equal_triggers_sentinel(self) -> None:
        """offset=+1 path that pushes the last entry past pool cols still byte-equals the ref (sentinel scatter)."""
        pool_cols = 4
        small_pool = self._make_pool(
            fill_fn=lambda rp, pos: rp * 1000 + pos, pool_max_context_len=pool_cols
        )
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([4]),
            pool=small_pool,
            offset=1,
        )
        assert torch.equal(
            got,
            self._expected(
                [1000 + pos + 1 if pos + 1 < pool_cols else -1 for pos in range(4)]
            ),
        )


class TestExpectedTokenPoolValidLens:
    _MAX_SEQ_LEN = 16
    _MAX_REQS = 4
    _POOL_COLS = 16

    def setup_method(self) -> None:
        self.req_to_token = make_req_to_token(
            kind="linear",
            max_reqs=self._MAX_REQS,
            max_seq_len=self._MAX_SEQ_LEN,
            device=_DEVICE,
        )
        self.pool = self._make_pool(fill_fn=lambda rp, pos: rp * 1000 + pos)

    def _make_pool(self, *, fill_fn) -> torch.Tensor:
        pool = torch.full(
            (self._MAX_REQS, self._POOL_COLS), -999, dtype=torch.int32, device=_DEVICE
        )
        for rp in range(self._MAX_REQS):
            for pos in range(self._POOL_COLS):
                pool[rp, pos] = fill_fn(rp, pos)
        return pool

    def _run(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        valid_lens: torch.Tensor,
        pool: torch.Tensor | None = None,
        offset: int = 0,
        swa_window_size: int = 0,
        full_to_swa_index_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run plan_diff with the per-class fixed inputs; return the expected_tokens slice up to verify_num_valid."""
        bs = int(req_pool_indices.shape[0])
        plans = _plan_pair(verify_capacity=64, write_req_capacity=4)
        run_plan_diff(
            plan_pair=plans,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=_tensor([1] * bs),
            req_to_token=self.req_to_token,
            extras=empty_extras(),
            swa_window_size=swa_window_size,
            full_to_swa_index_mapping=full_to_swa_index_mapping,
            req_to_verify_expected_tokens=self.pool if pool is None else pool,
            req_to_verify_expected_tokens_valid_lens=valid_lens,
            kv_token_id_vs_position_offset=offset,
        )
        triton_v = plans[0][0]
        n_valid = int(triton_v.verify_num_valid[0].item())
        return triton_v.verify_expected_tokens[:n_valid]

    @staticmethod
    def _expected(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=_DEVICE)

    def test_valid_lens_boundary_emits_sentinel_at_limit(self) -> None:
        """sot_pos == valid_lens[r] is OUT of range; the kernel must emit -1 even though the pool has a real value at that slot."""
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            valid_lens=_tensor([2]),
        )
        assert torch.equal(got, self._expected([1000, 1001, -1]))

    def test_valid_lens_within_limit_reads_pool(self) -> None:
        """sot_pos == valid_lens[r] - 1 is IN range; the kernel must gather the pool value, not -1."""
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            valid_lens=_tensor([3]),
        )
        assert torch.equal(got, self._expected([1000, 1001, 1002]))

    def test_valid_lens_mixed_across_reqs(self) -> None:
        """Per-req different valid_lens in one batch: each req's gather is bounded by its own lens, not the batch max."""
        got = self._run(
            req_pool_indices=_tensor([1, 2]),
            prefix_lens=_tensor([3, 3]),
            valid_lens=_tensor([2, 4]),
        )
        assert torch.equal(got, self._expected([1000, 1001, -1, 2000, 2001, 2002]))

    def test_valid_lens_zero_emits_all_sentinel_for_that_req(self) -> None:
        """valid_lens[r] == 0 disables every gather for req r regardless of pool content."""
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            valid_lens=_tensor([0]),
        )
        assert torch.equal(got, self._expected([-1, -1, -1]))

    def test_valid_lens_masks_stale_pool_data_above_bound(self) -> None:
        """Pool has realistic-looking values past valid_lens (the recycled-slot motivation): kernel still emits -1 for them."""
        # Positions 2..15 carry a longer previous owner's leftover token; the bound must hide them.
        stale_pool = self._make_pool(
            fill_fn=lambda rp, pos: 7777 if pos >= 2 else (rp * 1000 + pos),
        )
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([5]),
            valid_lens=_tensor([2]),
            pool=stale_pool,
        )
        assert torch.equal(got, self._expected([1000, 1001, -1, -1, -1]))

    def test_valid_lens_with_offset_plus_1_bounds_after_shift(self) -> None:
        """``sot_pos = position + offset`` is compared against valid_lens; the offset shifts before the bound check."""
        # sot_pos for positions 0,1,2 is 1,2,3. valid_lens=2 → only sot_pos=1 reads pool.
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([3]),
            valid_lens=_tensor([2]),
            offset=1,
        )
        assert torch.equal(got, self._expected([1001, -1, -1]))

    def test_valid_lens_applies_under_swa_window(self) -> None:
        """SWA-windowed verify entries are bounded by valid_lens the same way as the FULL pool path."""
        # prefix=5, swa_window=3 → entries cover positions 2,3,4. valid_lens=4 admits pos 2,3 only.
        got = self._run(
            req_pool_indices=_tensor([1]),
            prefix_lens=_tensor([5]),
            valid_lens=_tensor([4]),
            swa_window_size=3,
            full_to_swa_index_mapping=make_lut(
                kind="identity",
                pool_size=self._MAX_REQS * self._MAX_SEQ_LEN,
                device=_DEVICE,
            ),
        )
        assert torch.equal(got, self._expected([1002, 1003, -1]))

    def test_pool_set_but_valid_lens_missing_raises(self) -> None:
        """One-way contract: passing the pool without per-req valid_lens is rejected at the Python wrapper."""
        triton_v, triton_w = _plan_pair(verify_capacity=64, write_req_capacity=4)[0]
        with pytest.raises(
            ValueError, match="req_to_verify_expected_tokens_valid_lens"
        ):
            launch_canary_plan_kernels(
                verify_plan_out=triton_v,
                write_plan_out=triton_w,
                req_pool_indices=_tensor([1]),
                prefix_lens=_tensor([3]),
                extend_seq_lens=_tensor([1]),
                req_to_token=self.req_to_token,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
                verify_capacity=int(triton_v.verify_slot_indices.shape[0]),
                req_to_verify_expected_tokens=self.pool,
                req_to_verify_expected_tokens_valid_lens=None,
                kv_token_id_vs_position_offset=0,
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
