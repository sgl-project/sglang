"""Random differential fuzz tests: Triton canary_plan_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random as _random
from typing import Optional

import torch

from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_and_assert_plan_byte_equal as _run_both_and_assert_byte_equal,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    _allocate_plan_pair,
    _build_req_to_token,
    _empty_extras,
    _make_extras,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _build_random_plan_inputs(
    rng: _random.Random,
    *,
    bs: int,
    max_seq_len: int,
    max_prefix: int,
    max_extend: int,
    padding_fraction: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Build random (fb_req_pool_indices, fb_prefix_lens, fb_extend_seq_lens, req_to_token, max_rp)."""
    max_rp = bs + 2
    req_pool_indices: list[int] = []
    prefix_lens: list[int] = []
    extend_lens: list[int] = []
    for row in range(bs):
        if row > 0 and rng.random() < padding_fraction:
            req_pool_indices.append(0)
            prefix_lens.append(0)
            extend_lens.append(0)
        else:
            rp = rng.randint(1, max_rp - 1)
            req_pool_indices.append(rp)
            pfx = rng.randint(0, max_prefix)
            ext = rng.randint(1, max_extend)
            prefix_lens.append(pfx)
            extend_lens.append(ext)
    fb_req_pool_indices = torch.tensor(
        req_pool_indices, dtype=torch.int32, device=_DEVICE
    )
    fb_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
    rp_axis = torch.arange(max_rp, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    req_to_token = (rp_axis * max_seq_len + pos_axis).contiguous()
    return fb_req_pool_indices, fb_prefix_lens, fb_extend_seq_lens, req_to_token, max_rp


def test_plan_pure_random_fuzz_byte_equal() -> None:
    """100 fully-random plan inputs — Triton vs ref must be byte-equal on every iteration."""
    rng = _random.Random(0)
    for iteration in range(100):
        bs = rng.randint(1, 16)
        max_seq_len = rng.randint(16, 128)
        max_prefix = max_seq_len - 1
        max_extend = rng.randint(1, 16)
        swa_enabled = rng.random() < 0.4
        swa_window_size = rng.randint(4, max_seq_len) if swa_enabled else 0

        fb_rpi, fb_pfx, fb_ext, req_to_token, max_rp = _build_random_plan_inputs(
            rng,
            bs=bs,
            max_seq_len=max_seq_len,
            max_prefix=max_prefix,
            max_extend=max_extend,
            padding_fraction=0.1,
        )

        extra_count = rng.randint(0, 4)
        if extra_count > 0:
            extra_slots_list = rng.sample(range(500, 600), extra_count)
            extra_positions_list = sorted(rng.sample(range(200, 300), extra_count))
            extra_prevs_list = [-1] + extra_slots_list[: extra_count - 1]
            extras = _make_extras(
                slot_indices=extra_slots_list,
                positions=extra_positions_list,
                prev_slot_indices=extra_prevs_list,
                capacity=extra_count + 2,
            )
        else:
            extras = _empty_extras()

        total_prefix = int(fb_pfx.sum().item())
        verify_capacity = max(total_prefix + extra_count + 64, 128)
        write_req_capacity = bs + 4

        full_to_swa_lut: Optional[torch.Tensor]
        if swa_window_size > 0:
            pool_size = max_rp * max_seq_len
            full_to_swa_lut = torch.arange(
                pool_size + 1, dtype=torch.int32, device=_DEVICE
            )
        else:
            full_to_swa_lut = None

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=extras,
                swa_window_size=swa_window_size,
                full_to_swa_index_mapping=full_to_swa_lut,
            )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} "
                f"bs={bs} max_seq_len={max_seq_len} swa_window_size={swa_window_size} "
                f"extra_count={extra_count} "
                f"fb_rpi={fb_rpi.tolist()} fb_pfx={fb_pfx.tolist()} fb_ext={fb_ext.tolist()}"
            ) from exc


def test_plan_random_extend_only() -> None:
    """25 random extend-only batches (all prefix_lens=0) — byte-equal and write_offsets == cumsum(ext)."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 12)
        max_seq_len = 32
        extend_lens = [rng.randint(1, 12) for _ in range(bs)]
        rps = [rng.randint(1, bs + 1) for _ in range(bs)]

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.zeros(bs, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=64,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert (
                int(triton_v.verify_num_valid[0].item()) == 0
            ), f"iteration={iteration}: extend-only batch should have 0 verify entries"
            cumsum = 0
            for i, ext in enumerate(extend_lens):
                assert (
                    int(triton_w.write_offsets[i].item()) == cumsum
                ), f"iteration={iteration} i={i}: write_offsets mismatch"
                cumsum += ext
            assert int(triton_w.write_offsets[bs].item()) == cumsum
            for i in range(bs):
                assert (
                    int(triton_w.write_seed_slot_indices[i].item()) == -1
                ), f"iteration={iteration} i={i}: extend-only seed should be -1"
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} extend_lens={extend_lens} rps={rps}"
            ) from exc


def test_plan_random_decode_only() -> None:
    """25 random decode-only batches (all extend_lens=1, prefix>0) — byte-equal and seeds != -1."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 12)
        max_seq_len = 32
        prefix_lens = [rng.randint(1, 20) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=total_verify + 32,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert (
                int(triton_v.verify_num_valid[0].item()) == total_verify
            ), f"iteration={iteration}: verify_num_valid mismatch"
            for i, (rp, pfx) in enumerate(zip(rps, prefix_lens)):
                expected_seed = rp * max_seq_len + (pfx - 1)
                assert (
                    int(triton_w.write_seed_slot_indices[i].item()) == expected_seed
                ), f"iteration={iteration} i={i}: seed mismatch"
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} prefix_lens={prefix_lens} rps={rps}"
            ) from exc


def test_plan_random_mixed_extend_decode() -> None:
    """25 random mixed batches — byte-equal; verify_num_valid == sum(prefix_lens)."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(2, 10)
        max_seq_len = 32
        prefix_lens = [rng.randint(0, 15) for _ in range(bs)]
        extend_lens = [rng.randint(1, 8) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(total_verify + 32, 64),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert (
                int(triton_v.verify_num_valid[0].item()) == total_verify
            ), f"iteration={iteration}: verify_num_valid mismatch expected={total_verify}"
            cumsum = 0
            for i, ext in enumerate(extend_lens):
                assert (
                    int(triton_w.write_offsets[i].item()) == cumsum
                ), f"iteration={iteration} i={i}: write_offsets[{i}] expected={cumsum}"
                cumsum += ext
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} prefix_lens={prefix_lens} extend_lens={extend_lens}"
            ) from exc


def test_plan_random_swa_clip_window_boundary() -> None:
    """25 random SWA batches — verify_num_valid == sum(min(pfx, window)); byte-equal."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 8)
        max_seq_len = 256
        window = rng.choice([16, 32, 64, 128])
        prefix_lens = [rng.randint(0, max_seq_len - 1) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)
        pool_size = (bs + 2) * max_seq_len
        lut = torch.arange(pool_size + 1, dtype=torch.int32, device=_DEVICE)

        expected_verify = sum(min(pfx, window) for pfx in prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(expected_verify + 64, 128),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=window,
                full_to_swa_index_mapping=lut,
            )
            assert int(triton_v.verify_num_valid[0].item()) == expected_verify, (
                f"iteration={iteration} window={window} prefix_lens={prefix_lens} "
                f"expected_verify={expected_verify}"
            )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} window={window} prefix_lens={prefix_lens}"
            ) from exc


def test_plan_random_sweep_extras_only() -> None:
    """25 random batches with extras and no per-req verify — byte-equal; extras appear at tail."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 8)
        max_seq_len = 16
        rps = list(range(1, bs + 1))
        extend_lens = [rng.randint(1, 4) for _ in range(bs)]

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.zeros(bs, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        n_extras = rng.randint(1, 6)
        extra_slots = sorted(rng.sample(range(1000, 1100), n_extras))
        extra_positions = list(range(n_extras))
        extra_prevs = [-1] + extra_slots[: n_extras - 1]
        extras = _make_extras(
            slot_indices=extra_slots,
            positions=extra_positions,
            prev_slot_indices=extra_prevs,
            capacity=n_extras + 2,
        )

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=n_extras + 32,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert (
                int(triton_v.verify_num_valid[0].item()) == n_extras
            ), f"iteration={iteration}: verify_num_valid expected={n_extras}"
            for k, slot in enumerate(extra_slots):
                assert (
                    int(triton_v.verify_slot_indices[k].item()) == slot
                ), f"iteration={iteration} k={k}: slot mismatch"
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} n_extras={n_extras} extra_slots={extra_slots}"
            ) from exc


def test_plan_random_padding_rows_mixed() -> None:
    """25 random batches with padding rows (rpi==0) mixed in — byte-equal; padding contributes nothing."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(3, 12)
        max_seq_len = 24
        max_rp = bs + 2

        req_pool_indices: list[int] = []
        prefix_lens: list[int] = []
        extend_lens: list[int] = []
        for row in range(bs):
            is_pad = row > 0 and rng.random() < 0.25
            if is_pad:
                req_pool_indices.append(0)
                prefix_lens.append(0)
                extend_lens.append(0)
            else:
                req_pool_indices.append(rng.randint(1, max_rp - 1))
                prefix_lens.append(rng.randint(0, 10))
                extend_lens.append(rng.randint(1, 6))

        fb_rpi = torch.tensor(req_pool_indices, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=max_rp, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        total_extend = sum(extend_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(total_verify + 32, 64),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert (
                int(triton_v.verify_num_valid[0].item()) == total_verify
            ), f"iteration={iteration}: verify_num_valid expected={total_verify}"
            assert (
                int(triton_w.write_offsets[bs].item()) == total_extend
            ), f"iteration={iteration}: write_offsets[{bs}] expected={total_extend}"
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} "
                f"req_pool_indices={req_pool_indices} prefix_lens={prefix_lens} extend_lens={extend_lens}"
            ) from exc
