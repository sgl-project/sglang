from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary import write as write_module
from sglang.kernels.ops.kv_canary.consts import splitmix64, splitmix64_mix3
from sglang.kernels.ops.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    launch_canary_verify_kernel,
)
from sglang.kernels.ops.kv_canary.write import (
    launch_canary_write_kernel,
)
from sglang.kernels.testing.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_canary_state_equal,
    assert_only_bits_set,
    chain_anchor_signed,
    make_canary_buf,
    make_canary_buf_pair,
    make_log_pair,
    make_real_kv_source,
    make_real_kv_sources,
    make_verify_plan,
    make_write_plan,
    make_write_plan_pair,
    read_slot_fields,
    stamp_pair,
    to_signed_int64,
)
from sglang.kernels.testing.kv_canary._differential import (
    _run_both_write,
    run_write_diff,
)
from sglang.kernels.testing.kv_canary._fixtures import (
    clone_real_kv_sources,
    dummy_pseudo_tensors,
)
from sglang.kernels.testing.kv_canary._hand_oracle import (
    _hand_fold_all,
    _hand_fold_partial,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")


def _int32_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=_DEVICE)


def _make_default_buf_pair(
    num_slots: int = 16, slot_stride_bytes: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    return make_canary_buf_pair(
        num_slots=num_slots, slot_stride_bytes=slot_stride_bytes, device=_DEVICE
    )


def _run_write(
    *,
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    input_ids: list[int] | torch.Tensor,
    positions: list[int] | torch.Tensor,
    out_cache_loc: list[int] | torch.Tensor,
    write_offsets: list[int] | None = None,
    seed_slot_indices: list[int] = (-1,),
    num_valid_reqs: int = 1,
    req_capacity: int | None = None,
    enable_write_verify_inputs: bool = False,
    expected_input_tokens: torch.Tensor | None = None,
    expected_input_positions: torch.Tensor | None = None,
    real_kv_sources_pair: (
        tuple[tuple[RealKvSource, ...], tuple[RealKvSource, ...]] | None
    ) = None,
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE,
    assert_equal: bool = True,
) -> tuple[FakeViolationLog, FakeViolationLog]:
    """Shared scaffold: build write plan + pseudo tensors and call ``run_write_diff``.

    ``write_offsets`` defaults to ``[0, len(input_ids)]`` (one req covering all entries).
    ``expected_input_*`` default to ``dummy_pseudo_tensors(len(input_ids))``.
    """
    ids_t = (
        input_ids
        if isinstance(input_ids, torch.Tensor)
        else _int32_tensor(list(input_ids))
    )
    pos_t = (
        positions
        if isinstance(positions, torch.Tensor)
        else _int32_tensor(list(positions))
    )
    loc_t = (
        out_cache_loc
        if isinstance(out_cache_loc, torch.Tensor)
        else _int32_tensor(list(out_cache_loc))
    )
    n_tokens = int(ids_t.shape[0])

    if write_offsets is None:
        write_offsets = [0, n_tokens]

    plan_kwargs = dict(
        write_offsets=write_offsets,
        seed_slot_indices=list(seed_slot_indices),
        num_valid_reqs=num_valid_reqs,
        device=_DEVICE,
    )
    if req_capacity is not None:
        plan_kwargs["req_capacity"] = req_capacity
    plan_pair = make_write_plan_pair(**plan_kwargs)

    if expected_input_tokens is None or expected_input_positions is None:
        pseudo_tokens, pseudo_positions = dummy_pseudo_tensors(n_tokens)
        if expected_input_tokens is None:
            expected_input_tokens = pseudo_tokens
        if expected_input_positions is None:
            expected_input_positions = pseudo_positions

    extra_kwargs: dict = {}
    if real_kv_sources_pair is not None:
        extra_kwargs["real_kv_sources_pair"] = real_kv_sources_pair
    return run_write_diff(
        buf_pair=buf_pair,
        plan_pair=plan_pair,
        input_ids=ids_t,
        positions=pos_t,
        out_cache_loc=loc_t,
        enable_write_verify_inputs=enable_write_verify_inputs,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        real_kv_hash_mode=real_kv_hash_mode,
        assert_equal=assert_equal,
        **extra_kwargs,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _WriteSingleSlotInput:
    token: int = 42
    position: int = 0
    enable_write_verify_inputs: bool = False
    real_kv_sources: tuple[RealKvSource, ...] = ()
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE


class _RecordingWriteModule:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def canary_write_step_cuda(self, *args: object) -> None:
        self.calls.append(args)


def _run_write_single_slot_byte_equal(case: _WriteSingleSlotInput) -> None:
    sources_cuda = case.real_kv_sources
    sources_ref = clone_real_kv_sources(sources_cuda)
    _run_write(
        buf_pair=_make_default_buf_pair(),
        input_ids=[case.token],
        positions=[case.position],
        out_cache_loc=[0],
        enable_write_verify_inputs=case.enable_write_verify_inputs,
        real_kv_sources_pair=(sources_cuda, sources_ref),
        real_kv_hash_mode=case.real_kv_hash_mode,
    )


class TestSeedSlot:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_seed_slot_idx_negative_uses_anchor(self) -> None:
        """``seed_slot_idx == -1`` → initial ``running_prev_hash`` is ``splitmix64(consts.CANARY_CHAIN_ANCHOR)``."""
        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[42],
            positions=[0],
            out_cache_loc=[3],
        )

        stored_token, stored_position, stored_prev_hash, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=3
        )
        assert stored_token == 42
        assert stored_position == 0
        assert stored_prev_hash == chain_anchor_signed()

    def test_seed_slot_idx_loads_predecessor(self) -> None:
        """``seed_slot_idx >= 0`` → load 3 fields (token, position, prev_hash) from ``canary_buf[seed]`` and splitmix64_mix3-advance into prev_hash."""
        # Step: pre-stamp slot 7 with a known chain link.
        seed_token, seed_position = 100, 4
        seed_prev_signed = to_signed_int64(splitmix64(consts.CANARY_CHAIN_ANCHOR))
        stamp_pair(
            self.buf_pair,
            slot_idx=7,
            token=seed_token,
            position=seed_position,
            prev_hash=seed_prev_signed,
        )

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[999],
            positions=[5],
            out_cache_loc=[2],
            seed_slot_indices=[7],
        )

        expected_prev_hash = splitmix64_mix3(
            splitmix64(consts.CANARY_CHAIN_ANCHOR), seed_token, seed_position
        )
        _, _, stored_prev_hash, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=2
        )
        assert stored_prev_hash == to_signed_int64(expected_prev_hash)

    def test_seed_slot_chain_link_continuous(self) -> None:
        """After write, ``slot[0].prev_hash`` is consistent with verify's chain reconstruction from seed."""
        # Step 1: write a chain from seed slot=7 → newly written slot=2. Then run verify with prev=7 and
        # assert no violation — i.e., slot[2].prev_hash is splitmix64_mix3(seed.prev_hash, seed.token, seed.position).
        cuda_buf = self.buf_pair[0]
        seed_token, seed_position = 11, 0
        seed_prev_signed = to_signed_int64(splitmix64(consts.CANARY_CHAIN_ANCHOR))
        stamp_pair(
            self.buf_pair,
            slot_idx=7,
            token=seed_token,
            position=seed_position,
            prev_hash=seed_prev_signed,
        )

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[222],
            positions=[1],
            out_cache_loc=[2],
            seed_slot_indices=[7],
            assert_equal=False,
        )

        # Step 2: verify slot[2] with prev=7 — expects no violation.
        verify_plan = make_verify_plan(
            slot_indices=[2], positions=[1], prev_slot_indices=[7], device=_DEVICE
        )
        verify_log = FakeViolationLog.allocate(device=_DEVICE)
        launch_canary_verify_kernel(
            context=VerifyOrWriteContext(
                canary_buf=cuda_buf,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=verify_log.ring,
                violation_write_index=verify_log.write_index,
                slot_run_counter=verify_log.slot_run_counter,
                kernel_run_counter=verify_log.kernel_run_counter,
                enable_chain_position_assert=verify_log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
            ),
            plan=verify_plan,
            check_verify_expected_token=True,
        )
        torch.cuda.synchronize()
        assert int(verify_log.write_index[0].item()) == 0

    def test_seed_slot_resume_5_step_hardcoded(self) -> None:
        cuda_buf = make_canary_buf(num_slots=50, slot_stride_bytes=32, device=_DEVICE)
        ref_buf = cuda_buf.clone()
        buf_pair = (cuda_buf, ref_buf)
        seed_token = 7
        seed_position = 10
        seed_prev_hash_signed = to_signed_int64(splitmix64(consts.CANARY_CHAIN_ANCHOR))
        stamp_pair(
            buf_pair,
            slot_idx=42,
            token=seed_token,
            position=seed_position,
            prev_hash=seed_prev_hash_signed,
        )

        predecessor_advance = splitmix64_mix3(
            splitmix64(consts.CANARY_CHAIN_ANCHOR), seed_token, seed_position
        )

        tokens = [101, 202, 303, 404, 505]
        positions = [11, 12, 13, 14, 15]
        real_kv = [0, 0, 0, 0, 0]
        out_cache_loc = [0, 1, 2, 3, 4]

        expected_prev_hashes: list[int] = []
        running = predecessor_advance
        for t, p, r in zip(tokens, positions, real_kv):
            expected_prev_hashes.append(running)
            running = splitmix64_mix3(running, t, p)

        cuda_log, _ = _run_write(
            buf_pair=buf_pair,
            input_ids=tokens,
            positions=positions,
            out_cache_loc=out_cache_loc,
            write_offsets=[0, 5],
            seed_slot_indices=[42],
        )

        for slot_idx, expected_token, expected_position, expected_prev_u64 in zip(
            out_cache_loc, tokens, positions, expected_prev_hashes
        ):
            stored_token, stored_position, stored_prev_hash, stored_real_kv_hash = (
                read_slot_fields(canary_buf=cuda_buf, slot_idx=slot_idx)
            )
            assert stored_token == expected_token
            assert stored_position == expected_position
            assert stored_prev_hash == to_signed_int64(expected_prev_u64)
            assert stored_real_kv_hash == 0

        assert int(cuda_log.write_index[0].item()) == 0

    def test_seed_continues_existing_chain(self) -> None:
        """Pre-stamp seed slot; subsequent write should continue chain from splitmix64_mix3(seed.*)."""
        seed_slot = 3
        seed_token = 7
        seed_position = 1
        seed_real_kv = 0
        expected_seed_prev_hash = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        stamp_pair(
            self.buf_pair,
            slot_idx=seed_slot,
            token=seed_token,
            position=seed_position,
            prev_hash=to_signed_int64(expected_seed_prev_hash),
            real_kv_hash=to_signed_int64(seed_real_kv),
        )

        new_slot = 4
        new_token = 13
        new_position = 2
        expected_running = splitmix64_mix3(
            expected_seed_prev_hash, seed_token, seed_position
        )

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[new_token],
            positions=[new_position],
            out_cache_loc=[new_slot],
            seed_slot_indices=[seed_slot],
        )

        new_stored = read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=new_slot)
        assert new_stored[0] == new_token
        assert new_stored[1] == new_position
        assert new_stored[2] == to_signed_int64(
            expected_running
        ), f"new slot prev_hash {new_stored[2]} != expected {to_signed_int64(expected_running)}"


class TestChain:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_chain_link_byte_equal_5_step(self) -> None:
        """5-step chain, buf / ring / counters byte-equal against ref."""
        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[10, 20, 30, 40, 50],
            positions=[0, 1, 2, 3, 4],
            out_cache_loc=[0, 1, 2, 3, 4],
        )

    def test_chain_link_byte_equal_5_step_hardcoded(self) -> None:
        """5-step write chain with hand-computed splitmix64 expected fields per slot."""
        tokens = [101, 202, 303, 404, 505]
        positions = [0, 1, 2, 3, 4]
        out_cache_loc = [0, 1, 2, 3, 4]
        real_kv_hashes = [0, 0, 0, 0, 0]

        # Step 1: compute the expected stored prev_hash sequence in pure Python via splitmix64.
        expected_prev_hashes_u64: list[int] = []
        running = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        for token, position, real_kv_hash in zip(tokens, positions, real_kv_hashes):
            expected_prev_hashes_u64.append(running)
            running = splitmix64_mix3(running, token, position)
        expected_prev_hashes_signed = [
            to_signed_int64(h) for h in expected_prev_hashes_u64
        ]

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=tokens,
            positions=positions,
            out_cache_loc=out_cache_loc,
        )

        # Step 2: verify every slot's stored 4 fields match the hardcoded expected sequence.
        for slot_idx, expected_token, expected_position, expected_prev_signed in zip(
            out_cache_loc, tokens, positions, expected_prev_hashes_signed
        ):
            stored_token, stored_position, stored_prev_hash, stored_real_kv_hash = (
                read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=slot_idx)
            )
            assert stored_token == expected_token
            assert stored_position == expected_position
            assert stored_prev_hash == expected_prev_signed
            assert stored_real_kv_hash == 0

    def test_chain_advances_with_real_kv_hash_all(self) -> None:
        """ALL mode + 2 sources + 5-step chain: stored prev_hash recoverable from seed."""
        cuda_buf = self.buf_pair[0]
        sources_cuda = make_real_kv_sources(
            count=2,
            num_bytes_per_token=16,
            page_size=1,
            num_slots=16,
            device=_DEVICE,
        )
        sources_ref = clone_real_kv_sources(sources_cuda)

        slot_indices = [1, 2, 3, 4, 5]
        tokens = [11, 22, 33, 44, 55]
        positions = [0, 1, 2, 3, 4]

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=tokens,
            positions=positions,
            out_cache_loc=slot_indices,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        running = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        for slot_idx, token, position in zip(slot_indices, tokens, positions):
            stored_prev_signed, stored_real_kv_hash = read_slot_fields(
                canary_buf=cuda_buf, slot_idx=slot_idx
            )[2:]
            assert stored_prev_signed == to_signed_int64(
                running
            ), f"slot {slot_idx}: stored prev_hash != recomputed chain step"
            running = splitmix64_mix3(running, token, position)


class TestMockMode:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_mock_mode_off_ignores_expected(self) -> None:
        """``enable_write_verify_inputs = OFF`` → expected tensors are ignored (we pass garbage to prove the kernel skips them)."""
        # Garbage expected tensors that, if the kernel mistakenly reads, would generate mismatches.
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[1, 2, 3],
            positions=[0, 1, 2],
            out_cache_loc=[0, 1, 2],
            expected_input_tokens=_int32_tensor([999, 999, 999]),
            expected_input_positions=_int32_tensor([999, 999, 999]),
        )

        assert int(cuda_log.write_index[0].item()) == 0

    def test_mock_mode_on_match_no_violation(self) -> None:
        """``enable_write_verify_inputs = ON`` and expected matches actual → no violation, chain advances."""
        input_ids = _int32_tensor([7, 8, 9])
        positions = _int32_tensor([0, 1, 2])
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=[0, 1, 2],
            enable_write_verify_inputs=True,
            expected_input_tokens=input_ids.clone(),
            expected_input_positions=positions.clone(),
        )

        assert int(cuda_log.write_index[0].item()) == 0

    def test_mock_mode_on_token_mismatch_records_violation(self) -> None:
        """``enable_write_verify_inputs = ON`` token mismatch → violation recorded; chain advances on ACTUAL token."""
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[42],
            positions=[0],
            out_cache_loc=[0],
            enable_write_verify_inputs=True,
            expected_input_tokens=_int32_tensor([99]),
            expected_input_positions=_int32_tensor([0]),
        )

        fail_bits = int(
            cuda_log.ring[0, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item()
        )
        assert_only_bits_set(fail_bits, consts.FailReason.WRITE_TOKEN_MISMATCH)
        # Chain advances on actual (42), not expected (99). Stored token should be 42.
        stored_token, _, _, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=0
        )
        assert stored_token == 42

    def test_mock_mode_on_position_mismatch_records_violation(self) -> None:
        """``enable_write_verify_inputs = ON`` position mismatch → violation recorded; chain advances on ACTUAL position."""
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[42],
            positions=[7],
            out_cache_loc=[0],
            enable_write_verify_inputs=True,
            expected_input_tokens=_int32_tensor([42]),
            expected_input_positions=_int32_tensor([0]),
        )

        fail_bits = int(
            cuda_log.ring[0, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item()
        )
        assert_only_bits_set(fail_bits, consts.FailReason.WRITE_POSITION_MISMATCH)
        _, stored_position, _, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=0
        )
        assert stored_position == 7

    def test_mock_mode_chain_advances_on_actual_not_expected(self) -> None:
        """Expected differs from actual on every entry → downstream verify must NOT cascade chain errors."""
        cuda_buf = self.buf_pair[0]
        # Every actual differs from expected.
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            out_cache_loc=[1, 2, 3],
            enable_write_verify_inputs=True,
            expected_input_tokens=_int32_tensor([999, 999, 999]),
            expected_input_positions=_int32_tensor([999, 999, 999]),
        )

        # All 3 entries should fire a violation row.
        assert int(cuda_log.write_index[0].item()) == 3
        # Run a downstream verify — it must see no chain mismatch because chain advanced on actuals.
        verify_plan = make_verify_plan(
            slot_indices=[1, 2, 3],
            positions=[0, 1, 2],
            prev_slot_indices=[-1, 1, 2],
            device=_DEVICE,
        )
        verify_log = FakeViolationLog.allocate(device=_DEVICE)
        launch_canary_verify_kernel(
            context=VerifyOrWriteContext(
                canary_buf=cuda_buf,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=verify_log.ring,
                violation_write_index=verify_log.write_index,
                slot_run_counter=verify_log.slot_run_counter,
                kernel_run_counter=verify_log.kernel_run_counter,
                enable_chain_position_assert=verify_log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
            ),
            plan=verify_plan,
            check_verify_expected_token=True,
        )
        torch.cuda.synchronize()
        assert int(verify_log.write_index[0].item()) == 0

    @pytest.mark.parametrize("bit_to_trigger", ["MOCK_TOKEN", "MOCK_POSITION"])
    @pytest.mark.parametrize("injection_position", ["head", "mid", "last"])
    def test_mock_violation_bit_injection_position_matrix(
        self,
        bit_to_trigger: str,
        injection_position: str,
    ) -> None:
        """Sweep injection_position x bit_to_trigger for write-kernel pseudo-mode fail-reason coverage."""
        slot_count = 5
        tokens = [10, 20, 30, 40, 50]
        positions = [0, 1, 2, 3, 4]
        out_cache_locs = [0, 1, 2, 3, 4]
        corruption_index = {"head": 0, "mid": 2, "last": 4}[injection_position]
        corrupt_slot = out_cache_locs[corruption_index]

        expected_bit = {
            "MOCK_TOKEN": consts.FailReason.WRITE_TOKEN_MISMATCH,
            "MOCK_POSITION": consts.FailReason.WRITE_POSITION_MISMATCH,
        }[bit_to_trigger]

        input_ids = _int32_tensor(tokens)
        positions_t = _int32_tensor(positions)
        out_cache_loc = _int32_tensor(out_cache_locs)

        pseudo_tokens = input_ids.clone()
        pseudo_positions = positions_t.clone()

        if bit_to_trigger == "MOCK_TOKEN":
            pseudo_tokens[corruption_index] = tokens[corruption_index] + 999
        else:
            pseudo_positions[corruption_index] = positions[corruption_index] + 99

        cuda_log, ref_log = _run_write(
            buf_pair=self.buf_pair,
            input_ids=input_ids,
            positions=positions_t,
            out_cache_loc=out_cache_loc,
            write_offsets=[0, slot_count],
            enable_write_verify_inputs=True,
            expected_input_tokens=pseudo_tokens,
            expected_input_positions=pseudo_positions,
            assert_equal=False,
        )

        found = False
        for row_idx in range(int(cuda_log.write_index[0].item())):
            fail_bits = int(
                cuda_log.ring[row_idx, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item()
            )
            row_slot = int(cuda_log.ring[row_idx, 1].item())
            if (fail_bits & expected_bit) and row_slot == corrupt_slot:
                found = True
                break
        assert found, (
            f"expected bit {expected_bit:#x} at slot {corrupt_slot} not found in ring "
            f"(bit_to_trigger={bit_to_trigger} injection_position={injection_position})"
        )
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


class TestSlotHandling:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_negative_slot_skips_entry(self) -> None:
        """``out_cache_loc[i] < 0`` → that entry is skipped: no buf write, no violation, no
        canary slot mutation, and no write slot_run_counter increment.
        Covers both SWA out-of-window (after caller-side LUT gather) and explicit padding intents.
        """
        # Two entries: first writes to slot 4 normally; second has slot=-1 and must be skipped.
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[42, 99],
            positions=[0, 1],
            out_cache_loc=[4, -1],
        )

        stored_token, _, _, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=4
        )
        assert stored_token == 42
        assert int(cuda_log.slot_run_counter.item()) == 1

    def test_pre_translated_slot_writes_normally(self) -> None:
        """``out_cache_loc[i] >= 0`` → the kernel writes to exactly that slot, with no LUT applied. This
        confirms the kernel is SWA-agnostic: SWA endpoints feed the same shape of input here after their
        host-side gather, so the contract is symmetric across FULL / SWA groups.
        """
        # Slot 4 here could equally be a FULL-group raw out_cache_loc value, or the result of an SWA
        # endpoint's host gather. The kernel can't tell the difference and that's the point.
        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[55],
            positions=[0],
            out_cache_loc=[4],
        )

        stored_token, _, _, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=4
        )
        assert stored_token == 55

    def test_padding_block_skipped(self) -> None:
        """``blockIdx.x >= write_num_valid_reqs[0]`` → block early-exits, no write to canary_buf."""
        # Allocate plan with req_capacity=4 but only declare 1 active req.
        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[1],
            positions=[0],
            out_cache_loc=[0],
            req_capacity=4,
        )

        # Only slot 0 should have been written; padding blocks 1..3 must not touch the buffer.
        stored_token, _, _, _ = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=0
        )
        assert stored_token == 1
        for slot_idx in (1, 2, 3):
            stored_token_other, _, _, _ = read_slot_fields(
                canary_buf=self.buf_pair[0], slot_idx=slot_idx
            )
            assert stored_token_other == 0

    def test_write_skip_when_out_cache_loc_is_minus_one(self) -> None:
        """out_cache_loc[i] = -1 → that entry's slot is untouched by write kernel."""
        cuda_buf = self.buf_pair[0]
        cuda_buf_before_slot_view = cuda_buf.view(torch.int64).clone()

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[100, 200, 300],
            positions=[0, 1, 2],
            out_cache_loc=[5, -1, 7],
        )

        after = cuda_buf.view(torch.int64)
        for slot in range(cuda_buf.shape[0]):
            if slot in (5, 7):
                continue
            assert torch.equal(
                after[slot], cuda_buf_before_slot_view[slot]
            ), f"slot {slot} should not have been written"

    def test_shrink_active_reqs_does_not_write_stale_slots(self) -> None:
        """Run write with bs=3 plan after a bs=8 run on same buffer: stale slots from bs=8 stay intact."""
        cuda_buf = make_canary_buf(num_slots=32, slot_stride_bytes=32, device=_DEVICE)
        ref_buf = cuda_buf.clone()
        buf_pair = (cuda_buf, ref_buf)

        big_slots = list(range(1, 9))
        _run_write(
            buf_pair=buf_pair,
            input_ids=list(range(100, 108)),
            positions=[0] * 8,
            out_cache_loc=big_slots,
            write_offsets=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            seed_slot_indices=[-1] * 8,
            num_valid_reqs=8,
            assert_equal=False,
        )

        untouched_snapshot = cuda_buf.view(torch.int64).clone()

        small_slots = [20, 21, 22]
        _run_write(
            buf_pair=buf_pair,
            input_ids=[7, 8, 9],
            positions=[0, 0, 0],
            out_cache_loc=small_slots,
            write_offsets=[0, 1, 2, 3],
            seed_slot_indices=[-1, -1, -1],
            num_valid_reqs=3,
            assert_equal=False,
        )

        after = cuda_buf.view(torch.int64)
        for slot in big_slots:
            assert torch.equal(
                after[slot], untouched_snapshot[slot]
            ), f"slot {slot} from earlier bs=8 run was overwritten by bs=3 run"


class TestRealKvHash:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_real_kv_mode_off_writes_zero(self) -> None:
        """``consts.RealKvHashMode.NONE`` → ``real_kv_hash`` field is written as 0 regardless of source presence."""
        sources = make_real_kv_sources(count=2, device=_DEVICE)

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[1, 2],
            positions=[0, 1],
            out_cache_loc=[0, 1],
            real_kv_sources_pair=(sources, sources),
        )

        _, _, _, real_kv_0 = read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=0)
        _, _, _, real_kv_1 = read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=1)
        assert real_kv_0 == 0
        assert real_kv_1 == 0

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(consts.RealKvHashMode.PARTIAL, id="partial"),
            pytest.param(consts.RealKvHashMode.ALL, id="all"),
        ],
    )
    def test_real_kv_mode_byte_equal(self, mode: consts.RealKvHashMode) -> None:
        """PARTIAL / ALL modes both produce CUDA-vs-ref byte-equal write state on a 3-entry chain."""
        sources_cuda = make_real_kv_sources(count=2, device=_DEVICE)
        sources_ref = clone_real_kv_sources(sources_cuda)

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            out_cache_loc=[0, 1, 2],
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=mode,
        )

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    def test_real_kv_sources_fold_1_to_4(self, count: int) -> None:
        """Folding ``count`` sources sequentially → CUDA matches ref for every count in {1..4}."""
        sources_cuda = make_real_kv_sources(count=count, device=_DEVICE)
        sources_ref = clone_real_kv_sources(sources_cuda)

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[1, 2],
            positions=[0, 1],
            out_cache_loc=[0, 1],
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

    def test_real_kv_source_above_4_raises(self) -> None:
        """``len(real_kv_sources) > 4`` → host wrapper raises ValueError before launching."""
        cuda_buf = make_canary_buf(device=_DEVICE)
        plan = make_write_plan(
            write_offsets=[0, 1],
            seed_slot_indices=[-1],
            num_valid_reqs=1,
            device=_DEVICE,
        )
        input_ids = _int32_tensor([1])
        positions = _int32_tensor([0])
        out_cache_loc = _int32_tensor([0])
        log = FakeViolationLog.allocate(device=_DEVICE)
        sources = make_real_kv_sources(count=4, device=_DEVICE)
        extra = make_real_kv_source(device=_DEVICE)
        too_many = sources + (extra,)

        with pytest.raises(ValueError, match="at most 4 RealKvSource"):
            launch_canary_write_kernel(
                context=VerifyOrWriteContext(
                    canary_buf=cuda_buf,
                    kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                    violation_ring=log.ring,
                    violation_write_index=log.write_index,
                    slot_run_counter=log.slot_run_counter,
                    kernel_run_counter=log.kernel_run_counter,
                    enable_chain_position_assert=log.enable_chain_position_assert,
                    real_kv_sources=too_many,
                    real_kv_hash_mode=consts.RealKvHashMode.NONE,
                ),
                plan=plan,
                input_ids=input_ids,
                positions=positions,
                out_cache_loc=out_cache_loc,
                enable_write_input_assert=False,
                expected_input_tokens=None,
                expected_input_positions=None,
            )

    @pytest.mark.parametrize(
        "mode,fold_fn,expected_hash",
        [
            pytest.param(
                consts.RealKvHashMode.PARTIAL,
                _hand_fold_partial,
                0x6041580849E6407D,
                id="partial",
            ),
            pytest.param(
                consts.RealKvHashMode.ALL,
                _hand_fold_all,
                0x6041580849E6407D,
                id="all",
            ),
        ],
    )
    def test_real_kv_hash_fold_mode_writes_expected_hash_hardcoded(
        self,
        mode: consts.RealKvHashMode,
        fold_fn: Callable[[bytes], int],
        expected_hash: int,
    ) -> None:
        # Step 1: build one RealKvSource with read_bytes=16 and a fixed byte pattern at slot 0.
        _PATTERN = bytes(
            [
                0x01,
                0x02,
                0x04,
                0x08,
                0x10,
                0x20,
                0x40,
                0x80,
                0x81,
                0x82,
                0x84,
                0x88,
                0x90,
                0xA0,
                0xC0,
                0xFF,
            ]
        )

        # Step 2: verify hand-computed fold matches the hex literal.
        assert fold_fn(_PATTERN) == expected_hash

        source_cuda = make_real_kv_source(
            num_slots=16,
            num_bytes_per_token=16,
            page_size=1,
            read_bytes=16,
            device=_DEVICE,
        )
        source_cuda.tensor[0, :16] = torch.tensor(list(_PATTERN), dtype=torch.uint8)
        source_ref = RealKvSource(
            tensor=source_cuda.tensor.clone(),
            page_size=source_cuda.page_size,
            num_bytes_per_token=source_cuda.num_bytes_per_token,
            read_bytes=source_cuda.read_bytes,
        )

        # Step 3: run write kernel on slot 0 with the given mode.
        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[7],
            positions=[0],
            out_cache_loc=[0],
            real_kv_sources_pair=((source_cuda,), (source_ref,)),
            real_kv_hash_mode=mode,
        )

        # Step 4: assert stored real_kv_hash equals the hand-computed hex literal.
        _, _, _, stored_real_kv_hash = read_slot_fields(
            canary_buf=self.buf_pair[0], slot_idx=0
        )
        assert stored_real_kv_hash == to_signed_int64(
            expected_hash
        ), f"stored_real_kv_hash={stored_real_kv_hash:#x} expected={to_signed_int64(expected_hash):#x}"

    def test_paged_real_kv_hash_consistent_across_slots(self) -> None:
        """page=16: writing two slots inside same page yields independent real_kv_hash per slot."""
        sources_cuda = make_real_kv_sources(
            count=1,
            num_bytes_per_token=16,
            page_size=16,
            num_slots=16,
            device=_DEVICE,
        )
        pattern_slot3 = bytes(range(1, 17))
        pattern_slot7 = bytes(range(101, 117))
        sources_cuda[0].tensor[0, 3 * 16 : 4 * 16] = torch.tensor(
            list(pattern_slot3), dtype=torch.uint8, device=_DEVICE
        )
        sources_cuda[0].tensor[0, 7 * 16 : 8 * 16] = torch.tensor(
            list(pattern_slot7), dtype=torch.uint8, device=_DEVICE
        )
        sources_ref = clone_real_kv_sources(sources_cuda)

        _run_write(
            buf_pair=self.buf_pair,
            input_ids=[42, 84],
            positions=[0, 1],
            out_cache_loc=[3, 7],
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        slot3 = read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=3)
        slot7 = read_slot_fields(canary_buf=self.buf_pair[0], slot_idx=7)
        assert slot3[3] == to_signed_int64(_hand_fold_all(pattern_slot3))
        assert slot7[3] == to_signed_int64(_hand_fold_all(pattern_slot7))
        assert slot3[3] != slot7[3]

    def test_multi_source_real_kv_fold_order_matters(self) -> None:
        """Two sources folded in reverse order yields a different real_kv_hash (fold is ordered)."""
        sources_a = make_real_kv_sources(
            count=2, num_bytes_per_token=16, num_slots=8, device=_DEVICE
        )
        sources_b = tuple(reversed(sources_a))

        def _run_with(srcs: tuple[RealKvSource, ...]) -> tuple[int, int, int, int]:
            buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
            plan = make_write_plan(
                write_offsets=[0, 1],
                seed_slot_indices=[-1],
                num_valid_reqs=1,
                device=_DEVICE,
            )
            log = FakeViolationLog.allocate(device=_DEVICE)
            launch_canary_write_kernel(
                context=VerifyOrWriteContext(
                    canary_buf=buf,
                    kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                    violation_ring=log.ring,
                    violation_write_index=log.write_index,
                    slot_run_counter=log.slot_run_counter,
                    kernel_run_counter=log.kernel_run_counter,
                    enable_chain_position_assert=log.enable_chain_position_assert,
                    real_kv_sources=srcs,
                    real_kv_hash_mode=consts.RealKvHashMode.ALL,
                ),
                plan=plan,
                input_ids=_int32_tensor([1]),
                positions=_int32_tensor([0]),
                out_cache_loc=_int32_tensor([2]),
                enable_write_input_assert=False,
                expected_input_tokens=None,
                expected_input_positions=None,
            )
            torch.cuda.synchronize()
            return read_slot_fields(canary_buf=buf, slot_idx=2)

        fields_a = _run_with(sources_a)
        fields_b = _run_with(sources_b)
        assert fields_a[3] != 0
        assert fields_b[3] != 0
        assert (
            fields_a[3] != fields_b[3]
        ), "reversing source order must change real_kv_hash (fold is ordered)"


class TestRunCounter:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_kernel_run_counter_per_call(self) -> None:
        """``kernel_run_counter`` increments by 1 per call (even when ``write_num_valid_reqs == 0``)."""
        plan_pair = make_write_plan_pair(
            write_offsets=[0, 0],
            seed_slot_indices=[-1],
            num_valid_reqs=0,
            device=_DEVICE,
        )
        input_ids = _int32_tensor([0])
        positions = _int32_tensor([0])
        out_cache_loc = _int32_tensor([0])
        pseudo_tokens, pseudo_positions = dummy_pseudo_tensors(1)
        cuda_log, ref_log = make_log_pair(device=_DEVICE)

        for _ in range(3):
            _run_both_write(
                cuda_canary_buf=self.buf_pair[0],
                ref_canary_buf=self.buf_pair[1],
                plan_cuda=plan_pair[0],
                plan_ref=plan_pair[1],
                input_ids=input_ids,
                positions=positions,
                out_cache_loc=out_cache_loc,
                enable_write_verify_inputs=False,
                expected_input_tokens=pseudo_tokens,
                expected_input_positions=pseudo_positions,
                cuda_log=cuda_log,
                ref_log=ref_log,
                real_kv_sources_cuda=(),
                real_kv_sources_ref=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
                assert_equal=False,
            )

        assert int(cuda_log.kernel_run_counter[0].item()) == 3
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)

    def test_slot_run_counter_sums_entries(self) -> None:
        """``slot_run_counter`` += sum(entry_count) across all active reqs in this call."""
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[1, 2, 3, 4, 5],
            positions=[0, 1, 0, 1, 2],
            out_cache_loc=[0, 1, 2, 3, 4],
            write_offsets=[0, 2, 5],
            seed_slot_indices=[-1, -1],
            num_valid_reqs=2,
        )

        assert int(cuda_log.slot_run_counter[0].item()) == 5


class TestMisc:
    def test_empty_plan_no_op(self) -> None:
        """``write_num_valid_reqs = 0`` → no buf write, no slot_run_counter bump, only kernel_run_counter += 1."""
        cuda_log, _ = _run_write(
            buf_pair=_make_default_buf_pair(),
            input_ids=[0],
            positions=[0],
            out_cache_loc=[0],
            write_offsets=[0, 0],
            num_valid_reqs=0,
            req_capacity=4,
        )

        assert int(cuda_log.write_index[0].item()) == 0
        assert int(cuda_log.slot_run_counter[0].item()) == 0
        assert int(cuda_log.kernel_run_counter[0].item()) == 1

    def test_disabled_input_verify_does_not_deref_expected_inputs(self) -> None:
        """``enable_write_verify_inputs=False`` must short-circuit before any expected_input_* dereference;
        poison those tensors with 0x7F7F7F7F and assert nothing records a violation."""
        cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
        ref_buf = cuda_buf.clone()
        buf_pair = (cuda_buf, ref_buf)

        garbage_expected_tokens = torch.full(
            (1,), 0x7F7F7F7F, dtype=torch.int64, device=_DEVICE
        )
        garbage_expected_positions = torch.full(
            (1,), 0x7F7F7F7F, dtype=torch.int64, device=_DEVICE
        )

        cuda_log, ref_log = _run_write(
            buf_pair=buf_pair,
            input_ids=[42],
            positions=[0],
            out_cache_loc=[0],
            expected_input_tokens=garbage_expected_tokens,
            expected_input_positions=garbage_expected_positions,
            assert_equal=False,
        )

        assert int(cuda_log.write_index[0].item()) == 0
        assert int(ref_log.write_index[0].item()) == 0

    def test_disabled_assert_inputs_passes_none_to_cuda(self) -> None:
        """Disabled input assertions pass None instead of dummy tensors."""
        canary_buf = torch.zeros(
            4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=_DEVICE
        )
        plan = make_write_plan(
            write_offsets=[0, 0],
            seed_slot_indices=[-1],
            num_valid_reqs=0,
            device=_DEVICE,
        )
        input_ids = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
        positions = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
        out_cache_loc = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
        log = FakeViolationLog.allocate(capacity=2, device=_DEVICE)
        context = VerifyOrWriteContext(
            canary_buf=canary_buf,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=log.ring,
            violation_write_index=log.write_index,
            slot_run_counter=log.slot_run_counter,
            kernel_run_counter=log.kernel_run_counter,
            enable_chain_position_assert=log.enable_chain_position_assert,
            real_kv_sources=(),
            real_kv_hash_mode=consts.RealKvHashMode.NONE,
        )
        module = _RecordingWriteModule()

        with patch.object(write_module, "_jit_canary_write_module", lambda: module):
            launch_canary_write_kernel(
                context=context,
                plan=plan,
                input_ids=input_ids,
                positions=positions,
                out_cache_loc=out_cache_loc,
                enable_write_input_assert=False,
                expected_input_tokens=None,
                expected_input_positions=None,
            )

        assert len(module.calls) == 1
        call = module.calls[0]
        assert call[8] == 0
        assert call[9] is None
        assert call[10] is None


class TestBoundarySweep:
    @pytest.mark.parametrize(
        "token_val",
        [0, 1, 0xFFFFFFFF, -1, 0x80000000, 0x7FFFFFFF],
    )
    def test_token_boundary_byte_equal_sweep(self, token_val: int) -> None:
        """Sweep token boundary values; assert CUDA write vs ref buf + state byte-equal."""
        _run_write_single_slot_byte_equal(_WriteSingleSlotInput(token=token_val))

    @pytest.mark.parametrize(
        "position_val",
        [0, 1, 127, 128, 129, 0x7FFFFFFF],
    )
    def test_position_boundary_byte_equal_sweep(self, position_val: int) -> None:
        """Sweep position boundary values; assert CUDA write vs ref buf + state byte-equal."""
        _run_write_single_slot_byte_equal(_WriteSingleSlotInput(position=position_val))


class TestPseudoMode:
    def setup_method(self) -> None:
        self.buf_pair = _make_default_buf_pair()

    def test_pseudo_mode_on_catches_token_mismatch(self) -> None:
        """enable_write_verify_inputs=ON + intentional token mismatch → WRITE_TOKEN_MISMATCH bit recorded."""
        input_ids = _int32_tensor([10, 20, 30, 40, 50])
        positions = _int32_tensor([0, 1, 2, 3, 4])
        out_cache_loc = _int32_tensor([1, 2, 3, 4, 5])
        pseudo_tokens = _int32_tensor([10, 20, 30, 999, 50])
        pseudo_positions = positions.clone()

        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            enable_write_verify_inputs=True,
            expected_input_tokens=pseudo_tokens,
            expected_input_positions=pseudo_positions,
        )
        assert int(cuda_log.write_index[0].item()) >= 1
        bits = int(cuda_log.ring[0, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item())
        assert (
            bits & consts.FailReason.WRITE_TOKEN_MISMATCH
        ), f"expected WRITE_TOKEN_MISMATCH bit, got {bits:#b}"

    def test_pseudo_mode_off_skips_token_check(self) -> None:
        """enable_write_verify_inputs=False makes the caller pass no expected-input tensors."""
        cuda_log, _ = _run_write(
            buf_pair=self.buf_pair,
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            out_cache_loc=[1, 2, 3],
            expected_input_tokens=_int32_tensor([99, 99, 99]),
            expected_input_positions=_int32_tensor([99, 99, 99]),
        )
        assert int(cuda_log.write_index[0].item()) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
