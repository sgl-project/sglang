"""Host->device staging of the DFLASH decode plan (DFlashDraftInputV2).

The plan rows (page-aligned KV bounds + mamba track positions) live in one
per-device ring of pinned buffers and reach the device in a single copy that
is skipped on rounds where no consumed row changed. These tests pin the dirty
predicate (alloc round / track flip / request-set change / steady round), the
ring and its views, the invariant-check mode that always copies and flags a
missed writer, the non-lazy track staging (device row == reqs-derived plan,
no host tensor built at verify prep), the verify-prep wiring, and the graph
runner's fast-path skip of token copies the pre-plan already made.
"""

import weakref
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
import torch
from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    mamba_track_positions_from_reqs,
    set_mamba_track_indices_from_reqs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    _StagedTokenInputs,
)
from sglang.srt.speculative import dflash_info_v2, spec_utils
from sglang.srt.speculative.dflash_info_v2 import (
    DecodePlanStaging,
    spec_track_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(10, "base-a-test-cpu")


def _exec(extra_buffer: bool, lazy: bool = False) -> SimpleNamespace:
    """The runtime-context slice the plan reads for the mamba strategy."""
    return SimpleNamespace(
        mamba=SimpleNamespace(
            enable_mamba_extra_buffer=extra_buffer, enable_mamba_extra_buffer_lazy=lazy
        )
    )


DEVICE = torch.device("cpu")
POOL = 8


def _level(level: InvariantCheckLevel):
    return envs.SGLANG_INVARIANT_CHECK.override(int(level))


def _fill(staging: DecodePlanStaging, cur: list[int], nxt: list[int], track: list[int]):
    host = staging.acquire(len(cur))
    host.seq_lens.copy_(torch.tensor(cur, dtype=torch.int64))
    host.cur_kv_lens.copy_(torch.tensor(cur, dtype=torch.int32))
    host.nxt_kv_lens.copy_(torch.tensor(nxt, dtype=torch.int32))
    host.track_positions.copy_(torch.tensor(track, dtype=torch.int32))
    return host


def _lens_gpu(staging: DecodePlanStaging) -> torch.Tensor:
    assert staging.lens_gpu is not None
    return staging.lens_gpu


class TestDirtyPredicate:
    def test_alloc_track_reqset_bs_copy_and_steady_round_skips(self) -> None:
        staging = DecodePlanStaging(DEVICE)
        with _level(InvariantCheckLevel.OFF):
            # First round: device buffer is fresh -> copy regardless of rows.
            _fill(staging, [16], [32], [0])
            dev = staging.ship(
                bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",)
            )
            assert staging.h2d_copy_ct == 1
            assert _lens_gpu(staging)[:, 0].tolist() == [16, 32, 0]

            # Steady round: KV bounds advance on the host but nothing consumes
            # them on device (num_needed_tokens == 0) -> no copy, device rows
            # 0/1 intentionally stay at the last shipped values.
            _fill(staging, [24], [32], [0])
            staging.ship(bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",))
            assert staging.h2d_copy_ct == 1
            assert _lens_gpu(staging)[:, 0].tolist() == [16, 32, 0]

            # Alloc round: the device KV bounds are read -> copy.
            _fill(staging, [32], [64], [0])
            dev = staging.ship(
                bs=1, num_needed_tokens=32, track_positions=[0], req_ids=("a",)
            )
            assert staging.h2d_copy_ct == 2
            assert dev.cur_kv_lens.tolist() == [32]
            assert dev.nxt_kv_lens.tolist() == [64]

            # Track flip (a mamba_track_grid crossing, not an alloc round) -> copy.
            _fill(staging, [40], [64], [1])
            dev = staging.ship(
                bs=1, num_needed_tokens=0, track_positions=[1], req_ids=("a",)
            )
            assert staging.h2d_copy_ct == 3
            assert dev.track_positions is not None
            assert dev.track_positions.tolist() == [1]

            # Same rows, different request in the slot -> copy.
            _fill(staging, [40], [64], [1])
            staging.ship(bs=1, num_needed_tokens=0, track_positions=[1], req_ids=("b",))
            assert staging.h2d_copy_ct == 4

            # Batch grows (merge) -> copy, views resliced to the new bs.
            _fill(staging, [40, 8], [64, 32], [1, 0])
            dev = staging.ship(
                bs=2, num_needed_tokens=0, track_positions=[1, 0], req_ids=("b", "c")
            )
            assert staging.h2d_copy_ct == 5
            assert dev.cur_kv_lens.tolist() == [40, 8]
            assert dev.track_positions is not None
            assert dev.track_positions.tolist() == [1, 0]

            # Steady again at bs=2 -> no copy.
            _fill(staging, [48, 16], [64, 32], [1, 0])
            staging.ship(
                bs=2, num_needed_tokens=0, track_positions=[1, 0], req_ids=("b", "c")
            )
            assert staging.h2d_copy_ct == 5
            assert staging.stage_ct == 7

    def test_no_mamba_rows_ship_without_track_view(self) -> None:
        staging = DecodePlanStaging(DEVICE)
        with _level(InvariantCheckLevel.OFF):
            _fill(staging, [16], [32], [0])
            dev = staging.ship(
                bs=1, num_needed_tokens=0, track_positions=None, req_ids=("a",)
            )
            assert dev.track_positions is None
            _fill(staging, [24], [32], [0])
            staging.ship(
                bs=1, num_needed_tokens=0, track_positions=None, req_ids=("a",)
            )
            assert staging.h2d_copy_ct == 1


class TestRingAndCapacity:
    def test_ring_alternates_slots_and_keeps_last_round_views(self) -> None:
        staging = DecodePlanStaging(DEVICE)
        first = _fill(staging, [1], [2], [0])
        second = _fill(staging, [3], [4], [1])
        assert first.cur_kv_lens.data_ptr() != second.cur_kv_lens.data_ptr()
        # Writing the second slot leaves the first round's views intact: the
        # result of round N is processed after round N+1 was prepared.
        assert first.cur_kv_lens.tolist() == [1]
        assert first.seq_lens.tolist() == [1]
        third = _fill(staging, [5], [6], [0])
        assert third.cur_kv_lens.data_ptr() == first.cur_kv_lens.data_ptr()
        assert DecodePlanStaging.RING_DEPTH == 2

    def test_reacquiring_an_undrained_slot_fails_loudly(self) -> None:
        # Slot events exist only under invariant checking (ship records them
        # there); the ring's depth-1 argument is what the default path relies on.
        staging = DecodePlanStaging(DEVICE)
        _fill(staging, [1], [2], [0])
        pending = SimpleNamespace(query=lambda: False)
        staging._slot_copied[staging._slot] = cast(torch.Event, pending)
        _fill(staging, [3], [4], [1])  # the other slot: fine
        with pytest.raises(AssertionError, match="before its H2D drained"):
            staging.acquire(1)
        pending.query = lambda: True
        staging.acquire(1)

    def test_capacity_growth_reallocates_and_reships(self) -> None:
        staging = DecodePlanStaging(DEVICE)
        with _level(InvariantCheckLevel.OFF):
            _fill(staging, [1], [2], [0])
            staging.ship(bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",))
            assert staging.capacity == 32
            bs = 40
            cur = list(range(bs))
            nxt = [c + 8 for c in cur]
            track = [c % 2 for c in cur]
            _fill(staging, cur, nxt, track)
            dev = staging.ship(
                bs=bs,
                num_needed_tokens=0,
                track_positions=track,
                req_ids=tuple(str(i) for i in range(bs)),
            )
            assert staging.capacity == 64
            assert _lens_gpu(staging).shape == (3, 64)
            assert staging.h2d_copy_ct == 2
            assert dev.cur_kv_lens.tolist() == cur
            assert dev.nxt_kv_lens.tolist() == nxt
            assert dev.track_positions is not None
            assert dev.track_positions.tolist() == track


class TestInvariantCheckMode:
    def test_always_copies_and_stays_quiet_when_rows_agree(self) -> None:
        staging = DecodePlanStaging(DEVICE)
        with _level(InvariantCheckLevel.STRICT):
            _fill(staging, [16], [32], [0])
            staging.ship(bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",))
            _fill(staging, [24], [32], [0])
            staging.ship(bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",))
            assert staging.h2d_copy_ct == 2
            # Rows 0/1 follow the host every round in this mode.
            assert _lens_gpu(staging)[0, 0].item() == 24

    def test_missed_writer_fails_loudly_in_check_mode_only(self) -> None:
        # dflash.plan_staging_clean injection: the host track row moves while
        # the predicate is told the plan is unchanged.
        for level, raises in (
            (InvariantCheckLevel.STRICT, True),
            (InvariantCheckLevel.WARN, True),
            (InvariantCheckLevel.OFF, False),
        ):
            staging = DecodePlanStaging(DEVICE)
            with _level(level):
                _fill(staging, [16], [32], [0])
                staging.ship(
                    bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",)
                )
                _fill(staging, [16], [32], [1])
                if raises:
                    with pytest.raises(RuntimeError, match="plan_staging_clean"):
                        staging.ship(
                            bs=1,
                            num_needed_tokens=0,
                            track_positions=[0],
                            req_ids=("a",),
                        )
                else:
                    staging.ship(
                        bs=1, num_needed_tokens=0, track_positions=[0], req_ids=("a",)
                    )
                    assert staging.h2d_copy_ct == 1


def _req(rid: str, next_track_idx: int | None) -> Req:
    return cast(
        Req,
        SimpleNamespace(
            rid=rid, kv=SimpleNamespace(mamba_next_track_idx=next_track_idx)
        ),
    )


def _mapping() -> torch.Tensor:
    return torch.arange(POOL * 2, dtype=torch.int64).reshape(POOL, 2) * 10


class TestNonLazyTrackStaging:
    def test_positions_from_reqs_default_unallocated_to_slot0(self) -> None:
        reqs = [_req("a", 1), _req("b", None), _req("c", 0)]
        assert mamba_track_positions_from_reqs(reqs) == [1, 0, 0]

    def test_staged_row_gathers_the_same_slots_without_a_host_tensor(self) -> None:
        reqs = [_req("a", 1), _req("b", None), _req("c", 0)]
        rpi = torch.tensor([3, 5, 1], dtype=torch.int64)
        mapping = _mapping()
        batch = cast(
            ScheduleBatch,
            SimpleNamespace(
                reqs=reqs,
                req_pool_indices=rpi,
                req_to_token_pool=SimpleNamespace(
                    req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
                ),
                mamba_track_buffer_indices=None,
                mamba_track_indices=None,
            ),
        )
        positions = mamba_track_positions_from_reqs(reqs)
        staging = DecodePlanStaging(DEVICE)
        with _level(InvariantCheckLevel.OFF):
            _fill(staging, [16, 16, 16], [32, 32, 32], positions)
            dev = staging.ship(
                bs=3,
                num_needed_tokens=0,
                track_positions=positions,
                req_ids=("a", "b", "c"),
            )
        # The verify prep must not rebuild the positions host-side (that is the
        # per-step pinned alloc + H2D this staging replaces).
        with mock.patch.object(
            torch, "tensor", side_effect=AssertionError("host tensor built")
        ):
            set_mamba_track_indices_from_reqs(batch, None, dev.track_positions)
        expected = mapping[rpi, torch.tensor(positions)]
        assert batch.mamba_track_indices.tolist() == expected.tolist()
        assert batch.mamba_track_indices.dtype == torch.int64
        assert batch.mamba_track_buffer_indices == positions

    def test_spec_track_positions_by_mode(self) -> None:
        reqs = [_req("a", 1), _req("b", None)]
        batch = cast(
            ScheduleBatch,
            SimpleNamespace(reqs=reqs, mamba_lazy_spec_track_positions_cpu=None),
        )
        with mock.patch.object(dflash_info_v2, "get_exec", return_value=_exec(False)):
            assert spec_track_positions(batch) is None
        with mock.patch.object(dflash_info_v2, "get_exec", return_value=_exec(True)):
            assert spec_track_positions(batch) == [1, 0]
        with mock.patch.object(
            dflash_info_v2, "get_exec", return_value=_exec(True, lazy=True)
        ):
            with pytest.raises(AssertionError, match="mamba_lazy_spec_prepare"):
                spec_track_positions(batch)
            batch.mamba_lazy_spec_track_positions_cpu = [0, 1]
            assert spec_track_positions(batch) == [0, 1]

    def test_verify_prep_hands_the_staged_row_to_the_index_builder(self) -> None:
        staged = torch.tensor([1, 0], dtype=torch.int32)
        for lazy, plan in ((False, None), (True, [1, 0])):
            batch = cast(
                ScheduleBatch,
                SimpleNamespace(
                    reqs=[_req("a", 1), _req("b", 0)],
                    mamba_lazy_spec_track_positions_cpu=plan,
                    mamba_spec_track_positions=staged,
                    mamba_track_mask=torch.ones(2, dtype=torch.bool),
                    mamba_track_seqlens=torch.zeros(2, dtype=torch.int64),
                ),
            )
            with (
                mock.patch.object(
                    spec_utils, "get_exec", return_value=_exec(True, lazy=lazy)
                ),
                mock.patch.object(
                    spec_utils, "set_mamba_track_indices_from_reqs"
                ) as build,
            ):
                spec_utils.prepare_mamba_track_for_verify(batch)
            build.assert_called_once_with(batch, plan, staged)
            assert batch.mamba_track_mask is None
            assert batch.mamba_track_seqlens is None


class TestFastPathTokenCopySkip:
    def _fb(self) -> ForwardBatch:
        fb = ForwardBatch.__new__(ForwardBatch)
        fb.input_ids = torch.zeros(4, dtype=torch.int64)
        fb.positions = torch.zeros(4, dtype=torch.int64)
        return fb

    def test_skip_only_for_the_batch_and_tensors_the_full_path_staged(self) -> None:
        runner = cast(DecodeCudaGraphRunner, SimpleNamespace(_staged_token_inputs=None))
        staged = DecodeCudaGraphRunner._token_inputs_staged
        fb = self._fb()
        assert staged(runner, fb) is False  # nothing pre-planned

        runner._staged_token_inputs = _StagedTokenInputs(
            forward_batch=weakref.ref(fb),
            input_ids=fb.input_ids,
            positions=fb.positions,
        )
        assert staged(runner, fb) is True
        # A pre-planner that only ran the attention plan hands over a different
        # batch object; a batch whose tokens were rebound after the plan copies.
        assert staged(runner, self._fb()) is False
        fb.input_ids = fb.input_ids.clone()
        assert staged(runner, fb) is False

    def test_weak_batch_ref_does_not_pin_the_batch(self) -> None:
        fb = self._fb()
        staged = _StagedTokenInputs(
            forward_batch=weakref.ref(fb),
            input_ids=fb.input_ids,
            positions=fb.positions,
        )
        runner = cast(
            DecodeCudaGraphRunner, SimpleNamespace(_staged_token_inputs=staged)
        )
        del fb
        assert staged.forward_batch() is None
        assert DecodeCudaGraphRunner._token_inputs_staged(runner, self._fb()) is False
