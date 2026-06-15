from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.runner import swa_divergence as swa_div_module
from sglang.srt.kv_canary.runner.swa_divergence import (
    SwaDivergenceLog,
    SwaDivergenceReporter,
    compute_swa_full_idx_divergence,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import make_buffer_group
from sglang.test.kv_canary.runner_test_base import CanaryManagerTestCase, make_manager
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="extra-a-test-1-gpu-small-amd")

_DEVICE = torch.device("cuda")

_EMPTY_FORWARD_BATCH = SimpleNamespace(
    req_pool_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
    seq_lens=torch.empty(0, dtype=torch.int64, device=_DEVICE),
)


def _make_verify_plan(value: int) -> VerifyPlan:
    plan = VerifyPlan.allocate(verify_capacity=4, device=_DEVICE)
    plan.verify_num_valid.copy_(torch.tensor([value], dtype=torch.int32))
    return plan


def _make_allocator_stub(mapping: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(full_to_swa_index_mapping=mapping)


def _make_req_to_token_pool_stub(req_to_token: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(req_to_token=req_to_token)


def _make_identity_mapping(size: int) -> torch.Tensor:
    return torch.arange(size, dtype=torch.int64, device=_DEVICE)


def _make_identity_req_to_token(num_reqs: int, max_seq_len: int) -> torch.Tensor:
    base = torch.arange(num_reqs * max_seq_len, dtype=torch.int64, device=_DEVICE)
    return base.view(num_reqs, max_seq_len)


def _make_forward_batch(
    *, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
) -> SimpleNamespace:
    return SimpleNamespace(req_pool_indices=req_pool_indices, seq_lens=seq_lens)


def _parse_swa_divergence_line(line: str) -> SwaDivergenceLog:
    parsed = SwaDivergenceLog.parse(line)
    if parsed is None:
        raise AssertionError(f"line does not match swa_divergence format: {line!r}")
    return parsed


def _run_compute(
    *,
    swa_allocator: SimpleNamespace,
    req_to_token_pool: SimpleNamespace,
    forward_batch: SimpleNamespace,
) -> int:
    count = compute_swa_full_idx_divergence(
        swa_allocator=swa_allocator,
        req_to_token_pool=req_to_token_pool,
        maybe_inaccurate_forward_batch=forward_batch,
    )
    return int(count.item())


class TestSwaDivergenceReporter(CustomTestCase):
    def test_swa_divergence_log_emitted(self) -> None:
        d2h_stream = torch.cuda.Stream(device=_DEVICE)
        stats = SwaDivergenceReporter(
            device=_DEVICE,
            d2h_stream=d2h_stream,
            interval=10,
            swa_allocator=None,
            req_to_token_pool=None,
        )
        # First 3 forwards stay below the interval trigger (1, 2, 3 % 10 != 0) so
        # step() just bumps forward_ct and stages nothing.
        for forward_idx in range(3):
            stats.observe_after_invoke_plan(
                group=make_buffer_group(
                    device=_DEVICE, kind=PoolKind.FULL, has_v=False, num_slots=1
                ),
                verify_plan=_make_verify_plan(10),
            )
            stats.observe_after_invoke_plan(
                group=make_buffer_group(
                    device=_DEVICE, kind=PoolKind.SWA, has_v=False, num_slots=1
                ),
                verify_plan=_make_verify_plan(3),
            )
            stats.step(
                outer_step_counter=forward_idx + 1,
                maybe_inaccurate_forward_batch=_EMPTY_FORWARD_BATCH,
            )
        # 4th forward lands on outer_step_counter=10 = interval, so compute_on_device
        # snapshots {forward_ct:4, verify_full:40, verify_swa:12} into the dict and
        # the staged future hangs onto it. forward_ct is now 4.
        stats.observe_after_invoke_plan(
            group=make_buffer_group(
                device=_DEVICE, kind=PoolKind.FULL, has_v=False, num_slots=1
            ),
            verify_plan=_make_verify_plan(10),
        )
        stats.observe_after_invoke_plan(
            group=make_buffer_group(
                device=_DEVICE, kind=PoolKind.SWA, has_v=False, num_slots=1
            ),
            verify_plan=_make_verify_plan(3),
        )
        stats.step(
            outer_step_counter=10, maybe_inaccurate_forward_batch=_EMPTY_FORWARD_BATCH
        )

        # 5th step drains the previous stage and emits the log; forward_ct is now 5
        # but the staged dict still carries the snapshot forward_ct=4 from step 4.
        with self.assertLogs(
            swa_div_module.logger.name, level=logging.INFO
        ) as captured:
            stats.step(
                outer_step_counter=11,
                maybe_inaccurate_forward_batch=_EMPTY_FORWARD_BATCH,
            )

        lines = [
            line for line in captured.output if SwaDivergenceLog.parse(line) is not None
        ]
        self.assertEqual(len(lines), 1, lines)
        fields = _parse_swa_divergence_line(lines[0])
        self.assertEqual(fields.forward_ct, 4)
        self.assertEqual(fields.verify_full, 40)
        self.assertEqual(fields.verify_swa, 12)
        self.assertEqual(fields.swa_full_idx_divergence, 0)

    def test_swa_divergence_counts_monotonic_increasing(self) -> None:
        d2h_stream = torch.cuda.Stream(device=_DEVICE)
        stats = SwaDivergenceReporter(
            device=_DEVICE,
            d2h_stream=d2h_stream,
            interval=10,
            swa_allocator=None,
            req_to_token_pool=None,
        )

        snapshots: list[SwaDivergenceLog] = []

        def _take_snapshot(stage_step: int, drain_step: int) -> None:
            # Stage the dict at the interval-aligned step (no log emitted yet,
            # DelayedDeviceHostHandler still has nothing to drain), then call
            # step() again at the next counter to drain and emit the log.
            stats.step(
                outer_step_counter=stage_step,
                maybe_inaccurate_forward_batch=_EMPTY_FORWARD_BATCH,
            )
            with self.assertLogs(
                swa_div_module.logger.name, level=logging.INFO
            ) as captured:
                stats.step(
                    outer_step_counter=drain_step,
                    maybe_inaccurate_forward_batch=_EMPTY_FORWARD_BATCH,
                )
            matching = [
                line
                for line in captured.output
                if SwaDivergenceLog.parse(line) is not None
            ]
            self.assertTrue(matching, captured.output)
            snapshots.append(_parse_swa_divergence_line(matching[-1]))

        for batch in range(3):
            for _ in range(5):
                stats.observe_after_invoke_plan(
                    group=make_buffer_group(
                        device=_DEVICE, kind=PoolKind.FULL, has_v=False, num_slots=1
                    ),
                    verify_plan=_make_verify_plan(7),
                )
                stats.observe_after_invoke_plan(
                    group=make_buffer_group(
                        device=_DEVICE, kind=PoolKind.SWA, has_v=False, num_slots=1
                    ),
                    verify_plan=_make_verify_plan(2),
                )
            stage_step = 10 + 20 * batch
            _take_snapshot(stage_step=stage_step, drain_step=stage_step + 1)

        for idx in range(1, len(snapshots)):
            self.assertGreaterEqual(
                snapshots[idx].verify_full, snapshots[idx - 1].verify_full
            )
            self.assertGreaterEqual(
                snapshots[idx].verify_swa, snapshots[idx - 1].verify_swa
            )


class TestSwaFullIdxDivergenceCompute(CustomTestCase):
    def test_compute_returns_zero_when_empty_batch(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=forward_batch,
            ),
            0,
        )

    def test_compute_returns_zero_when_all_identity(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8, 5], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=forward_batch,
            ),
            0,
        )

    def test_compute_counts_swa_full_idx_divergence_in_live_range(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        mapping[0] = 50
        mapping[1] = 51
        mapping[17] = 60

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8, 8], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=forward_batch,
            ),
            3,
        )

    def test_compute_ignores_swa_mapping_zero(self) -> None:
        # SWATokenToKVPoolAllocator writes 0 into full_to_swa_index_mapping for
        # FULL pool slots beyond the sliding window. Those entries are expected,
        # not real divergence, so the count must skip them.
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        mapping[3] = 0
        mapping[5] = 0
        mapping[7] = 42

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=forward_batch,
            ),
            1,
        )

    def test_compute_ignores_writes_outside_seq_lens(self) -> None:
        mapping = _make_identity_mapping(size=128)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=32)

        mapping[20] = 99
        mapping[28] = 77

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([10], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=forward_batch,
            ),
            0,
        )

    def test_compute_reflects_current_forward_batch(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        mapping[0] = 41
        mapping[1] = 42
        mapping[32] = 99
        mapping[33] = 100

        fb_req0 = _make_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([4], dtype=torch.int64, device=_DEVICE),
        )
        fb_req2 = _make_forward_batch(
            req_pool_indices=torch.tensor([2], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([4], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=fb_req0,
            ),
            2,
        )
        self.assertEqual(
            _run_compute(
                swa_allocator=_make_allocator_stub(mapping),
                req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
                forward_batch=fb_req2,
            ),
            2,
        )


class TestSwaDivergenceReporterWithCompute(CustomTestCase):
    def test_swa_divergence_report_emits_swa_full_idx_divergence_from_compute(
        self,
    ) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)

        mapping[0] = 50
        mapping[1] = 51
        mapping[2] = 52

        forward_batch = _make_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8], dtype=torch.int64, device=_DEVICE),
        )

        swa_allocator = _make_allocator_stub(mapping)
        req_to_token_pool = _make_req_to_token_pool_stub(req_to_token)
        d2h_stream = torch.cuda.Stream(device=_DEVICE)
        stats = SwaDivergenceReporter(
            device=_DEVICE,
            d2h_stream=d2h_stream,
            interval=10,
            swa_allocator=swa_allocator,
            req_to_token_pool=req_to_token_pool,
        )
        stats.observe_after_invoke_plan(
            group=make_buffer_group(
                device=_DEVICE, kind=PoolKind.FULL, has_v=False, num_slots=1
            ),
            verify_plan=_make_verify_plan(11),
        )
        stats.observe_after_invoke_plan(
            group=make_buffer_group(
                device=_DEVICE, kind=PoolKind.SWA, has_v=False, num_slots=1
            ),
            verify_plan=_make_verify_plan(3),
        )

        # Stage at the interval-aligned step, then drain on the next step so the
        # DelayedDeviceHostHandler has a pending future to postprocess.
        stats.step(outer_step_counter=10, maybe_inaccurate_forward_batch=forward_batch)
        with self.assertLogs(
            swa_div_module.logger.name, level=logging.INFO
        ) as captured:
            stats.step(
                outer_step_counter=11, maybe_inaccurate_forward_batch=forward_batch
            )

        matching = [
            line for line in captured.output if SwaDivergenceLog.parse(line) is not None
        ]
        self.assertEqual(len(matching), 1, matching)
        parsed = SwaDivergenceLog.parse(matching[0])
        assert parsed is not None
        self.assertEqual(parsed.swa_full_idx_divergence, 3)
        self.assertEqual(parsed.verify_full, 11)
        self.assertEqual(parsed.verify_swa, 3)


class TestSwaDivergenceLogFindAll(CustomTestCase):
    def test_find_all_returns_every_sample_in_order(self) -> None:
        text = "\n".join(
            SwaDivergenceLog(
                forward_ct=ct,
                verify_full=100 * ct,
                verify_swa=10 * ct,
                swa_full_idx_divergence=ct,
                swa_out_of_window_tokens=0,
            ).format()
            for ct in (20, 40, 60)
        )
        parsed = SwaDivergenceLog.find_all(text)
        self.assertEqual([p.forward_ct for p, _ in parsed], [20, 40, 60])

    def test_find_all_peak_survives_trailing_zero_sample(self) -> None:
        text = "\n".join(
            SwaDivergenceLog(
                forward_ct=ct,
                verify_full=1,
                verify_swa=0,
                swa_full_idx_divergence=1,
                swa_out_of_window_tokens=oow,
            ).format()
            for ct, oow in ((20, 0), (40, 4080), (60, 0))
        )
        parsed = SwaDivergenceLog.find_all(text)
        self.assertEqual(max(p.swa_out_of_window_tokens for p, _ in parsed), 4080)
        self.assertEqual(parsed[-1][0].swa_out_of_window_tokens, 0)

    def test_find_all_returns_empty_list_when_no_lines(self) -> None:
        self.assertEqual(SwaDivergenceLog.find_all("nothing here\n"), [])


class TestCanaryManagerSwaDivergenceWiring(CanaryManagerTestCase):
    def test_swa_divergence_report_is_none_when_env_disabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL.override(
            0
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            manager = make_manager(device=self.device)
        self.assertIsNone(manager._swa_divergence_report)

    def test_swa_divergence_report_present_when_env_enabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL.override(
            20
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            manager = make_manager(device=self.device)
        self.assertIsNotNone(manager._swa_divergence_report)
        self.assertIsInstance(manager._swa_divergence_report, SwaDivergenceReporter)


if __name__ == "__main__":
    unittest.main()
