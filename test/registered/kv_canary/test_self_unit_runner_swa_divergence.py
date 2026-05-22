from __future__ import annotations

import contextlib
import logging
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch
from kv_canary_runner_unit_utils import CanaryRunnerTestCase, make_runner

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner import future_tensor as future_tensor_module
from sglang.srt.kv_canary.runner import swa_divergence as swa_div_module
from sglang.srt.kv_canary.runner.swa_divergence import (
    SwaDivergenceLog,
    SwaDivergenceReport,
    compute_swa_full_idx_divergence,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="extra-a", runner_config="cpu-small")

_DEVICE = torch.device("cpu")

_EMPTY_FORWARD_BATCH = SimpleNamespace(
    req_pool_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
    seq_lens=torch.empty(0, dtype=torch.int64, device=_DEVICE),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordingFuture:
    value: torch.Tensor

    def wait(self) -> torch.Tensor:
        return self.value


def _make_verify_plan(value: int) -> VerifyPlan:
    plan = VerifyPlan.allocate(verify_capacity=4, device=_DEVICE)
    plan.verify_num_valid.copy_(torch.tensor([value], dtype=torch.int32))
    return plan


def _make_group(kind: PoolKind) -> CanaryBufferGroup:
    return CanaryBufferGroup(
        kind=kind,
        k_head=torch.zeros(1, 1, dtype=torch.uint8, device=_DEVICE),
        k_tail=torch.zeros(1, 1, dtype=torch.uint8, device=_DEVICE),
        v_head=None,
        v_tail=None,
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )


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


def _patch_future_tensor():
    def _fake_device_to_host(src_device, *, stream=None) -> _RecordingFuture:
        if isinstance(src_device, dict):
            return _RecordingFuture(
                value={k: v.detach().cpu().clone() for k, v in src_device.items()}
            )
        return _RecordingFuture(value=src_device.detach().cpu().clone())

    return patch.object(
        future_tensor_module.FutureTensors, "device_to_host", _fake_device_to_host
    )


def _patch_cuda_stream_ctx():
    @contextlib.contextmanager
    def _noop_stream(_stream):
        yield

    return patch.object(torch.cuda, "stream", _noop_stream)


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
        forward_batch=forward_batch,
    )
    return int(count.item())


class TestSwaDivergenceReport(CustomTestCase):
    def test_swa_divergence_log_emitted(self) -> None:
        with _patch_future_tensor():
            stats = SwaDivergenceReport(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator=None,
                req_to_token_pool=None,
            )
            for _ in range(4):
                stats.observe_after_invoke_plan(
                    group=_make_group(PoolKind.FULL),
                    verify_plan=_make_verify_plan(10),
                )
                stats.observe_after_invoke_plan(
                    group=_make_group(PoolKind.SWA),
                    verify_plan=_make_verify_plan(3),
                )

            with self.assertLogs(
                swa_div_module.logger.name, level=logging.INFO
            ) as captured:
                stats.step(
                    step_counter=10, period=10, forward_batch=_EMPTY_FORWARD_BATCH
                )

            lines = [
                line
                for line in captured.output
                if SwaDivergenceLog.parse(line) is not None
            ]
            self.assertEqual(len(lines), 1, lines)
            fields = _parse_swa_divergence_line(lines[0])
            self.assertEqual(fields.forward_ct, 4)
            self.assertEqual(fields.verify_full, 40)
            self.assertEqual(fields.verify_swa, 12)
            self.assertEqual(fields.swa_full_idx_divergence, 0)

    def test_swa_divergence_counts_monotonic_increasing(self) -> None:
        with _patch_future_tensor():
            stats = SwaDivergenceReport(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator=None,
                req_to_token_pool=None,
            )

            snapshots: list[SwaDivergenceLog] = []

            def _take_snapshot(step: int) -> None:
                with self.assertLogs(
                    swa_div_module.logger.name, level=logging.INFO
                ) as captured:
                    stats.step(
                        step_counter=step, period=10, forward_batch=_EMPTY_FORWARD_BATCH
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
                        group=_make_group(PoolKind.FULL),
                        verify_plan=_make_verify_plan(7),
                    )
                    stats.observe_after_invoke_plan(
                        group=_make_group(PoolKind.SWA),
                        verify_plan=_make_verify_plan(2),
                    )
                _take_snapshot(step=10 + 20 * batch)

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


class TestSwaDivergenceReportWithCompute(CustomTestCase):
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
        with _patch_cuda_stream_ctx(), _patch_future_tensor():
            stats = SwaDivergenceReport(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator=swa_allocator,
                req_to_token_pool=req_to_token_pool,
            )
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.FULL),
                verify_plan=_make_verify_plan(11),
            )
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.SWA),
                verify_plan=_make_verify_plan(3),
            )

            with self.assertLogs(
                swa_div_module.logger.name, level=logging.INFO
            ) as captured:
                stats.step(step_counter=10, period=10, forward_batch=forward_batch)

        matching = [
            line for line in captured.output if SwaDivergenceLog.parse(line) is not None
        ]
        self.assertEqual(len(matching), 1, matching)
        parsed = SwaDivergenceLog.parse(matching[0])
        assert parsed is not None
        self.assertEqual(parsed.swa_full_idx_divergence, 3)
        self.assertEqual(parsed.verify_full, 11)
        self.assertEqual(parsed.verify_swa, 3)


class TestCanaryRunnerSwaDivergenceWiring(CanaryRunnerTestCase):
    def test_swa_divergence_report_is_none_when_env_disabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            False
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            runner = make_runner(device=self.device)
        self.assertIsNone(runner._swa_divergence_report)

    def test_swa_divergence_report_present_when_env_enabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            True
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            runner = make_runner(device=self.device)
        self.assertIsNotNone(runner._swa_divergence_report)
        self.assertIsInstance(runner._swa_divergence_report, SwaDivergenceReport)


if __name__ == "__main__":
    unittest.main()
