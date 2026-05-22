from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import torch
from kv_canary_runner_unit_utils import CanaryRunnerTestCase, make_runner

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner import swa_divergence_stats as swa_div_module
from sglang.srt.kv_canary.runner.swa_divergence_stats import (
    SwaDivergenceLog,
    SwaDivergenceStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="extra-a", runner_config="cpu-small")

_DEVICE = torch.device("cpu")


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


def _patch_future_tensor():
    def _fake_device_to_host(
        *, src_device: torch.Tensor, stream: Optional[torch.cuda.Stream] = None
    ) -> _RecordingFuture:
        return _RecordingFuture(value=src_device.detach().cpu().clone())

    return patch.object(
        swa_div_module.FutureTensor, "device_to_host", _fake_device_to_host
    )


def _parse_swa_divergence_line(line: str) -> SwaDivergenceLog:
    parsed = SwaDivergenceLog.parse(line)
    if parsed is None:
        raise AssertionError(f"line does not match swa_divergence format: {line!r}")
    return parsed


class TestSwaDivergenceStats(CustomTestCase):
    def test_swa_divergence_log_emitted(self) -> None:
        with _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_live_divergence_observer=None,
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
                stats.on_forward_completed()

            with self.assertLogs(
                swa_div_module.logger.name, level=logging.INFO
            ) as captured:
                stats.emit_log_if_due(step_counter=10, period=10)
                stats.emit_log_if_due(step_counter=20, period=10)

            lines = [
                line for line in captured.output if SwaDivergenceLog.parse(line) is not None
            ]
            self.assertEqual(len(lines), 1, lines)
            fields = _parse_swa_divergence_line(lines[0])
            self.assertEqual(fields.forward_ct, 4)
            self.assertEqual(fields.verify_full, 40)
            self.assertEqual(fields.verify_swa, 12)
            self.assertEqual(fields.mapping_nonidentity, 0)

    def test_swa_divergence_counts_monotonic_increasing(self) -> None:
        with _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_live_divergence_observer=None,
            )

            snapshots: list[SwaDivergenceLog] = []

            def _take_snapshot(step: int) -> None:
                with self.assertLogs(
                    swa_div_module.logger.name, level=logging.INFO
                ) as captured:
                    stats.emit_log_if_due(step_counter=step, period=10)
                    stats.emit_log_if_due(step_counter=step + 10, period=10)
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
                    stats.on_forward_completed()
                _take_snapshot(step=10 + 20 * batch)

            for idx in range(1, len(snapshots)):
                self.assertGreaterEqual(
                    snapshots[idx].verify_full, snapshots[idx - 1].verify_full
                )
                self.assertGreaterEqual(
                    snapshots[idx].verify_swa, snapshots[idx - 1].verify_swa
                )


class TestCanaryRunnerSwaDivergenceWiring(CanaryRunnerTestCase):
    def test_swa_divergence_stats_is_none_when_env_disabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            False
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            runner = make_runner(device=self.device)
        self.assertIsNone(runner._swa_divergence_stats)

    def test_swa_divergence_stats_present_when_env_enabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            True
        ), envs.SGLANG_KV_CANARY_PERTURB_TARGET_GROUP.override("full"):
            runner = make_runner(device=self.device)
        self.assertIsNotNone(runner._swa_divergence_stats)
        self.assertIsInstance(runner._swa_divergence_stats, SwaDivergenceStats)


if __name__ == "__main__":
    unittest.main()
