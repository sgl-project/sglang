from __future__ import annotations

import logging
import re
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner import swa_divergence_stats as swa_div_module
from sglang.srt.kv_canary.runner.swa_divergence_stats import (
    SWA_DIVERGENCE_LOG_PREFIX,
    SwaDivergenceStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="extra-a", runner_config="cpu-small")

_DEVICE = torch.device("cpu")


@dataclass
class _RecordingFuture:
    value: torch.Tensor

    def wait(self) -> torch.Tensor:
        return self.value


class _FakeAllocator:
    def __init__(self, wrap_count: int = 0) -> None:
        self.wrap_count = wrap_count


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


class _LogCapture:
    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._handler: Optional[logging.Handler] = None

    def __enter__(self) -> "_LogCapture":
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self.records.append(record)
        logger = logging.getLogger(swa_div_module.__name__)
        self._previous_level = logger.level
        logger.addHandler(self._handler)
        logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        logger = logging.getLogger(swa_div_module.__name__)
        if self._handler is not None:
            logger.removeHandler(self._handler)
        logger.setLevel(self._previous_level)

    def lines(self) -> list[str]:
        return [record.getMessage() for record in self.records]


def _parse_swa_divergence_line(line: str) -> dict[str, int]:
    pattern = (
        re.escape(SWA_DIVERGENCE_LOG_PREFIX)
        + r" forward_ct=(\d+) verify_full=(\d+) verify_swa=(\d+) "
        r"mapping_nonidentity=(\d+) swa_pool_wrap=(\d+)"
    )
    match = re.search(pattern, line)
    if match is None:
        raise AssertionError(f"line does not match swa_divergence format: {line!r}")
    keys = (
        "forward_ct",
        "verify_full",
        "verify_swa",
        "mapping_nonidentity",
        "swa_pool_wrap",
    )
    return {key: int(match.group(idx + 1)) for idx, key in enumerate(keys)}


class TestSwaDivergenceStats(CustomTestCase):
    def setUp(self) -> None:
        self._stream_override = patch.object(
            swa_div_module, "FutureTensor", swa_div_module.FutureTensor
        )

    def test_swa_divergence_log_silent_when_env_disabled(self) -> None:
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(False):
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: _FakeAllocator(wrap_count=5),
            )
            self.assertFalse(stats.enabled)
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.FULL),
                verify_plan=_make_verify_plan(42),
            )
            stats.on_forward_completed()
            with _LogCapture() as capture:
                stats.emit_log_if_due(
                    step_counter=10, period=10, full_to_swa_index_mapping=None
                )
            self.assertEqual(capture.lines(), [])

    def test_swa_divergence_log_emitted_when_env_enabled(self) -> None:
        mapping = torch.arange(8, dtype=torch.int64, device=_DEVICE)
        mapping[3] = 7
        mapping[5] = 1
        mapping_with_sentinel = torch.cat(
            [mapping, torch.tensor([-1], dtype=torch.int64, device=_DEVICE)]
        )

        allocator = _FakeAllocator(wrap_count=13)
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            True
        ), _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: allocator,
            )
            self.assertTrue(stats.enabled)
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

            with _LogCapture() as capture:
                stats.emit_log_if_due(
                    step_counter=10,
                    period=10,
                    full_to_swa_index_mapping=mapping_with_sentinel,
                )
                stats.emit_log_if_due(
                    step_counter=20,
                    period=10,
                    full_to_swa_index_mapping=mapping_with_sentinel,
                )

            lines = [
                line for line in capture.lines() if SWA_DIVERGENCE_LOG_PREFIX in line
            ]
            self.assertEqual(len(lines), 1, lines)
            fields = _parse_swa_divergence_line(lines[0])
            self.assertEqual(fields["forward_ct"], 4)
            self.assertEqual(fields["verify_full"], 40)
            self.assertEqual(fields["verify_swa"], 12)
            self.assertEqual(fields["mapping_nonidentity"], 2)
            self.assertEqual(fields["swa_pool_wrap"], 13)

    def test_swa_divergence_counts_monotonic_increasing(self) -> None:
        allocator = _FakeAllocator(wrap_count=0)
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            True
        ), _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: allocator,
            )

            snapshots: list[dict[str, int]] = []

            def _take_snapshot(step: int) -> None:
                with _LogCapture() as capture:
                    stats.emit_log_if_due(
                        step_counter=step,
                        period=10,
                        full_to_swa_index_mapping=None,
                    )
                    stats.emit_log_if_due(
                        step_counter=step + 10,
                        period=10,
                        full_to_swa_index_mapping=None,
                    )
                matching = [
                    line
                    for line in capture.lines()
                    if SWA_DIVERGENCE_LOG_PREFIX in line
                ]
                self.assertTrue(matching, capture.lines())
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
                allocator.wrap_count = (batch + 1) * 4
                _take_snapshot(step=10 + 20 * batch)

            for idx in range(1, len(snapshots)):
                self.assertGreaterEqual(
                    snapshots[idx]["verify_full"], snapshots[idx - 1]["verify_full"]
                )
                self.assertGreaterEqual(
                    snapshots[idx]["verify_swa"], snapshots[idx - 1]["verify_swa"]
                )
                self.assertGreaterEqual(
                    snapshots[idx]["swa_pool_wrap"],
                    snapshots[idx - 1]["swa_pool_wrap"],
                )

    def test_swa_divergence_mapping_nonidentity_zero_when_only_identity_writes(
        self,
    ) -> None:
        identity_mapping = torch.cat(
            [
                torch.arange(16, dtype=torch.int64, device=_DEVICE),
                torch.tensor([-1], dtype=torch.int64, device=_DEVICE),
            ]
        )
        with envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS.override(
            True
        ), _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: _FakeAllocator(wrap_count=0),
            )
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.FULL),
                verify_plan=_make_verify_plan(1),
            )
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.SWA),
                verify_plan=_make_verify_plan(1),
            )
            stats.on_forward_completed()

            with _LogCapture() as capture:
                stats.emit_log_if_due(
                    step_counter=10,
                    period=10,
                    full_to_swa_index_mapping=identity_mapping,
                )
                stats.emit_log_if_due(
                    step_counter=20,
                    period=10,
                    full_to_swa_index_mapping=identity_mapping,
                )

            matching = [
                line for line in capture.lines() if SWA_DIVERGENCE_LOG_PREFIX in line
            ]
            self.assertEqual(len(matching), 1, matching)
            fields = _parse_swa_divergence_line(matching[0])
            self.assertEqual(fields["mapping_nonidentity"], 0)


if __name__ == "__main__":
    unittest.main()
