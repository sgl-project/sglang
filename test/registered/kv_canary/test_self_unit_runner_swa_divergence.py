from __future__ import annotations

import logging
import re
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
    SWA_DIVERGENCE_LOG_PREFIX,
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


class _FakeAllocator:
    __slots__ = ("_wrap_count_device", "_nonidentity_write_count_device")

    def __init__(
        self,
        *,
        wrap_count: int = 0,
        nonidentity_write_count: int = 0,
    ) -> None:
        self._wrap_count_device = torch.tensor(
            [wrap_count], dtype=torch.int64, device=_DEVICE
        )
        self._nonidentity_write_count_device = torch.tensor(
            [nonidentity_write_count], dtype=torch.int64, device=_DEVICE
        )

    @property
    def wrap_count(self) -> int:
        return int(self._wrap_count_device.item())

    @property
    def nonidentity_write_count(self) -> int:
        return int(self._nonidentity_write_count_device.item())


class _AllocatorWrapObserver:
    __slots__ = (
        "_wrap_count_device",
        "_max_observed_swa_idx_device",
        "_nonidentity_write_count_device",
    )

    def __init__(self) -> None:
        self._wrap_count_device = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
        self._max_observed_swa_idx_device = torch.zeros(
            1, dtype=torch.int64, device=_DEVICE
        )
        self._nonidentity_write_count_device = torch.zeros(
            1, dtype=torch.int64, device=_DEVICE
        )

    @property
    def wrap_count(self) -> int:
        return int(self._wrap_count_device.item())

    @property
    def nonidentity_write_count(self) -> int:
        return int(self._nonidentity_write_count_device.item())

    def observe_swa_alloc(self, alloc_swa_indices: torch.Tensor) -> None:
        if alloc_swa_indices.numel() == 0:
            return
        flat = alloc_swa_indices.reshape(-1).to(torch.int64)
        batch_min = flat.min().view(1)
        batch_max = flat.max().view(1)
        wrapped = (batch_min < self._max_observed_swa_idx_device).to(torch.int64)
        self._wrap_count_device.add_(wrapped)
        torch.maximum(
            self._max_observed_swa_idx_device,
            batch_max,
            out=self._max_observed_swa_idx_device,
        )

    def observe_swa_mapping_write(
        self,
        full_indices: torch.Tensor,
        swa_indices: torch.Tensor,
    ) -> None:
        if full_indices.numel() == 0:
            return
        full_flat = full_indices.reshape(-1).to(torch.int64)
        swa_flat = swa_indices.reshape(-1).to(torch.int64)
        nonidentity = (swa_flat != full_flat).sum().to(torch.int64).view(1)
        self._nonidentity_write_count_device.add_(nonidentity)


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
        self._previous_level: Optional[int] = None

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
        if self._previous_level is not None:
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
    def test_swa_divergence_log_emitted(self) -> None:
        allocator = _FakeAllocator(wrap_count=13, nonidentity_write_count=2)
        with _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: allocator,
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

            with _LogCapture() as capture:
                stats.emit_log_if_due(step_counter=10, period=10)
                stats.emit_log_if_due(step_counter=20, period=10)

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
        with _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: allocator,
            )

            snapshots: list[dict[str, int]] = []

            def _take_snapshot(step: int) -> None:
                with _LogCapture() as capture:
                    stats.emit_log_if_due(step_counter=step, period=10)
                    stats.emit_log_if_due(step_counter=step + 10, period=10)
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
                allocator._wrap_count_device.fill_((batch + 1) * 4)
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

    def test_nonidentity_write_count_emitted_from_allocator(self) -> None:
        allocator = _FakeAllocator(wrap_count=0, nonidentity_write_count=7)
        with _patch_future_tensor():
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_allocator_getter=lambda: allocator,
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
                stats.emit_log_if_due(step_counter=10, period=10)
                stats.emit_log_if_due(step_counter=20, period=10)

            matching = [
                line for line in capture.lines() if SWA_DIVERGENCE_LOG_PREFIX in line
            ]
            self.assertEqual(len(matching), 1, matching)
            fields = _parse_swa_divergence_line(matching[0])
            self.assertEqual(fields["mapping_nonidentity"], 7)


class TestSwaPoolWrapCount(CustomTestCase):
    def test_wrap_count_zero_when_no_wraparound(self) -> None:
        observer = _AllocatorWrapObserver()
        observer.observe_swa_alloc(torch.tensor([1, 2, 3], dtype=torch.int64))
        observer.observe_swa_alloc(torch.tensor([4, 5, 6], dtype=torch.int64))
        observer.observe_swa_alloc(torch.tensor([7, 8], dtype=torch.int64))
        self.assertEqual(observer.wrap_count, 0)

    def test_wrap_count_increments_on_pointer_wraparound(self) -> None:
        observer = _AllocatorWrapObserver()
        observer.observe_swa_alloc(torch.tensor([10, 11, 12], dtype=torch.int64))
        observer.observe_swa_alloc(torch.tensor([13, 14], dtype=torch.int64))
        observer.observe_swa_alloc(torch.tensor([3, 4], dtype=torch.int64))
        self.assertEqual(observer.wrap_count, 1)
        observer.observe_swa_alloc(torch.tensor([20], dtype=torch.int64))
        self.assertEqual(observer.wrap_count, 1)
        observer.observe_swa_alloc(torch.tensor([5], dtype=torch.int64))
        self.assertEqual(observer.wrap_count, 2)


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


class TestSwaPoolNonidentityWriteCount(CustomTestCase):
    def test_nonidentity_write_count_zero_when_no_alloc(self) -> None:
        observer = _AllocatorWrapObserver()
        self.assertEqual(observer.nonidentity_write_count, 0)

    def test_nonidentity_write_count_increments_on_nonidentity_write(self) -> None:
        observer = _AllocatorWrapObserver()
        observer.observe_swa_mapping_write(
            full_indices=torch.tensor([5], dtype=torch.int64),
            swa_indices=torch.tensor([100], dtype=torch.int64),
        )
        self.assertEqual(observer.nonidentity_write_count, 1)
        observer.observe_swa_mapping_write(
            full_indices=torch.tensor([6, 7], dtype=torch.int64),
            swa_indices=torch.tensor([200, 300], dtype=torch.int64),
        )
        self.assertEqual(observer.nonidentity_write_count, 3)

    def test_nonidentity_write_count_unchanged_on_identity_write(self) -> None:
        observer = _AllocatorWrapObserver()
        observer.observe_swa_mapping_write(
            full_indices=torch.tensor([5], dtype=torch.int64),
            swa_indices=torch.tensor([5], dtype=torch.int64),
        )
        self.assertEqual(observer.nonidentity_write_count, 0)
        observer.observe_swa_mapping_write(
            full_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
            swa_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
        )
        self.assertEqual(observer.nonidentity_write_count, 0)


if __name__ == "__main__":
    unittest.main()
