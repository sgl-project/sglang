from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.runner import swa_divergence_stats as swa_div_module
from sglang.srt.kv_canary.runner import (
    swa_pool_static_observer as swa_pool_observer_module,
)
from sglang.srt.kv_canary.runner.swa_divergence_stats import (
    SwaDivergenceLog,
    SwaDivergenceStats,
)
from sglang.srt.kv_canary.runner.swa_pool_static_observer import SwaPoolStaticObserver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="extra-a", runner_config="cpu-small")

_DEVICE = torch.device("cpu")


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordingFuture:
    value: torch.Tensor

    def wait(self) -> torch.Tensor:
        return self.value


def _make_allocator_stub(mapping: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(full_to_swa_index_mapping=mapping)


def _patch_future_tensor():
    def _fake_device_to_host(
        *, src_device: torch.Tensor, stream: Optional[torch.cuda.Stream] = None
    ) -> _RecordingFuture:
        return _RecordingFuture(value=src_device.detach().cpu().clone())

    return [
        patch.object(
            swa_div_module.FutureTensor, "device_to_host", _fake_device_to_host
        ),
        patch.object(
            swa_pool_observer_module.FutureTensor,
            "device_to_host",
            _fake_device_to_host,
        ),
    ]


def _patch_cuda_stream_ctx():
    """Replace ``torch.cuda.stream`` with a noop ctxmgr so CPU tests can exercise
    ``SwaPoolStaticObserver.snapshot_nonidentity_future`` without a real CUDA
    stream."""
    import contextlib

    @contextlib.contextmanager
    def _noop_stream(_stream):
        yield

    return patch.object(torch.cuda, "stream", _noop_stream)


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


class TestSwaPoolStaticObserver(CustomTestCase):
    def test_baseline_clone_at_install(self) -> None:
        mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
        allocator = _make_allocator_stub(mapping)
        observer = SwaPoolStaticObserver(swa_allocator=allocator)

        mapping[0] = 99
        mapping[3] = 77

        self.assertTrue(
            torch.equal(
                observer._baseline_mapping,
                torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE),
            )
        )

    def test_snapshot_returns_zero_when_no_writes(self) -> None:
        mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
        allocator = _make_allocator_stub(mapping)
        observer = SwaPoolStaticObserver(swa_allocator=allocator)

        patchers = _patch_future_tensor()
        with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
            future = observer.snapshot_nonidentity_future(stream=None)
        self.assertEqual(int(future.wait().item()), 0)

    def test_snapshot_counts_nonidentity_writes(self) -> None:
        mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
        allocator = _make_allocator_stub(mapping)
        observer = SwaPoolStaticObserver(swa_allocator=allocator)

        mapping[1] = 100
        mapping[3] = 200

        patchers = _patch_future_tensor()
        with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
            future = observer.snapshot_nonidentity_future(stream=None)
        self.assertEqual(int(future.wait().item()), 2)

    def test_snapshot_ignores_writes_equal_to_baseline(self) -> None:
        mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
        allocator = _make_allocator_stub(mapping)
        observer = SwaPoolStaticObserver(swa_allocator=allocator)

        mapping[1] = 100
        mapping[1] = 1

        patchers = _patch_future_tensor()
        with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
            future = observer.snapshot_nonidentity_future(stream=None)
        self.assertEqual(int(future.wait().item()), 0)

    def test_init_raises_when_mapping_not_registered(self) -> None:
        allocator = _make_allocator_stub(mapping=None)
        with self.assertRaisesRegex(RuntimeError, "mapping to be registered"):
            SwaPoolStaticObserver(swa_allocator=allocator)


class TestSwaDivergenceStatsWithObserver(CustomTestCase):
    def test_swa_divergence_stats_emits_mapping_nonidentity_from_observer(
        self,
    ) -> None:
        mapping = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=_DEVICE)
        allocator = _make_allocator_stub(mapping)
        observer = SwaPoolStaticObserver(swa_allocator=allocator)

        mapping[1] = 50
        mapping[2] = 60
        mapping[4] = 80

        patchers = _patch_future_tensor()
        with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_pool_static_observer=observer,
            )
            stats.observe_after_invoke_plan(
                group=_make_group(PoolKind.FULL),
                verify_plan=_make_verify_plan(11),
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

        matching = [
            line for line in captured.output if SwaDivergenceLog.parse(line) is not None
        ]
        self.assertEqual(len(matching), 1, matching)
        parsed = SwaDivergenceLog.parse(matching[0])
        assert parsed is not None
        self.assertEqual(parsed.mapping_nonidentity, 3)
        self.assertEqual(parsed.verify_full, 11)
        self.assertEqual(parsed.verify_swa, 3)


if __name__ == "__main__":
    unittest.main()
