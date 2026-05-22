from __future__ import annotations

import contextlib
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
    swa_live_divergence_observer as swa_observer_module,
)
from sglang.srt.kv_canary.runner.swa_divergence_stats import (
    SwaDivergenceLog,
    SwaDivergenceStats,
)
from sglang.srt.kv_canary.runner.swa_live_divergence_observer import (
    SwaLiveDivergenceObserver,
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


def _make_allocator_stub(mapping: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(full_to_swa_index_mapping=mapping)


def _make_req_to_token_pool_stub(req_to_token: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(req_to_token=req_to_token)


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
            swa_observer_module.FutureTensor,
            "device_to_host",
            _fake_device_to_host,
        ),
    ]


def _patch_cuda_stream_ctx():
    @contextlib.contextmanager
    def _noop_stream(_stream):
        yield

    return patch.object(torch.cuda, "stream", _noop_stream)


def _make_identity_mapping(size: int) -> torch.Tensor:
    return torch.arange(size, dtype=torch.int64, device=_DEVICE)


def _make_identity_req_to_token(num_reqs: int, max_seq_len: int) -> torch.Tensor:
    base = torch.arange(num_reqs * max_seq_len, dtype=torch.int64, device=_DEVICE)
    return base.view(num_reqs, max_seq_len)


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


def _run_snapshot(observer: SwaLiveDivergenceObserver) -> int:
    patchers = _patch_future_tensor()
    with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
        future = observer.snapshot_nonidentity_future(stream=None)
    return int(future.wait().item())


class TestSwaLiveDivergenceObserver(CustomTestCase):
    def test_snapshot_returns_zero_when_no_observe(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        self.assertEqual(_run_snapshot(observer), 0)

    def test_snapshot_returns_zero_when_all_identity(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8, 5], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(_run_snapshot(observer), 0)

    def test_snapshot_counts_nonidentity_in_live_range(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        mapping[0] = 50
        mapping[1] = 51
        mapping[17] = 60

        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8, 8], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(_run_snapshot(observer), 3)

    def test_snapshot_ignores_writes_outside_seq_lens(self) -> None:
        mapping = _make_identity_mapping(size=128)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=32)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        mapping[20] = 99
        mapping[28] = 77

        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([10], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(_run_snapshot(observer), 0)

    def test_snapshot_uses_latest_observe(self) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        mapping[0] = 41
        mapping[1] = 42
        mapping[32] = 99
        mapping[33] = 100

        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([4], dtype=torch.int64, device=_DEVICE),
        )
        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([2], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([4], dtype=torch.int64, device=_DEVICE),
        )

        self.assertEqual(_run_snapshot(observer), 2)


class TestSwaDivergenceStatsWithObserver(CustomTestCase):
    def test_swa_divergence_stats_emits_mapping_nonidentity_from_observer(
        self,
    ) -> None:
        mapping = _make_identity_mapping(size=64)
        req_to_token = _make_identity_req_to_token(num_reqs=4, max_seq_len=16)
        observer = SwaLiveDivergenceObserver(
            swa_allocator=_make_allocator_stub(mapping),
            req_to_token_pool=_make_req_to_token_pool_stub(req_to_token),
        )

        mapping[0] = 50
        mapping[1] = 51
        mapping[2] = 52

        observer.observe_forward_batch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
            seq_lens=torch.tensor([8], dtype=torch.int64, device=_DEVICE),
        )

        patchers = _patch_future_tensor()
        with _patch_cuda_stream_ctx(), patchers[0], patchers[1]:
            stats = SwaDivergenceStats(
                device=_DEVICE,
                d2h_stream=None,
                swa_live_divergence_observer=observer,
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
