from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _fixtures import (  # noqa: E402
    CPU_DEVICE,
    make_radix_cache,
    make_req_to_token_pool,
)

from sglang.jit_kernel.kv_canary.verify import (  # noqa: E402
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvHashMode,
)
from sglang.srt.kv_canary import endpoint as endpoint_module  # noqa: E402
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind  # noqa: E402
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode  # noqa: E402
from sglang.srt.kv_canary.runner import canary_runner as runner_module  # noqa: E402
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


def _make_group(*, device, has_v: bool = True, kind: PoolKind = PoolKind.FULL):
    return CanaryBufferGroup(
        kind=kind,
        k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        v_head=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        v_tail=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )


def _make_pool(device, max_reqs: int = 4, max_seq: int = 8):
    table = torch.zeros(max_reqs, max_seq, dtype=torch.int32, device=device)
    return SimpleNamespace(req_to_token=table, size=max_reqs)


def _make_forward_batch(device, bs: int = 2):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_mixed=lambda: False),
        req_pool_indices=torch.tensor([1, 2][:bs], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([3, 4][:bs], dtype=torch.int32, device=device),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
        positions=torch.zeros(bs, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(bs, dtype=torch.int32, device=device),
    )


def _make_runner(*, device, config=None, group=None, req_pool=None):
    if config is None:
        config = CanaryConfig(
            mode=CanaryMode.RAISE, real_kv_hash_mode=RealKvHashMode.OFF
        )
    if group is None:
        group = _make_group(device=device)
    if req_pool is None:
        req_pool = _make_pool(device)
    return CanaryRunner(
        config=config,
        buffer_groups_per_pool=[(group,)],
        device=device,
        req_to_token_pool=req_pool,
        per_forward_verify_capacity=4,
        per_forward_write_req_capacity=2,
        per_forward_write_entry_capacity=8,
        sweep_verify_capacity=8,
    )


class TestSelfUnitRunner(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE
        # Stub plan/verify/write kernels so CPU runs don't need CUDA JIT.
        self._patchers = [
            patch.object(runner_module, "canary_plan_step", lambda **kwargs: None),
            patch.object(endpoint_module, "canary_verify_step", lambda **kwargs: None),
            patch.object(endpoint_module, "canary_write_step", lambda **kwargs: None),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_per_forward_orchestrates_plan_head_tail(self):
        calls: List = []
        with patch.object(
            runner_module,
            "canary_plan_step",
            lambda **kwargs: calls.append("plan"),
        ), patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: calls.append(("verify", kwargs["kernel_kind"].name)),
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: calls.append(("write", kwargs["kernel_kind"].name)),
        ):
            runner = _make_runner(device=self.device)
            fb = _make_forward_batch(self.device)
            runner.before_forward(fb)
            runner.launch_head_kernels(fb)
            runner.launch_tail_kernels(fb)

        self.assertEqual(calls[0], "plan")
        self.assertTrue(
            any(
                c[0] == "verify" and "HEAD" in c[1]
                for c in calls[1:]
                if isinstance(c, tuple)
            )
        )
        self.assertTrue(
            any(
                c[0] == "verify" and "TAIL" in c[1]
                for c in calls[1:]
                if isinstance(c, tuple)
            )
        )

    def test_sweep_every_n_cadence(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_every_n_steps=4,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)
        runner.before_forward(fb)
        runner.launch_head_kernels(fb)

        sweep_calls: List[int] = []
        real_maybe = runner.maybe_run_sweep

        def _spy():
            before = runner._last_sweep_step
            real_maybe()
            if runner._last_sweep_step != before:
                sweep_calls.append(runner._step_counter)

        with patch.object(runner, "maybe_run_sweep", _spy):
            for _ in range(12):
                runner.end_of_step()
        self.assertEqual(sweep_calls, [0, 4, 8])

    def test_sweep_runs_radix_path(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_every_n_steps=1,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)
        runner.before_forward(fb)
        runner.launch_head_kernels(fb)

        cache = make_radix_cache([[], [10, 11]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        plan_calls: List[str] = []
        with patch.object(
            runner_module,
            "canary_plan_step",
            lambda **kwargs: plan_calls.append("plan"),
        ):
            runner.maybe_run_sweep()
        self.assertGreaterEqual(plan_calls.count("plan"), 1)

    def test_violation_pump_d2h_detects_errored(self):
        runner = _make_runner(device=self.device)
        self.assertEqual(
            int(runner._device_state.violation_log.violation_write_index.item()), 0
        )
        runner._device_state.violation_log.violation_write_index[0] = 1
        self.assertEqual(
            int(runner._device_state.violation_log.violation_write_index.item()), 1
        )

    def test_cross_rank_allreduce_lockstep_raise(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            allreduce_violation_signal=True,
        )
        runner = _make_runner(device=self.device, config=config)

        fake_group = SimpleNamespace(device_group=object())
        runner._tp_group = fake_group

        def fake_all_reduce(tensor, op, group):
            tensor.fill_(1)

        with patch.object(
            torch.distributed, "is_initialized", lambda: True
        ), patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            self.assertIsNotNone(runner._device_state.allreduce_buf)
            runner._device_state.allreduce_buf.fill_(0)
            torch.distributed.all_reduce(
                runner._device_state.allreduce_buf,
                op=torch.distributed.ReduceOp.MAX,
                group=fake_group.device_group,
            )
            self.assertEqual(int(runner._device_state.allreduce_buf.item()), 1)

    def test_kernel_run_counter_watchdog_raises_on_zero(self):
        runner = _make_runner(device=self.device)
        runner._step_counter = 1000
        runner._device_state.kernel_run_counters.zero_()
        with self.assertRaises(RuntimeError):
            runner.health_check_step()

    def test_runner_disabled_short_circuits(self):
        config = CanaryConfig(mode=CanaryMode.OFF)
        runner = _make_runner(device=self.device, config=config)

        plan_calls: List[str] = []
        with patch.object(
            runner_module,
            "canary_plan_step",
            lambda **kwargs: plan_calls.append("plan"),
        ):
            fb = _make_forward_batch(self.device)
            runner.before_forward(fb)
            runner.launch_head_kernels(fb)
            runner.launch_tail_kernels(fb)
            runner.maybe_run_sweep()
            runner.end_of_step()
        self.assertEqual(plan_calls, [])

    def test_periodic_stats_log_every_n_step(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            stats_print_every_n_steps=5,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        runner._device_state.slot_run_counters.fill_(7)

        with self.assertLogs(runner_module.logger.name, level=logging.INFO) as cm:
            for _ in range(10):
                runner._print_periodic_stats()
                runner._step_counter += 1
        log_text = "\n".join(cm.output)
        self.assertIn("protected_tokens=", log_text)
        self.assertTrue("step=5" in log_text or "step=10" in log_text)

    def test_per_forward_launches_both_head_and_tail(self):
        runner = _make_runner(device=self.device)

        head_count = sum(
            1
            for endpoints in runner._endpoints_per_pool
            for ep in endpoints
            if ep.kernel_kind
            in (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.HEAD_V_FULL)
        )
        tail_count = sum(
            1
            for endpoints in runner._endpoints_per_pool
            for ep in endpoints
            if ep.kernel_kind
            in (CanaryLaunchTag.TAIL_K_FULL, CanaryLaunchTag.TAIL_V_FULL)
        )
        self.assertGreaterEqual(head_count, 1)
        self.assertGreaterEqual(tail_count, 1)

        counters: List[str] = []
        with patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: counters.append(kwargs["kernel_kind"].name),
        ), patch.object(endpoint_module, "canary_write_step", lambda **kwargs: None):
            fb = _make_forward_batch(self.device)
            runner.before_forward(fb)
            runner.launch_head_kernels(fb)
            runner.launch_tail_kernels(fb)
        self.assertTrue(any("HEAD" in name for name in counters))
        self.assertTrue(any("TAIL" in name for name in counters))

    def test_sweep_path_detects_chain_mismatch(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_every_n_steps=1,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)
        runner.before_forward(fb)
        runner.launch_head_kernels(fb)

        cache = make_radix_cache([[], [10, 11, 12]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        sweep_kernel_kinds: List[str] = []
        with patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: sweep_kernel_kinds.append(kwargs["kernel_kind"].name),
        ):
            runner.maybe_run_sweep()
        self.assertTrue(any("SWEEP" in k for k in sweep_kernel_kinds))

    def test_runner_raises_when_other_rank_errored_but_local_clean(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            allreduce_violation_signal=True,
        )
        runner = _make_runner(device=self.device, config=config)
        runner._tp_group = SimpleNamespace(device_group=object())

        def lockstep_all_reduce(tensor, op, group):
            tensor.fill_(1)

        with patch.object(
            torch.distributed, "is_initialized", lambda: True
        ), patch.object(torch.distributed, "all_reduce", lockstep_all_reduce):
            self.assertEqual(
                int(runner._device_state.violation_log.violation_write_index.item()), 0
            )
            runner._device_state.allreduce_buf.fill_(0)
            torch.distributed.all_reduce(
                runner._device_state.allreduce_buf,
                op=torch.distributed.ReduceOp.MAX,
                group=runner._tp_group.device_group,
            )
            self.assertEqual(int(runner._device_state.allreduce_buf.item()), 1)


if __name__ == "__main__":
    unittest.main()
