from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
)
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner import canary_runner as runner_module
from sglang.srt.kv_canary.runner import launch as launch_module
from sglang.srt.kv_canary.runner.canary_runner import (
    CanaryLaunchCapacities,
    CanaryRunner,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    CPU_DEVICE,
    make_radix_cache,
    make_req_to_token_pool,
)
from sglang.test.test_utils import CustomTestCase

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


def _make_forward_batch(device, bs: int = 2, seq_lens_list=(3, 4)):
    seq_lens_list = list(seq_lens_list[:bs])
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_mixed=lambda: False),
        batch_size=bs,
        req_pool_indices=torch.tensor([1, 2][:bs], dtype=torch.int64, device=device),
        seq_lens=torch.tensor(seq_lens_list, dtype=torch.int32, device=device),
        seq_lens_sum=int(sum(seq_lens_list)),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        extend_prefix_lens_cpu=None,
        input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
        positions=torch.zeros(bs, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(bs, dtype=torch.int32, device=device),
    )


def _make_runner(
    *,
    device,
    config=None,
    group=None,
    req_pool=None,
    per_forward_verify_capacity: int = 16,
):
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
        buffer_groups=(group,),
        device=device,
        req_to_token_pool=req_pool,
        launch_capacities=CanaryLaunchCapacities(
            per_forward_verify_capacity=per_forward_verify_capacity,
            per_forward_write_req_capacity=2,
            per_forward_write_entry_capacity=8,
            sweep_verify_capacity=8,
        ),
    )


class TestSelfUnitRunner(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE
        # Stub plan/verify/write kernels so CPU runs don't need CUDA JIT.
        self._patchers = [
            patch.object(launch_module, "canary_plan_step", lambda **kwargs: None),
            patch.object(endpoint_module, "canary_verify_step", lambda **kwargs: None),
            patch.object(endpoint_module, "canary_write_step", lambda **kwargs: None),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_per_forward_orchestrates_plan_head_tail(self):
        calls: List = []
        with patch.object(
            launch_module,
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
            with runner.with_forward_pass(fb):
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
            sweep_interval=4,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)

        sweep_calls: List[int] = []
        real_maybe = runner._sweep_orchestrator.maybe_run_sweep

        def _spy():
            before = runner._sweep_orchestrator._last_sweep_step
            real_maybe()
            if runner._sweep_orchestrator._last_sweep_step != before:
                sweep_calls.append(runner._pump_and_allreduce._step_counter)

        with patch.object(runner._sweep_orchestrator, "maybe_run_sweep", _spy):
            for _ in range(12):
                with runner.with_forward_pass(fb):
                    pass
        self.assertEqual(sweep_calls, [0, 4, 8])

    def test_sweep_runs_radix_path(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_interval=1,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)
        with runner.with_forward_pass(fb):
            runner.launch_head_kernels(fb)

        cache = make_radix_cache([[], [10, 11]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        plan_calls: List[str] = []
        with patch.object(
            launch_module,
            "canary_plan_step",
            lambda **kwargs: plan_calls.append("plan"),
        ):
            runner._sweep_orchestrator.maybe_run_sweep()
        self.assertGreaterEqual(plan_calls.count("plan"), 1)

    def test_kernel_run_counter_watchdog_raises_on_zero(self):
        runner = _make_runner(device=self.device)
        runner._pump_and_allreduce._step_counter = 1000
        runner._device_state.kernel_run_counters.zero_()
        with self.assertRaises(RuntimeError):
            runner._health_and_stats.health_check_step()

    def test_runner_disabled_short_circuits(self):
        config = CanaryConfig(mode=CanaryMode.OFF)
        runner = _make_runner(device=self.device, config=config)

        plan_calls: List[str] = []
        with patch.object(
            launch_module,
            "canary_plan_step",
            lambda **kwargs: plan_calls.append("plan"),
        ):
            fb = _make_forward_batch(self.device)
            with runner.with_forward_pass(fb):
                runner.launch_head_kernels(fb)
                runner.launch_tail_kernels(fb)
            runner._sweep_orchestrator.maybe_run_sweep()
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
                runner._health_and_stats.print_periodic_stats()
                runner._pump_and_allreduce._step_counter += 1
        log_text = "\n".join(cm.output)
        self.assertIn("protected_tokens=", log_text)
        self.assertTrue("step=5" in log_text or "step=10" in log_text)

    def test_sweep_path_launches_sweep_kernels(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_interval=1,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        fb = _make_forward_batch(self.device)
        with runner.with_forward_pass(fb):
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
            runner._sweep_orchestrator.maybe_run_sweep()
        self.assertTrue(any("SWEEP" in k for k in sweep_kernel_kinds))

    def test_before_forward_throws_when_sum_prefix_lens_exceeds_verify_capacity(self):
        # Multi-req batch whose summed prefix lens exceeds any single req's length: with the old
        # sizing (per_forward_verify_capacity = max_seq_len_per_req) this slipped past the plan
        # kernel's silent cap_mask and OOB-read the verify kernel's tail threads. The runtime
        # check must throw (not silently execute) so the kernel never launches with a busted plan.
        runner = _make_runner(device=self.device, per_forward_verify_capacity=4)
        # decode: sum(prefix_lens) = (5 - 1) + (5 - 1) = 8 > capacity=4
        fb = _make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with self.assertRaisesRegex(RuntimeError, "sum\\(prefix_lens\\)=8"):
            with runner.with_forward_pass(fb):
                pass

    def test_before_forward_passes_when_sum_prefix_lens_fits(self):
        # Same multi-req shape that breaks the old sizing now fits the new capacity formula.
        runner = _make_runner(device=self.device, per_forward_verify_capacity=16)
        fb = _make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(fb):
            pass


class TestComputeLaunchCapacities(CustomTestCase):
    def test_per_forward_verify_capacity_covers_multi_req_prefix_sum(self):
        from sglang.srt.kv_canary.api import _compute_launch_capacities

        max_bs = 8
        max_seq_len = 64
        model_runner = SimpleNamespace(
            server_args=SimpleNamespace(
                cuda_graph_max_bs=max_bs,
                speculative_num_draft_tokens=0,
                chunked_prefill_size=None,
                max_prefill_tokens=128,
            ),
            req_to_token_pool=SimpleNamespace(
                size=max_bs,
                req_to_token=torch.zeros(max_bs, max_seq_len, dtype=torch.int32),
            ),
            max_total_num_tokens=max_bs * max_seq_len,
        )
        capacities = _compute_launch_capacities(model_runner=model_runner)
        # Old buggy sizing was max_seq_len (= 64); new sizing must fit the full table extent so a
        # multi-req batch with sum(prefix_lens) up to max_bs * max_seq_len never OOBs.
        self.assertGreaterEqual(
            capacities.per_forward_verify_capacity, max_bs * max_seq_len
        )


if __name__ == "__main__":
    unittest.main()
