from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.plan_ref import canary_plan_step_torch_reference
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    canary_verify_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner import canary_runner as runner_module
from sglang.srt.kv_canary.runner import launch as launch_module
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
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


class _FakeDecodeForwardMode:
    def is_extend(self) -> bool:
        return False

    def is_mixed(self) -> bool:
        return False

    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend_v2(self) -> bool:
        return False

    def is_extend_or_draft_extend_or_mixed(self) -> bool:
        return False


def _make_forward_batch(device, bs: int = 2, seq_lens_list=(3, 4)):
    seq_lens_list = list(seq_lens_list[:bs])
    return SimpleNamespace(
        forward_mode=_FakeDecodeForwardMode(),
        spec_info=None,
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
    sweep_verify_capacity: int = 8,
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
            sweep_verify_capacity=sweep_verify_capacity,
        ),
    )


class TestSelfUnitRunner(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE
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

    def test_req_to_token_perturb_uses_live_slot_as_replacement(self):
        pool = _make_pool(self.device, max_reqs=4, max_seq=8)
        pool.req_to_token[1, :3] = torch.tensor(
            [11, 22, 33], dtype=torch.int32, device=self.device
        )
        pool.req_to_token[2, :3] = torch.tensor(
            [44, 55, 66], dtype=torch.int32, device=self.device
        )
        manager = PerturbManager(
            config=PerturbConfig(req_to_token_prob=1.0, warmup_steps=0),
            req_to_token_pool=pool,
            buffer_groups=(),
            pump_and_allreduce=SimpleNamespace(step_counter=10),
        )
        forward_batch = _make_forward_batch(self.device, bs=2, seq_lens_list=(3, 3))

        snapshot = pool.req_to_token.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            manager.perturb_req_to_token(forward_batch)

        diff = pool.req_to_token != snapshot
        self.assertEqual(int(diff.sum().item()), 1)
        rows, cols = torch.nonzero(diff, as_tuple=True)
        row, col = int(rows[0].item()), int(cols[0].item())
        original = int(snapshot[row, col].item())
        replacement = int(pool.req_to_token[row, col].item())
        live_slots = {11, 22, 33, 44, 55, 66}
        self.assertIn(original, live_slots)
        self.assertIn(replacement, live_slots)
        self.assertNotEqual(replacement, original)

    def test_kernel_run_counter_watchdog_raises_on_zero(self):
        runner = _make_runner(device=self.device)
        runner._pump_and_allreduce._step_counter = 1000
        runner._device_state.kernel_run_counters.zero_()
        runner._health_and_stats.health_check_step()
        runner._pump_and_allreduce._step_counter = 2000
        with self.assertRaises(RuntimeError):
            runner._health_and_stats.health_check_step()

    def test_kernel_run_counter_watchdog_ignores_sweep_when_sweep_is_disabled(self):
        config = CanaryConfig(
            mode=CanaryMode.RAISE,
            real_kv_hash_mode=RealKvHashMode.OFF,
            sweep_interval=0,
            allreduce_violation_signal=False,
        )
        runner = _make_runner(device=self.device, config=config)
        runner._device_state.kernel_run_counters.zero_()
        for tag in (
            CanaryLaunchTag.HEAD_K_FULL,
            CanaryLaunchTag.HEAD_V_FULL,
            CanaryLaunchTag.TAIL_K_FULL,
            CanaryLaunchTag.TAIL_V_FULL,
        ):
            runner._device_state.kernel_run_counters[tag.value] = 1

        runner._pump_and_allreduce._step_counter = 1000
        runner._health_and_stats.health_check_step()
        runner._pump_and_allreduce._step_counter = 2000
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
            for _ in range(11):
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

    def test_before_forward_does_not_throw_on_oversized_prefix_sum(self):
        # Overflow no longer raises host-side: the plan kernel sets VerifyPlan.enable=0 and the
        # verify kernel skips the step on-device; host logs a throttled warning instead.
        runner = _make_runner(device=self.device, per_forward_verify_capacity=4)
        fb = _make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(fb):
            pass

    def test_before_forward_passes_when_sum_prefix_lens_fits(self):
        # Same multi-req shape that breaks the old sizing now fits the new capacity formula.
        runner = _make_runner(device=self.device, per_forward_verify_capacity=16)
        fb = _make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(fb):
            pass

    def test_sweep_throws_when_walker_output_exceeds_sweep_capacity(self):
        runner = _make_runner(device=self.device, sweep_verify_capacity=1)
        cache = make_radix_cache([[], [10, 11], [12, 13, 14]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        with self.assertRaisesRegex(
            RuntimeError, r"radix-walker emitted .* sweep verify entries"
        ):
            runner._sweep_orchestrator.maybe_run_sweep()


class TestComputeLaunchCapacities(CustomTestCase):
    @staticmethod
    def _make_server_args(*, max_bs):
        return SimpleNamespace(
            cuda_graph_max_bs=max_bs,
            speculative_num_draft_tokens=0,
            chunked_prefill_size=None,
            max_prefill_tokens=128,
        )

    @staticmethod
    def _from_args(*, max_bs, max_seq_len, max_total_num_tokens=None):
        if max_total_num_tokens is None:
            max_total_num_tokens = max_bs * max_seq_len
        return CanaryLaunchCapacities.from_args(
            server_args=TestComputeLaunchCapacities._make_server_args(max_bs=max_bs),
            req_to_token_pool_size=max_bs,
            max_seq_len_per_req=max_seq_len,
            pool_slot_count=max_total_num_tokens,
        )

    def test_per_forward_verify_capacity_covers_multi_req_prefix_sum(self):
        max_bs = 8
        max_seq_len = 64
        max_total_num_tokens = 1024
        capacities = self._from_args(
            max_bs=max_bs,
            max_seq_len=max_seq_len,
            max_total_num_tokens=max_total_num_tokens,
        )
        self.assertEqual(
            capacities.per_forward_verify_capacity,
            max(1, int(max_total_num_tokens * 1.2)),
        )


class TestPlanRefOverflowGate(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE

    @staticmethod
    def _empty_extras(device):
        return (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    def _run_plan_ref(
        self, *, verify_capacity: int, bs: int, prefix_lens: list[int]
    ) -> VerifyPlan:
        max_seq_len = max(prefix_lens) + 1
        verify_plan = VerifyPlan.allocate(
            verify_capacity=verify_capacity, device=self.device
        )
        write_plan = WritePlan.allocate(write_req_capacity=bs, device=self.device)
        fb_req_pool_indices = torch.tensor(
            list(range(1, bs + 1)), dtype=torch.int32, device=self.device
        )
        fb_prefix_lens = torch.tensor(
            prefix_lens, dtype=torch.int32, device=self.device
        )
        fb_extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=self.device)
        req_to_token = torch.arange(
            (bs + 1) * max_seq_len, dtype=torch.int32, device=self.device
        ).reshape(bs + 1, max_seq_len)
        extras = self._empty_extras(self.device)
        canary_plan_step_torch_reference(
            verify_plan_out=verify_plan,
            write_plan_out=write_plan,
            fb_req_pool_indices=fb_req_pool_indices,
            fb_prefix_lens=fb_prefix_lens,
            fb_extend_seq_lens=fb_extend_seq_lens,
            req_to_token=req_to_token,
            extra_verify_slot_indices=extras[0],
            extra_verify_positions=extras[1],
            extra_verify_prev_slot_indices=extras[2],
            extra_verify_num_valid=extras[3],
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
        )
        return verify_plan

    def test_plan_ref_sets_enable_zero_and_clamps_when_overflow(self):
        # requested = sum(prefix_lens) = 8 > capacity = 4.
        plan = self._run_plan_ref(verify_capacity=4, bs=2, prefix_lens=[5, 5])
        self.assertEqual(int(plan.enable[0].item()), 0)
        self.assertEqual(int(plan.verify_num_valid[0].item()), 4)

    def test_plan_ref_sets_enable_one_when_within_capacity(self):
        # requested = 4 <= capacity = 16.
        plan = self._run_plan_ref(verify_capacity=16, bs=2, prefix_lens=[2, 2])
        self.assertEqual(int(plan.enable[0].item()), 1)
        self.assertEqual(int(plan.verify_num_valid[0].item()), 4)

    def test_verify_ref_skips_when_enable_zero(self):
        plan = self._run_plan_ref(verify_capacity=4, bs=2, prefix_lens=[5, 5])
        self.assertEqual(int(plan.enable[0].item()), 0)

        canary_buf = torch.zeros(64, 32, dtype=torch.uint8, device=self.device)
        violation_ring = torch.zeros(
            4, consts.VIOLATION_FIELDS, dtype=torch.int64, device=self.device
        )
        violation_write_index = torch.zeros(1, dtype=torch.int32, device=self.device)
        slot_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        ring_before = violation_ring.clone()

        canary_verify_step_torch_reference(
            canary_buf=canary_buf,
            plan=plan,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=violation_ring,
            violation_write_index=violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )
        # kernel_run_counter bumps regardless of enable; everything else must be untouched.
        self.assertEqual(int(kernel_run_counter[0].item()), 1)
        self.assertEqual(int(violation_write_index[0].item()), 0)
        self.assertEqual(int(slot_run_counter[0].item()), 0)
        self.assertTrue(torch.equal(violation_ring, ring_before))


if __name__ == "__main__":
    unittest.main()
