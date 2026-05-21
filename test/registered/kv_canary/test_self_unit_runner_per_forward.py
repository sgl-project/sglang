from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from kv_canary_runner_unit_utils import (
    CanaryRunnerTestCase,
    RecordingEndpoint,
    make_forward_batch,
    make_group,
    make_runner,
)

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.runner import kernel_launch as kernel_launch_module
from sglang.srt.kv_canary.state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitRunnerPerForward(CanaryRunnerTestCase):
    def test_per_forward_orchestrates_plan_head_tail(self) -> None:
        """Verify per-forward execution launches plan, head, and tail kernels."""
        calls: list[object] = []
        with patch.object(
            kernel_launch_module,
            "launch_canary_plan_kernels",
            lambda **kwargs: calls.append("plan"),
        ), patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: calls.append(
                ("verify", kwargs["context"].kernel_kind.name)
            ),
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: calls.append(
                ("write", kwargs["context"].kernel_kind.name)
            ),
        ):
            runner = make_runner(device=self.device)
            forward_batch = make_forward_batch(self.device)
            with runner.with_forward_pass(forward_batch):
                runner.launch_head_kernels(forward_batch)
                runner.launch_tail_kernels(forward_batch)

        self.assertEqual(calls[0], "plan")
        self.assertTrue(
            any(
                call[0] == "verify" and "HEAD" in call[1]
                for call in calls[1:]
                if isinstance(call, tuple)
            )
        )
        self.assertTrue(
            any(
                call[0] == "verify" and "TAIL" in call[1]
                for call in calls[1:]
                if isinstance(call, tuple)
            )
        )

    def test_launch_endpoints_per_forward_keeps_padded_token_tensors(self) -> None:
        """Verify endpoint launch preserves CUDA graph-stable tensor shapes."""
        group = make_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [101, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.positions = torch.tensor(
            [10, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.out_cache_loc = torch.tensor(
            [7, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.num_token_non_padded_cpu = 1

        kernel_launch_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=3, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.OFF,
            input_check_mode=False,
        )

        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertTrue(
            torch.equal(
                call["input_ids"],
                torch.tensor([101, 0, 0], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                call["positions"],
                torch.tensor([10, 0, 0], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                call["out_cache_loc"],
                torch.tensor([7, 0, 0], dtype=torch.int64, device=self.device),
            )
        )

    def test_launch_endpoints_per_forward_accepts_int32_boundary_tensors(self) -> None:
        """Verify int32 ForwardBatch tensors are promoted at the canary boundary."""
        group = make_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [101], dtype=torch.int32, device=self.device
        )
        forward_batch.positions = torch.tensor(
            [10], dtype=torch.int32, device=self.device
        )
        forward_batch.out_cache_loc = torch.tensor(
            [7], dtype=torch.int32, device=self.device
        )

        kernel_launch_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.OFF,
            input_check_mode=False,
        )

        call = endpoint.calls[0]
        self.assertEqual(call["input_ids"].dtype, torch.int64)
        self.assertEqual(call["positions"].dtype, torch.int64)
        self.assertEqual(call["out_cache_loc"].dtype, torch.int64)

    def test_before_forward_does_not_throw_on_oversized_prefix_sum(self) -> None:
        """Verify oversized prefix sums are handled without host-side errors."""
        # Overflow no longer raises host-side: the plan kernel sets VerifyPlan.enable=0 and the
        # verify kernel skips the step on-device; host logs a throttled warning instead.
        runner = make_runner(device=self.device, per_forward_verify_capacity=4)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(forward_batch):
            pass

    def test_before_forward_passes_when_sum_prefix_lens_fits(self) -> None:
        """Verify prefix sums within capacity pass before-forward handling."""
        # Same multi-req shape that breaks the old sizing now fits the new capacity formula.
        runner = make_runner(device=self.device, per_forward_verify_capacity=16)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(forward_batch):
            pass


if __name__ == "__main__":
    unittest.main()
