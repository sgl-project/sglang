from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.runner import kernel_launch as kernel_launch_module
from sglang.srt.kv_canary.state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import make_buffer_group, make_forward_batch
from sglang.test.kv_canary.runner_test_base import (
    CanaryManagerTestCase,
    RecordingEndpoint,
    make_manager,
)

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestManagerPerForward(CanaryManagerTestCase):
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
            manager = make_manager(device=self.device)
            forward_batch = make_forward_batch(self.device)
            sfm = manager.get_single_forward_manager(0)
            sfm.pre_ops_outside_graph(maybe_non_mature_forward_batch=forward_batch)
            with manager.with_single_forward_manager_index(0):
                sfm.pre_ops_maybe_inside_graph(forward_batch)
                sfm.post_ops_maybe_inside_graph(forward_batch)
            sfm.post_ops_outside_graph(snapshot=sfm.snapshot)

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


class TestLaunchEndpointsPerForward(CanaryManagerTestCase):
    def test_launch_endpoints_per_forward_keeps_padded_token_tensors(self) -> None:
        """Verify endpoint launch preserves CUDA graph-stable tensor shapes."""
        group = make_buffer_group(device=self.device)
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

    def test_launch_endpoints_per_forward_rejects_int32_boundary_tensors(self) -> None:
        """Phase 2 must fail fast on int32 ForwardBatch tensors.

        Under the SFM design the kernel-launch boundary asserts int64
        contiguous rather than silently allocating a converted copy
        (capture-unsafe). This pins the new fail-fast behavior so the
        contract cannot regress to the old promote-and-copy form.
        """
        group = make_buffer_group(device=self.device)
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

        with self.assertRaises(AssertionError):
            kernel_launch_module.launch_endpoints_per_forward(
                endpoints=(endpoint,),
                group=group,
                tag_filter=lambda tag: True,
                verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
                write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
                forward_batch=forward_batch,
                expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
                violation_log=ViolationLog.allocate(
                    ring_capacity=2, device=self.device
                ),
                real_kv_hash_mode=RealKvHashMode.OFF,
                input_check_mode=False,
            )

    def test_launch_endpoints_per_forward_rejects_strided_boundary_tensors(
        self,
    ) -> None:
        """Phase 2 must fail fast on non-contiguous ForwardBatch views.

        Same rationale as the int32 case — capture-safety requires the
        upstream phase-1 hook to provide already-contiguous tensors.
        """
        group = make_buffer_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [[101, 102]], dtype=torch.int64, device=self.device
        )[:, 0]
        forward_batch.positions = torch.tensor(
            [[10, 11]], dtype=torch.int64, device=self.device
        )[:, 0]
        forward_batch.out_cache_loc = torch.tensor(
            [[7, 8]], dtype=torch.int64, device=self.device
        )[:, 0]

        with self.assertRaises(AssertionError):
            kernel_launch_module.launch_endpoints_per_forward(
                endpoints=(endpoint,),
                group=group,
                tag_filter=lambda tag: True,
                verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
                write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
                forward_batch=forward_batch,
                expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
                violation_log=ViolationLog.allocate(
                    ring_capacity=2, device=self.device
                ),
                real_kv_hash_mode=RealKvHashMode.OFF,
                input_check_mode=False,
            )


class TestManagerBeforeForward(CanaryManagerTestCase):
    def test_before_forward_does_not_throw_on_oversized_prefix_sum(self) -> None:
        """Verify oversized prefix sums are handled without host-side errors."""
        # Overflow no longer raises host-side: the plan kernel sets VerifyPlan.enable=0 and the
        # verify kernel skips the step on-device; host logs a throttled warning instead.
        manager = make_manager(device=self.device, per_forward_verify_capacity=4)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        _drive_one_cycle(manager, forward_batch)

    def test_before_forward_passes_when_sum_prefix_lens_fits(self) -> None:
        """Verify prefix sums within capacity pass before-forward handling."""
        # Same multi-req shape that breaks the old sizing now fits the new capacity formula.
        manager = make_manager(device=self.device, per_forward_verify_capacity=16)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        _drive_one_cycle(manager, forward_batch)


def _drive_one_cycle(manager, forward_batch) -> None:
    sfm = manager.get_single_forward_manager(0)
    sfm.pre_ops_outside_graph(maybe_non_mature_forward_batch=forward_batch)
    with manager.with_single_forward_manager_index(0):
        sfm.pre_ops_maybe_inside_graph(forward_batch)
        sfm.post_ops_maybe_inside_graph(forward_batch)
    sfm.post_ops_outside_graph(snapshot=sfm.snapshot)


if __name__ == "__main__":
    unittest.main()
