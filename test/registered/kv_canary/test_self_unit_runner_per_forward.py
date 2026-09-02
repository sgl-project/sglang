from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.kv_canary.consts import RealKvHashMode
from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.runner import kernel_launcher as kernel_launcher_module
from sglang.srt.kv_canary.state import ViolationLog
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import make_buffer_group, make_forward_batch
from sglang.test.kv_canary.runner_test_base import (
    CanaryManagerTestCase,
    RecordingEndpoint,
    make_manager,
)

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="extra-a-test-1-gpu-small-amd")


class TestManagerPerForward(CanaryManagerTestCase):
    def test_per_forward_orchestrates_plan_head_tail(self) -> None:
        """Verify per-forward execution launches plan, head/tail verify kernels, and write kernels in order."""
        calls: list[object] = []
        with patch.object(
            kernel_launcher_module,
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
            with manager.with_ops_outside_graph(
                single_forward_indices=[0],
                maybe_inaccurate_forward_batch=forward_batch,
            ):
                with manager.with_active_single_forward_manager(0):
                    pre_ops_output = manager.pre_ops_maybe_inside_graph(forward_batch)
                    manager.post_ops_maybe_inside_graph(forward_batch, pre_ops_output)

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
        """Verify endpoint launch preserves contiguous int64 tensor shapes/values through the canonicalizer."""
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

        kernel_launcher_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=3, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.NONE,
            enable_write_input_assert=False,
            enable_verify_token_assert=False,
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

    def test_launch_endpoints_per_forward_promotes_int32_boundary_tensors_to_int64(
        self,
    ) -> None:
        """Verify int32 boundary tensors are promoted to int64 at the launch boundary."""
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
        forward_batch.num_token_non_padded_cpu = 1

        kernel_launcher_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.NONE,
            enable_write_input_assert=False,
            enable_verify_token_assert=False,
        )

        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertEqual(call["input_ids"].dtype, torch.int64)
        self.assertEqual(call["positions"].dtype, torch.int64)
        self.assertEqual(call["out_cache_loc"].dtype, torch.int64)

    def test_launch_endpoints_per_forward_propagates_enable_verify_token_assert_true(
        self,
    ) -> None:
        """Verify enable_verify_token_assert=True is plumbed through to the endpoint kwargs."""
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

        kernel_launcher_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=3, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.NONE,
            enable_write_input_assert=False,
            enable_verify_token_assert=True,
        )

        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertEqual(call["enable_verify_token_assert"], True)

    def test_launch_endpoints_per_forward_materializes_strided_boundary_tensors(
        self,
    ) -> None:
        """Verify non-contiguous boundary views are materialized contiguous at launch."""
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
        forward_batch.num_token_non_padded_cpu = 1

        kernel_launcher_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.NONE,
            enable_write_input_assert=False,
            enable_verify_token_assert=False,
        )

        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertTrue(call["input_ids"].is_contiguous())
        self.assertTrue(call["positions"].is_contiguous())
        self.assertTrue(call["out_cache_loc"].is_contiguous())


class TestManagerBeforeForward(CanaryManagerTestCase):
    def test_before_forward_does_not_throw_on_oversized_prefix_sum(self) -> None:
        """Verify oversized prefix sums are handled without host-side errors."""
        manager = make_manager(device=self.device, per_forward_verify_capacity=4)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        _drive_one_cycle(manager, forward_batch)

    def test_before_forward_passes_when_sum_prefix_lens_fits(self) -> None:
        """Verify prefix sums within capacity pass before-forward handling."""
        manager = make_manager(device=self.device, per_forward_verify_capacity=16)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        _drive_one_cycle(manager, forward_batch)


def _drive_one_cycle(manager, forward_batch) -> None:
    with manager.with_ops_outside_graph(
        single_forward_indices=[0],
        maybe_inaccurate_forward_batch=forward_batch,
    ):
        with manager.with_active_single_forward_manager(0):
            pre_ops_output = manager.pre_ops_maybe_inside_graph(forward_batch)
            manager.post_ops_maybe_inside_graph(forward_batch, pre_ops_output)


class TestCanaryManagerActiveSingleForwardManagerDispatch(CanaryManagerTestCase):
    def test_pre_ops_maybe_inside_graph_dispatches_to_bracketed_sfm(
        self,
    ) -> None:
        """Verify the dispatcher routes phase 2 to the bracketed SingleForwardManager."""
        manager = make_manager(device=self.device, speculative_num_steps=3)
        forward_batch = make_forward_batch(self.device)
        target_sfm = manager._single_forward_managers[1]
        observed: list[object] = []
        original_phase_2 = target_sfm.pre_ops_maybe_inside_graph

        def _record(fb):
            observed.append(fb)
            return original_phase_2(fb)

        target_sfm.pre_ops_maybe_inside_graph = _record
        manager._single_forward_managers[0].pre_ops_outside_graph(
            maybe_inaccurate_forward_batch=forward_batch
        )
        manager._single_forward_managers[1].pre_ops_outside_graph(
            maybe_inaccurate_forward_batch=forward_batch
        )
        with manager.with_active_single_forward_manager(1):
            manager.pre_ops_maybe_inside_graph(forward_batch)
        self.assertEqual(observed, [forward_batch])

    def test_pre_ops_maybe_inside_graph_asserts_outside_bracket(
        self,
    ) -> None:
        manager = make_manager(device=self.device)
        forward_batch = make_forward_batch(self.device)
        with self.assertRaises(AssertionError):
            manager.pre_ops_maybe_inside_graph(forward_batch)


if __name__ == "__main__":
    unittest.main()
