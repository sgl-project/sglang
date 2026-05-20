from __future__ import annotations

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
    make_base_config,
    make_buffer_group,
)

from sglang.jit_kernel.kv_canary.verify import (  # noqa: E402
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode, WritePlan  # noqa: E402
from sglang.srt.kv_canary import endpoint as endpoint_module  # noqa: E402
from sglang.srt.kv_canary.endpoint import (  # noqa: E402
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.violation_state import (  # noqa: E402
    CanaryDeviceState,
    ViolationLog,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def _record_call(call_log: List, name: str):
    def _stub(**kwargs):
        call_log.append(name)
        call_log.append(kwargs)

    return _stub


def _make_endpoint(*, device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL, swa_lut=None):
    canary_buf = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    slot_view = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_view = torch.zeros(1, dtype=torch.int64, device=device)
    return CanaryEndpoint(
        kernel_kind=kernel_kind,
        canary_buf=canary_buf,
        full_to_swa_index_mapping=swa_lut,
        real_kv_sources=(),
        slot_run_counter_view=slot_view,
        kernel_run_counter_view=kernel_view,
    )


def _make_kernel_args(device):
    verify_plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    write_plan = WritePlan.allocate(write_req_capacity=1, device=device)
    log = ViolationLog.allocate(ring_capacity=2, device=device)
    return SimpleNamespace(
        verify_plan=verify_plan,
        write_plan=write_plan,
        violation_log=log,
        fb_input_ids=torch.zeros(1, dtype=torch.int32, device=device),
        fb_positions=torch.zeros(1, dtype=torch.int32, device=device),
        fb_out_cache_loc=torch.zeros(1, dtype=torch.int32, device=device),
        input_check_mode=CanaryPseudoMode.OFF,
        expected_input_tokens=torch.zeros(1, dtype=torch.int32, device=device),
        expected_input_positions=torch.zeros(1, dtype=torch.int32, device=device),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )


class TestSelfUnitEndpoint(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE

    def test_launch_per_forward_calls_verify_then_write(self):
        calls: List = []
        with patch.object(
            endpoint_module, "canary_verify_step", _record_call(calls, "verify")
        ), patch.object(
            endpoint_module, "canary_write_step", _record_call(calls, "write")
        ):
            ep = _make_endpoint(device=self.device)
            args = _make_kernel_args(self.device)
            ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                fb_input_ids=args.fb_input_ids,
                fb_positions=args.fb_positions,
                fb_out_cache_loc=args.fb_out_cache_loc,
                input_check_mode=args.input_check_mode,
                expected_input_tokens=args.expected_input_tokens,
                expected_input_positions=args.expected_input_positions,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        self.assertEqual(calls[0], "verify")
        self.assertEqual(calls[2], "write")

    def test_launch_per_forward_passes_kernel_kind(self):
        captured: List = []
        with patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: captured.append(("verify", kwargs["kernel_kind"])),
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: captured.append(("write", kwargs["kernel_kind"])),
        ):
            ep = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.TAIL_V_SWA
            )
            args = _make_kernel_args(self.device)
            ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                fb_input_ids=args.fb_input_ids,
                fb_positions=args.fb_positions,
                fb_out_cache_loc=args.fb_out_cache_loc,
                input_check_mode=args.input_check_mode,
                expected_input_tokens=args.expected_input_tokens,
                expected_input_positions=args.expected_input_positions,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        self.assertIn(("verify", CanaryLaunchTag.TAIL_V_SWA), captured)
        self.assertIn(("write", CanaryLaunchTag.TAIL_V_SWA), captured)

    def test_launch_sweep_only_calls_verify(self):
        calls: List[str] = []
        with patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: calls.append("verify"),
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: calls.append("write"),
        ):
            ep = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.SWEEP_K_FULL
            )
            args = _make_kernel_args(self.device)
            ep.launch_sweep(
                verify_plan=args.verify_plan,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        self.assertEqual(calls, ["verify"])

    def test_endpoint_shares_violation_log_across_launches(self):
        captured_rings: List[int] = []
        with patch.object(
            endpoint_module,
            "canary_verify_step",
            lambda **kwargs: captured_rings.append(kwargs["violation_ring"].data_ptr()),
        ), patch.object(endpoint_module, "canary_write_step", lambda **kwargs: None):
            shared_log = ViolationLog.allocate(ring_capacity=2, device=self.device)
            ep_a = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.SWEEP_K_FULL
            )
            ep_b = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.SWEEP_V_FULL
            )

            plan = VerifyPlan.allocate(verify_capacity=1, device=self.device)
            ep_a.launch_sweep(
                verify_plan=plan,
                violation_log=shared_log,
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
            ep_b.launch_sweep(
                verify_plan=plan,
                violation_log=shared_log,
                real_kv_hash_mode=RealKvHashMode.OFF,
            )
        self.assertEqual(captured_rings[0], captured_rings[1])
        self.assertEqual(captured_rings[0], shared_log.violation_ring.data_ptr())

    def test_swa_endpoint_passes_lut(self):
        captured: List = []
        with patch.object(
            endpoint_module, "canary_verify_step", lambda **kwargs: None
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: captured.append(kwargs["full_to_swa_index_mapping"]),
        ):
            lut = torch.arange(8, dtype=torch.int32, device=self.device)
            swa_ep = _make_endpoint(
                device=self.device,
                kernel_kind=CanaryLaunchTag.HEAD_K_SWA,
                swa_lut=lut,
            )
            full_ep = _make_endpoint(
                device=self.device,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                swa_lut=None,
            )
            args = _make_kernel_args(self.device)

            swa_ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                fb_input_ids=args.fb_input_ids,
                fb_positions=args.fb_positions,
                fb_out_cache_loc=args.fb_out_cache_loc,
                input_check_mode=args.input_check_mode,
                expected_input_tokens=args.expected_input_tokens,
                expected_input_positions=args.expected_input_positions,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
            full_ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                fb_input_ids=args.fb_input_ids,
                fb_positions=args.fb_positions,
                fb_out_cache_loc=args.fb_out_cache_loc,
                input_check_mode=args.input_check_mode,
                expected_input_tokens=args.expected_input_tokens,
                expected_input_positions=args.expected_input_positions,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        self.assertIs(captured[0], lut)
        self.assertIsNone(captured[1])

    def test_head_tail_share_class(self):
        group = make_buffer_group(self.device)
        device_state = CanaryDeviceState.allocate(
            config=make_base_config(),
            device=self.device,
            num_tags=len(CanaryLaunchTag),
        )
        endpoints = build_endpoints_from_group(group=group, device_state=device_state)

        head = next(
            ep for ep in endpoints if ep.kernel_kind == CanaryLaunchTag.HEAD_K_FULL
        )
        tail = next(
            ep for ep in endpoints if ep.kernel_kind == CanaryLaunchTag.TAIL_K_FULL
        )

        self.assertIs(type(head), type(tail))


if __name__ == "__main__":
    unittest.main()
