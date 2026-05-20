from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode, WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.violation_state import (
    CanaryDeviceState,
    ViolationLog,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    CPU_DEVICE,
    make_base_config,
    make_buffer_group,
)
from sglang.test.test_utils import CustomTestCase

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

    def test_swa_endpoint_pre_translates_fb_out_cache_loc(self):
        """SWA endpoint host-gathers ``lut[fb_out_cache_loc]`` before calling canary_write_step; FULL
        endpoint passes fb_out_cache_loc through unchanged. The kernel never sees a LUT.
        """
        captured: List[torch.Tensor] = []
        with patch.object(
            endpoint_module, "canary_verify_step", lambda **kwargs: None
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: captured.append(kwargs["fb_out_cache_loc"]),
        ):
            # LUT maps full slot i → swa slot (i + 100) so we can verify the gather happened.
            lut = (torch.arange(8, dtype=torch.int32, device=self.device) + 100).to(
                torch.int32
            )
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
        # SWA call: fb_out_cache_loc was rewritten via lut gather (so identity-shifted by +100 here).
        expected_swa = lut[args.fb_out_cache_loc.to(torch.int64)].to(torch.int32)
        self.assertTrue(torch.equal(captured[0], expected_swa))
        # FULL call: fb_out_cache_loc passed through unchanged.
        self.assertIs(captured[1], args.fb_out_cache_loc)

    def test_swa_endpoint_trailing_sentinel_row_yields_skip(self):
        """LUT trailing-row sentinel (-1) propagates through the host gather to the kernel as a skip
        signal. Regression for the post-write-SWA-decouple contract: the endpoint must turn a
        pre-cleanup "kernel-side LUT[full_pool_size] = -1 → skip" path into "host gather → -1 → skip"
        without losing the sentinel semantics. (Replaces the kernel-side coverage of the deleted
        test_full_to_swa_lut_sentinel_skips_entry; the kernel-side -1 → skip path itself is covered by
        test_negative_slot_skips_entry in test_write.py.)
        """
        captured: List[torch.Tensor] = []
        with patch.object(
            endpoint_module, "canary_verify_step", lambda **kwargs: None
        ), patch.object(
            endpoint_module,
            "canary_write_step",
            lambda **kwargs: captured.append(kwargs["fb_out_cache_loc"]),
        ):
            # 8 in-window rows + 1 trailing sentinel row at index 8.
            lut = torch.arange(8, dtype=torch.int32, device=self.device)
            lut = torch.cat(
                [lut, torch.tensor([-1], dtype=torch.int32, device=self.device)]
            )
            swa_ep = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.HEAD_K_SWA, swa_lut=lut
            )
            args = _make_kernel_args(self.device)
            # Point fb_out_cache_loc at the trailing-sentinel-row index — this is how sglang signals
            # "this token is out-of-window for the SWA group" pre-cleanup, and the new host gather must
            # produce -1 here.
            args.fb_out_cache_loc.fill_(8)

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

        self.assertTrue(
            torch.equal(
                captured[0],
                torch.tensor([-1], dtype=torch.int32, device=self.device),
            )
        )

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
