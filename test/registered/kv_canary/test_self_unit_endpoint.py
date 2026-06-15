from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
)
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.state import (
    ViolationLog,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="extra-a-test-1-gpu-small-amd")


def _make_endpoint(*, device, kernel_kind=CanaryLaunchTag.HEAD_K_FULL, swa_lut=None):
    canary_buf = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    slot_view = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_view = torch.zeros(1, dtype=torch.int64, device=device)
    enable_chain_position_assert = torch.ones(1, dtype=torch.int32, device=device)
    return CanaryEndpoint(
        kernel_kind=kernel_kind,
        canary_buf=canary_buf,
        full_to_swa_index_mapping=swa_lut,
        real_kv_sources=(),
        slot_run_counter_view=slot_view,
        kernel_run_counter_view=kernel_view,
        enable_chain_position_assert=enable_chain_position_assert,
    )


def _make_kernel_args(device):
    verify_plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    write_plan = WritePlan.allocate(write_req_capacity=1, device=device)
    log = ViolationLog.allocate(ring_capacity=2, device=device)
    return SimpleNamespace(
        verify_plan=verify_plan,
        write_plan=write_plan,
        violation_log=log,
        input_ids=torch.zeros(1, dtype=torch.int64, device=device),
        positions=torch.zeros(1, dtype=torch.int64, device=device),
        out_cache_loc=torch.zeros(1, dtype=torch.int64, device=device),
        enable_write_input_assert=False,
        enable_verify_token_assert=False,
        expected_inputs=ExpectedInputs.allocate(capacity=1, device=device),
        real_kv_hash_mode=RealKvHashMode.NONE,
    )


class TestSelfUnitEndpoint(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE

    def test_launch_sweep_only_calls_verify(self):
        """Verify sweep launch invokes only the verify kernel."""
        calls: list[str] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: calls.append("verify"),
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
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

    def test_launch_per_forward_passes_kernel_kind(self):
        """Verify per-forward launch passes the endpoint kernel kind."""
        captured: list[tuple[str, CanaryLaunchTag]] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: captured.append(("verify", kwargs["context"].kernel_kind)),
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: captured.append(("write", kwargs["context"].kernel_kind)),
        ):
            ep = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.TAIL_V_SWA
            )
            args = _make_kernel_args(self.device)
            ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                input_ids=args.input_ids,
                positions=args.positions,
                out_cache_loc=args.out_cache_loc,
                enable_write_input_assert=args.enable_write_input_assert,
                enable_verify_token_assert=args.enable_verify_token_assert,
                expected_inputs=args.expected_inputs,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        self.assertIn(("verify", CanaryLaunchTag.TAIL_V_SWA), captured)
        self.assertIn(("write", CanaryLaunchTag.TAIL_V_SWA), captured)

    def test_endpoint_shares_violation_log_across_launches(self):
        """Verify endpoints can reuse the same violation log."""
        captured_rings: list[int] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: captured_rings.append(
                kwargs["context"].violation_ring.data_ptr()
            ),
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: None,
        ):
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
                real_kv_hash_mode=RealKvHashMode.NONE,
            )
            ep_b.launch_sweep(
                verify_plan=plan,
                violation_log=shared_log,
                real_kv_hash_mode=RealKvHashMode.NONE,
            )
        self.assertEqual(captured_rings[0], captured_rings[1])
        self.assertEqual(captured_rings[0], shared_log.violation_ring.data_ptr())

    def test_swa_endpoint_pre_translates_out_cache_loc(self):
        """Verify SWA endpoints translate cache locations before write launch."""
        captured: list[torch.Tensor] = []
        with patch.object(
            endpoint_module, "launch_canary_verify_kernel", lambda **kwargs: None
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: captured.append(kwargs["out_cache_loc"]),
        ):
            # LUT maps full slot i → swa slot (i + 100) so we can verify the gather happened.
            lut = torch.arange(8, dtype=torch.int64, device=self.device) + 100
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
                input_ids=args.input_ids,
                positions=args.positions,
                out_cache_loc=args.out_cache_loc,
                enable_write_input_assert=args.enable_write_input_assert,
                enable_verify_token_assert=args.enable_verify_token_assert,
                expected_inputs=args.expected_inputs,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
            full_ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                input_ids=args.input_ids,
                positions=args.positions,
                out_cache_loc=args.out_cache_loc,
                enable_write_input_assert=args.enable_write_input_assert,
                enable_verify_token_assert=args.enable_verify_token_assert,
                expected_inputs=args.expected_inputs,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )
        # SWA call: out_cache_loc was rewritten via lut gather (so identity-shifted by +100 here).
        expected_swa = lut[args.out_cache_loc]
        self.assertTrue(torch.equal(captured[0], expected_swa))
        # FULL call: out_cache_loc keeps the same values and dtype.
        self.assertIs(captured[1], args.out_cache_loc)

    def test_swa_endpoint_trailing_sentinel_row_yields_skip(self):
        """Verify SWA sentinel cache rows become write-skip markers."""
        captured: list[torch.Tensor] = []
        with patch.object(
            endpoint_module, "launch_canary_verify_kernel", lambda **kwargs: None
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: captured.append(kwargs["out_cache_loc"]),
        ):
            # 8 in-window rows + 1 trailing sentinel row at index 8.
            lut = torch.arange(8, dtype=torch.int64, device=self.device)
            lut = torch.cat(
                [lut, torch.tensor([-1], dtype=torch.int64, device=self.device)]
            )
            swa_ep = _make_endpoint(
                device=self.device, kernel_kind=CanaryLaunchTag.HEAD_K_SWA, swa_lut=lut
            )
            args = _make_kernel_args(self.device)
            # Point out_cache_loc at the trailing-sentinel-row index — this is how sglang signals
            # "this token is out-of-window for the SWA group" pre-cleanup, and the new host gather must
            # produce -1 here.
            args.out_cache_loc.fill_(8)

            swa_ep.launch_per_forward(
                verify_plan=args.verify_plan,
                write_plan=args.write_plan,
                input_ids=args.input_ids,
                positions=args.positions,
                out_cache_loc=args.out_cache_loc,
                enable_write_input_assert=args.enable_write_input_assert,
                enable_verify_token_assert=args.enable_verify_token_assert,
                expected_inputs=args.expected_inputs,
                violation_log=args.violation_log,
                real_kv_hash_mode=args.real_kv_hash_mode,
            )

        self.assertTrue(
            torch.equal(
                captured[0],
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            )
        )


if __name__ == "__main__":
    unittest.main()
