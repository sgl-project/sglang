from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary import write as write_module
from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    VerifyOrWriteContext,
)
from sglang.jit_kernel.kv_canary.write import WritePlan, launch_canary_write_kernel
from sglang.srt.kv_canary.state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-large")


class _RecordingWriteModule:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def canary_write_step_cuda(self, *args: object) -> None:
        self.calls.append(args)


class TestSelfUnitWrite(CustomTestCase):
    def setUp(self) -> None:
        self.device = DEFAULT_DEVICE

    def test_disabled_assert_inputs_passes_none_to_cuda(self) -> None:
        """Verify disabled input assertions pass None instead of dummy tensors."""
        canary_buf = torch.zeros(
            4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=self.device
        )
        plan = WritePlan.allocate(write_req_capacity=1, device=self.device)
        input_ids = torch.zeros(1, dtype=torch.int64, device=self.device)
        positions = torch.zeros(1, dtype=torch.int64, device=self.device)
        out_cache_loc = torch.zeros(1, dtype=torch.int64, device=self.device)
        violation_log = ViolationLog.allocate(ring_capacity=2, device=self.device)
        slot_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        context = VerifyOrWriteContext(
            canary_buf=canary_buf,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )
        module = _RecordingWriteModule()

        with patch.object(write_module, "_jit_canary_write_module", lambda: module):
            launch_canary_write_kernel(
                context=context,
                plan=plan,
                input_ids=input_ids,
                positions=positions,
                out_cache_loc=out_cache_loc,
                enable_assert_inputs=False,
                expected_input_tokens=None,
                expected_input_positions=None,
            )

        self.assertEqual(len(module.calls), 1)
        call = module.calls[0]
        self.assertEqual(call[8], 0)
        self.assertIsNone(call[9])
        self.assertIsNone(call[10])


if __name__ == "__main__":
    unittest.main()
