from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _fixtures import CPU_DEVICE  # noqa: E402

from sglang.jit_kernel.kv_canary.verify import VIOLATION_FIELDS  # noqa: E402
from sglang.srt.kv_canary.state import ViolationLog  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitViolationState(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE

    def test_violation_log_allocate_zeroed(self):
        log = ViolationLog.allocate(ring_capacity=8, device=self.device)
        self.assertEqual(log.violation_ring.shape, (8, VIOLATION_FIELDS))
        self.assertEqual(log.violation_ring.dtype, torch.int64)
        self.assertEqual(int(log.violation_ring.abs().sum()), 0)
        self.assertEqual(log.violation_write_index.shape, (1,))
        self.assertEqual(log.violation_write_index.dtype, torch.int32)
        self.assertEqual(int(log.violation_write_index.item()), 0)

    def test_clear_resets_all(self):
        log = ViolationLog.allocate(ring_capacity=4, device=self.device)
        log.violation_ring[0, 0] = 7
        log.violation_ring[2, 3] = 11
        log.violation_write_index[0] = 5
        log.clear()
        self.assertEqual(int(log.violation_ring.abs().sum()), 0)
        self.assertEqual(int(log.violation_write_index.item()), 0)

    @unittest.expectedFailure
    def test_raise_message_includes_idx_expected_actual(self):
        from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
        from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
        from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner

        config = CanaryConfig(mode=CanaryMode.RAISE)
        k_head = torch.zeros(4, 32, dtype=torch.uint8, device=self.device)
        k_tail = torch.zeros(4, 32, dtype=torch.uint8, device=self.device)
        group = CanaryBufferGroup(
            kind=PoolKind.FULL,
            k_head=k_head,
            k_tail=k_tail,
            v_head=None,
            v_tail=None,
            real_kv_sources_k=(),
            real_kv_sources_v=(),
            swa_index_lut=None,
        )
        runner = CanaryRunner(
            config=config,
            buffer_group=group,
            device=self.device,
            per_forward_verify_capacity=1,
            per_forward_write_req_capacity=1,
            running_sweep_verify_capacity=1,
            radix_sweep_verify_capacity=1,
            radix_sweep_extras_capacity=1,
        )
        runner.violation_log.violation_ring[0] = torch.tensor(
            [1, 42, 5, 100, 200, 0xDEAD, 0xBEEF, 1],
            dtype=torch.int64,
            device=self.device,
        )
        runner.violation_log.violation_write_index[0] = 1

        with self.assertRaises(RuntimeError) as ctx:
            runner._raise_with_first_violation()

        payload = ctx.exception.args[0]
        self.assertIsInstance(payload, dict)
        for key in (
            "slot_idx",
            "position",
            "expected",
            "actual",
            "fail_reason",
            "kernel_kind",
        ):
            self.assertIn(key, payload)
            self.assertIsNotNone(payload[key])


if __name__ == "__main__":
    unittest.main()
