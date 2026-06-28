"""Tests for sglang.srt.kv_canary.state: ViolationLog and CanaryDeviceState allocation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.jit_kernel.kv_canary.consts import VIOLATION_FIELDS, RealKvHashMode
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.state import CanaryDeviceState, ViolationLog
from sglang.test.test_utils import CustomTestCase

_DEVICE = torch.device("cpu")


def _make_config(
    *,
    ring_capacity=64,
    enable_write_input_assert=False,
    enable_verify_token_assert=False,
):
    return CanaryConfig(
        mode=CanaryMode.NONE,
        ring_capacity=ring_capacity,
        sweep_interval=0,
        real_kv_hash_mode=RealKvHashMode.PARTIAL,
        enable_write_input_assert=enable_write_input_assert,
        enable_verify_token_assert=enable_verify_token_assert,
        stats_print_every_n_steps=100,
    )


class TestViolationLogAllocate(CustomTestCase):
    def test_violation_ring_shape(self):
        vl = ViolationLog.allocate(ring_capacity=64, device=_DEVICE)
        self.assertEqual(vl.violation_ring.shape, torch.Size([64, VIOLATION_FIELDS]))

    def test_violation_ring_uses_violation_fields_const(self):
        # VIOLATION_FIELDS == 8 per the import
        self.assertEqual(VIOLATION_FIELDS, 8)
        vl = ViolationLog.allocate(ring_capacity=1, device=_DEVICE)
        self.assertEqual(vl.violation_ring.shape[1], VIOLATION_FIELDS)

    def test_violation_ring_dtype(self):
        vl = ViolationLog.allocate(ring_capacity=16, device=_DEVICE)
        self.assertEqual(vl.violation_ring.dtype, torch.int64)

    def test_violation_ring_initialised_to_zero(self):
        vl = ViolationLog.allocate(ring_capacity=4, device=_DEVICE)
        self.assertTrue(vl.violation_ring.eq(0).all())

    def test_violation_write_index_shape(self):
        vl = ViolationLog.allocate(ring_capacity=32, device=_DEVICE)
        self.assertEqual(vl.violation_write_index.shape, torch.Size([1]))

    def test_violation_write_index_dtype(self):
        vl = ViolationLog.allocate(ring_capacity=32, device=_DEVICE)
        self.assertEqual(vl.violation_write_index.dtype, torch.int32)

    def test_violation_write_index_initialised_to_zero(self):
        vl = ViolationLog.allocate(ring_capacity=32, device=_DEVICE)
        self.assertEqual(int(vl.violation_write_index[0]), 0)

    def test_ring_capacity_one(self):
        vl = ViolationLog.allocate(ring_capacity=1, device=_DEVICE)
        self.assertEqual(vl.violation_ring.shape[0], 1)

    def test_zero_ring_capacity_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ViolationLog.allocate(ring_capacity=0, device=_DEVICE)
        self.assertIn("ring_capacity", str(ctx.exception))

    def test_negative_ring_capacity_raises(self):
        with self.assertRaises(ValueError):
            ViolationLog.allocate(ring_capacity=-5, device=_DEVICE)

    def test_frozen_dataclass_rejects_mutation(self):
        vl = ViolationLog.allocate(ring_capacity=8, device=_DEVICE)
        with self.assertRaises((AttributeError, TypeError)):
            vl.violation_ring = torch.zeros(1)


class TestCanaryDeviceStateAllocate(CustomTestCase):
    def test_kernel_run_counters_shape(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=4)
        self.assertEqual(state.kernel_run_counters.shape, torch.Size([4]))

    def test_kernel_run_counters_dtype(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=4)
        self.assertEqual(state.kernel_run_counters.dtype, torch.int64)

    def test_kernel_run_counters_initialised_to_zero(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=3)
        self.assertTrue(state.kernel_run_counters.eq(0).all())

    def test_slot_run_counters_shape_matches_num_tags(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=7)
        self.assertEqual(state.slot_run_counters.shape, torch.Size([7]))

    def test_slot_run_counters_dtype(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=2)
        self.assertEqual(state.slot_run_counters.dtype, torch.int64)

    def test_slot_run_counters_initialised_to_zero(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=2)
        self.assertTrue(state.slot_run_counters.eq(0).all())

    def test_enable_chain_position_assert_initialised_to_one(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=1)
        self.assertEqual(int(state.enable_chain_position_assert[0]), 1)

    def test_enable_chain_position_assert_shape(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=1)
        self.assertEqual(state.enable_chain_position_assert.shape, torch.Size([1]))

    def test_enable_chain_position_assert_dtype(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=1)
        self.assertEqual(state.enable_chain_position_assert.dtype, torch.int32)

    def test_req_to_verify_expected_tokens_none_when_disabled(self):
        cfg = _make_config(enable_verify_token_assert=False)
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=2)
        self.assertIsNone(state.req_to_verify_expected_tokens)

    def test_req_to_verify_expected_tokens_allocated_when_enabled(self):
        cfg = _make_config(enable_verify_token_assert=True)
        state = CanaryDeviceState.allocate(
            config=cfg,
            device=_DEVICE,
            num_tags=2,
            req_to_token_alloc_size=10,
            max_context_len=20,
        )
        self.assertIsNotNone(state.req_to_verify_expected_tokens)
        self.assertEqual(
            state.req_to_verify_expected_tokens.shape, torch.Size([10, 20])
        )

    def test_req_to_verify_expected_tokens_dtype(self):
        cfg = _make_config(enable_verify_token_assert=True)
        state = CanaryDeviceState.allocate(
            config=cfg,
            device=_DEVICE,
            num_tags=2,
            req_to_token_alloc_size=5,
            max_context_len=8,
        )
        self.assertEqual(state.req_to_verify_expected_tokens.dtype, torch.int32)

    def test_missing_alloc_size_raises_when_verify_enabled(self):
        cfg = _make_config(enable_verify_token_assert=True)
        with self.assertRaises(ValueError) as ctx:
            CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=2)
        self.assertIn("req_to_token_alloc_size", str(ctx.exception))

    def test_missing_max_context_len_raises_when_verify_enabled(self):
        cfg = _make_config(enable_verify_token_assert=True)
        with self.assertRaises(ValueError):
            CanaryDeviceState.allocate(
                config=cfg,
                device=_DEVICE,
                num_tags=2,
                req_to_token_alloc_size=10,
            )

    def test_zero_num_tags_raises(self):
        cfg = _make_config()
        with self.assertRaises(ValueError) as ctx:
            CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=0)
        self.assertIn("num_tags", str(ctx.exception))

    def test_negative_num_tags_raises(self):
        cfg = _make_config()
        with self.assertRaises(ValueError):
            CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=-1)

    def test_violation_log_ring_capacity_matches_config(self):
        cfg = _make_config(ring_capacity=128)
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=1)
        self.assertEqual(state.violation_log.violation_ring.shape[0], 128)

    def test_frozen_dataclass_rejects_mutation(self):
        cfg = _make_config()
        state = CanaryDeviceState.allocate(config=cfg, device=_DEVICE, num_tags=1)
        with self.assertRaises((AttributeError, TypeError)):
            state.kernel_run_counters = torch.zeros(1)


if __name__ == "__main__":
    unittest.main(verbosity=3)
