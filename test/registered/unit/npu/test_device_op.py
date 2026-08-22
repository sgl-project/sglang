"""
Unit tests for sglang.srt.hardware_backend.npu.device_op.

Covers:
- Ascend device generation detection (mocked, no real device needed)
- BaseDeviceOperator / A5DeviceOperator MXFP scale layout contract
- NPUMoEInitRouting_v2 routing regression: the MXFP scale runtime layout is
  produced by the DeviceOperator instead of feature-local branching.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.device_op import (
    A5DeviceOperator,
    AscendDeviceGeneration,
    BaseDeviceOperator,
    get_device_operator,
    get_npu_device_generation,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
)


class _CacheClearingTestCase(unittest.TestCase):
    """Clear the lru_cache'd detection/factory around every test."""

    def setUp(self):
        get_npu_device_generation.cache_clear()
        get_device_operator.cache_clear()

    def tearDown(self):
        get_npu_device_generation.cache_clear()
        get_device_operator.cache_clear()


def _fake_torch_npu(device_name, current_device=0):
    return SimpleNamespace(
        current_device=lambda: current_device,
        get_device_name=lambda device_id: device_name,
    )


class TestDeviceGenerationDetection(_CacheClearingTestCase):
    def test_non_npu_environment_returns_legacy(self):
        with patch("sglang.srt.utils.is_npu", return_value=False):
            self.assertEqual(get_npu_device_generation(), AscendDeviceGeneration.LEGACY)
            self.assertIsInstance(get_device_operator(), BaseDeviceOperator)

    def test_ascend950_detected_as_a5(self):
        probed = {}

        def fake_get_device_name(device_id):
            probed["device_id"] = device_id
            return "Ascend950A101"

        fake_npu = SimpleNamespace(
            current_device=lambda: 3,
            get_device_name=fake_get_device_name,
        )
        with patch("sglang.srt.utils.is_npu", return_value=True), patch.object(
            torch, "npu", fake_npu, create=True
        ):
            self.assertEqual(get_npu_device_generation(), AscendDeviceGeneration.A5)
            self.assertIsInstance(get_device_operator(), A5DeviceOperator)
        # The current device is probed, never a hardcoded device 0.
        self.assertEqual(probed["device_id"], 3)

    def test_ascend910_detected_as_legacy(self):
        fake_npu = _fake_torch_npu("Ascend910B4")
        with patch("sglang.srt.utils.is_npu", return_value=True), patch.object(
            torch, "npu", fake_npu, create=True
        ):
            self.assertEqual(get_npu_device_generation(), AscendDeviceGeneration.LEGACY)
            self.assertIsInstance(get_device_operator(), BaseDeviceOperator)

    def test_explicit_device_id_is_forwarded(self):
        probed = {}

        def fake_get_device_name(device_id):
            probed["device_id"] = device_id
            return "Ascend950A101"

        fake_npu = SimpleNamespace(
            current_device=lambda: 3,
            get_device_name=fake_get_device_name,
        )
        with patch("sglang.srt.utils.is_npu", return_value=True), patch.object(
            torch, "npu", fake_npu, create=True
        ):
            self.assertEqual(
                get_npu_device_generation(device_id=5),
                AscendDeviceGeneration.A5,
            )
        self.assertEqual(probed["device_id"], 5)


class TestNormalizeMxfpScaleLayout(unittest.TestCase):
    def test_base_returns_identity(self):
        scale = torch.empty(4, 16)
        self.assertIs(BaseDeviceOperator.normalize_mxfp_scale_layout(scale), scale)

    def test_base_passes_none(self):
        self.assertIsNone(BaseDeviceOperator.normalize_mxfp_scale_layout(None))

    def test_a5_pair_splits_2d_scale(self):
        scale = torch.arange(64, dtype=torch.float32).reshape(4, 16)
        out = A5DeviceOperator.normalize_mxfp_scale_layout(scale)
        self.assertEqual(out.shape, (4, 8, 2))
        self.assertTrue(torch.equal(out.reshape(4, 16), scale))

    def test_a5_passes_through_3d_scale(self):
        scale = torch.empty(4, 8, 2)
        self.assertIs(A5DeviceOperator.normalize_mxfp_scale_layout(scale), scale)

    def test_a5_passes_none(self):
        self.assertIsNone(A5DeviceOperator.normalize_mxfp_scale_layout(None))


class TestInitRoutingMxfpScale(unittest.TestCase):
    """Routing regression: MXFP scale layout comes from the DeviceOperator."""

    NUM_TOKENS = 4
    TOP_K = 2
    NUM_EXPERTS = 8
    SCALE_COLS = 16

    def _flat_scale(self):
        return torch.arange(
            self.NUM_TOKENS * self.SCALE_COLS, dtype=torch.float32
        ).reshape(self.NUM_TOKENS, self.SCALE_COLS)

    def _run_routing(self, quant_mode):
        hidden_states = torch.randn(self.NUM_TOKENS, 16)
        topk_ids = torch.zeros(self.NUM_TOKENS, self.TOP_K, dtype=torch.int32)
        flat_scale = self._flat_scale()

        def fake_init_routing_v2(hidden, topk, **kwargs):
            return (
                hidden,
                torch.zeros(self.NUM_TOKENS * self.TOP_K, dtype=torch.int32),
                torch.zeros(self.NUM_EXPERTS, dtype=torch.int32),
                flat_scale,
            )

        routing = NPUMoEInitRouting_v2(quant_mode=quant_mode)
        with patch.object(
            torch.ops.npu,
            "npu_moe_init_routing_v2",
            fake_init_routing_v2,
            create=True,
        ):
            return routing._init_routing(
                hidden_states, topk_ids, self.NUM_EXPERTS, self.TOP_K
            )

    def test_a5_operator_keeps_pair_split_contract(self):
        # Before the DeviceOperator refactor, quant_mode==3 always pair-split
        # the flat [N, M] scale; on A5 the runtime contract is unchanged.
        with patch(
            "sglang.srt.hardware_backend.npu.moe.init_routing.get_device_operator",
            return_value=A5DeviceOperator(),
        ):
            _, _, _, scale = self._run_routing(MXFP8_QUANT_MODE)
        self.assertEqual(scale.shape, (self.NUM_TOKENS, self.SCALE_COLS // 2, 2))
        self.assertTrue(
            torch.equal(scale.reshape(self.NUM_TOKENS, -1), self._flat_scale())
        )

    def test_legacy_operator_keeps_native_scale_layout(self):
        # Legacy devices keep the producer's native representation.
        with patch(
            "sglang.srt.hardware_backend.npu.moe.init_routing.get_device_operator",
            return_value=BaseDeviceOperator(),
        ):
            _, _, _, scale = self._run_routing(MXFP8_QUANT_MODE)
        self.assertEqual(scale.shape, (self.NUM_TOKENS, self.SCALE_COLS))

    def test_no_quant_returns_none_scale(self):
        with patch(
            "sglang.srt.hardware_backend.npu.moe.init_routing.get_device_operator",
            return_value=A5DeviceOperator(),
        ):
            _, _, _, scale = self._run_routing(-1)
        self.assertIsNone(scale)


if __name__ == "__main__":
    unittest.main()
