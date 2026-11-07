"""CPU contracts for A2 Normal BF16 and quantized LowLatency dispatch."""

import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DispatcherOutputDtype as Dtype
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestNormalBf16Policy(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            patch.dict(os.environ, {"DEEP_NORMAL_MODE_USE_INT8_QUANT": "0"})
        )

    @staticmethod
    def dispatcher(mode, requested):
        cls = (
            deepep._DeepEPDispatcherImplNormal
            if mode == "normal"
            else deepep._DeepEPDispatcherImplLowLatency
        )
        obj = cls.__new__(cls)
        obj.quant_config = {"dispatcher_output_dtype": requested.value}
        obj.set_deepep_dispatcher_dtype()
        return obj

    def test_dtype_matrix(self):
        for enabled in (False, True):
            for npu, arch35 in ((False, False), (True, False), (True, True)):
                for mode in ("normal", "low_latency"):
                    for dtype in (Dtype.BF16, Dtype.INT8, Dtype.FP8, Dtype.MXFP8):
                        with (
                            self.subTest(
                                enabled=enabled,
                                npu=npu,
                                arch35=arch35,
                                mode=mode,
                                dtype=dtype,
                            ),
                            envs.SGLANG_NPU_DEEPEP_NORMAL_BF16.override(enabled),
                            patch.object(deepep, "_is_npu", npu),
                            patch.object(
                                deepep, "get_deepep_output_dtype", return_value=dtype
                            ),
                            patch(
                                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                                return_value=arch35,
                            ),
                        ):
                            obj = self.dispatcher(mode, dtype)
                            expected = (
                                Dtype.FP8 if not npu and dtype == Dtype.INT8 else dtype
                            )
                            if (
                                enabled
                                and npu
                                and not arch35
                                and mode == "normal"
                                and dtype in (Dtype.INT8, Dtype.FP8)
                            ):
                                expected = Dtype.BF16
                            self.assertEqual(obj.deepep_output_dtype, expected)
                            self.assertEqual(
                                obj.use_fp8, expected in (Dtype.INT8, Dtype.FP8)
                            )
                            self.assertEqual(obj.use_mxfp8, expected == Dtype.MXFP8)

    def test_rejects_legacy_int8_before_normal_dispatch(self):
        with (
            envs.SGLANG_NPU_DEEPEP_NORMAL_BF16.override(True),
            patch.object(deepep, "_is_npu", True),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch.dict(os.environ, {"DEEP_NORMAL_MODE_USE_INT8_QUANT": "1"}),
        ):
            for dtype in (Dtype.BF16, Dtype.INT8, Dtype.FP8):
                with (
                    self.subTest(dtype=dtype),
                    patch.object(deepep, "get_deepep_output_dtype", return_value=dtype),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "DEEP_NORMAL_MODE_USE_INT8_QUANT=0"
                    ):
                        self.dispatcher("normal", dtype)
                    self.assertEqual(
                        self.dispatcher("low_latency", dtype).deepep_output_dtype,
                        dtype,
                    )

    def test_selected_cli_dtype_and_quant_config_refresh(self):
        with (
            get_context().override_server_args(deepep_dispatcher_output_dtype="int8"),
            envs.SGLANG_NPU_DEEPEP_NORMAL_BF16.override(True),
            patch.object(deepep, "_is_npu", True),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
        ):
            normal = self.dispatcher("normal", Dtype.INT8)
            low_latency = self.dispatcher("low_latency", Dtype.INT8)
            normal.set_quant_config({"dispatcher_output_dtype": "int8"})
            self.assertEqual(normal.deepep_output_dtype, Dtype.BF16)
            self.assertEqual(low_latency.deepep_output_dtype, Dtype.INT8)

            class FlagBuffer:
                def dispatch(self, use_fp8=False, use_mxfp4=False, use_mxfp8=False):
                    pass

            class QuantModeBuffer:
                def dispatch(self, quant_mode="int8"):
                    pass

            self.assertEqual(
                normal._get_quantization_kwargs(FlagBuffer()),
                {
                    "use_fp8": False,
                    "use_mxfp4": False,
                    "use_mxfp8": False,
                },
            )
            self.assertEqual(
                normal._get_quantization_kwargs(QuantModeBuffer()),
                {"quant_mode": "bf16"},
            )


if __name__ == "__main__":
    unittest.main()
