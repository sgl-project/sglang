"""CPU regressions for A5 MXFP8 DeepEP low-latency dispatch wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNPUMXFP8DeepEPDispatch(CustomTestCase):
    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "mxfp8",
        }
        common = dict(
            quant_config=quant_config,
        )
        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
        ):
            normal = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.NORMAL)
            )
            low_latency = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.LOW_LATENCY)
            )

        self.assertEqual(normal, DispatcherOutputDtype.BF16)
        self.assertEqual(low_latency, DispatcherOutputDtype.MXFP8)


if __name__ == "__main__":
    unittest.main()
