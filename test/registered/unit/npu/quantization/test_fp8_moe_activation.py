"""A5 FP8 must apply GLM's gate/up clamp on both dispatch routes."""

import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

# Initialize quantization before importing the NPU methods (their base classes
# and linear methods otherwise participate in a cold-import cycle).
import sglang.srt.layers.quantization  # noqa: E402, F401
from sglang.srt.hardware_backend.npu.moe.activation import (  # noqa: E402
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluOAI,
    NPUSwigluQuant,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (  # noqa: E402
    NPUFp8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.ascend import (  # noqa: E402
    AscendRunnerCore,
    AscendRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402


class TestFP8MoEActivation(CustomTestCase):
    def setUp(self):
        # Substitute external device kernels; all activation selection and math
        # under test come from the production runner/activation modules.
        modules = {}
        for name in ("swiglu_quant", "swiglu_oai_quant", "swiglu_oai"):
            module = ModuleType(f"sgl_kernel_npu.activation.{name}")
            if name == "swiglu_quant":
                module.swiglu_quant = self._plain_grouped_swiglu
            elif name == "swiglu_oai_quant":
                module.swiglu_oai_quant = lambda *args, **kwargs: None
            else:
                module.swiglu_oai_triton = lambda *args, **kwargs: None
            modules[module.__name__] = module
        self._modules = patch.dict(sys.modules, modules)
        self._modules.start()
        self.addCleanup(self._modules.stop)

    @staticmethod
    def _plain_grouped_swiglu(x, *args, **kwargs):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up, None

    @staticmethod
    def _make_runner(deepep, *, kernel_type=NPUFp8MoEMethod, **kwargs):
        # GMMs are the device boundary: w13 supplies known gate/up values and
        # w2 exposes its activation input as the observable runner output.
        kernel = kernel_type.__new__(kernel_type)
        kernel.apply = lambda quant_info, x, *args, **kwargs: x
        layer = SimpleNamespace(w13_kernel=kernel, w2_kernel=kernel)
        config = MoeRunnerConfig(layer=layer, **kwargs)
        with patch(
            "sglang.srt.layers.moe.moe_runner.ascend.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_deepep=lambda: deepep),
        ):
            return AscendRunnerCore(config)

    def test_glm_clamp_reaches_gmm2_for_both_dispatchers(self):
        # Gate -20 must retain its negative tail; gate 12 is clipped BEFORE
        # sigmoid. Up values hit both bounds. No OAI alpha or (up + 1) applies.
        x = torch.tensor([[-20.0, 12.0, 0.0, -2.0, 20.0, -20.0, 4.0, 2.0]])
        expected = torch.tensor(
            [
                [
                    -200.0 / (1 + math.exp(20)),
                    -100.0 / (1 + math.exp(-10)),
                    0.0,
                    -4.0 / (1 + math.exp(2)),
                ]
            ]
        )
        for deepep in (False, True):
            for group_list_type in (0, 1):
                with self.subTest(deepep=deepep, group_list_type=group_list_type):
                    runner = self._make_runner(deepep, swiglu_limit=10.0)
                    output = runner.run(
                        AscendRunnerInput(
                            hidden_states=x,
                            hidden_states_scale=None,
                            expert_tokens=torch.tensor([1]),
                            group_list_type=group_list_type,
                        ),
                        quant_info=None,
                        running_state={},
                    )
                    torch.testing.assert_close(output.hidden_states, expected)
                    self.assertIsNone(runner.config.gemm1_clamp_limit)

    def test_existing_activation_branches_keep_their_semantics(self):
        for deepep in (False, True):
            for limit in (None, 0.0, -1.0):
                with self.subTest(deepep=deepep, limit=limit):
                    runner = self._make_runner(deepep, swiglu_limit=limit)
                    self.assertIsInstance(
                        runner.activation,
                        NPUSwigluDeepEPKernel if deepep else NPUSwiglu,
                    )
            with self.subTest(deepep=deepep, activation="int8"):
                runner = self._make_runner(
                    deepep, kernel_type=NPUW8A8Int8MoEMethod, swiglu_limit=10.0
                )
                self.assertIsInstance(
                    runner.activation,
                    NPUSwigluDeepEPKernel if deepep else NPUSwigluQuant,
                )
            with self.subTest(deepep=deepep, activation="oai"):
                runner = self._make_runner(
                    deepep,
                    activation="npu_swiglu_oai",
                    swiglu_limit=10.0,
                    gemm1_alpha=1.702,
                    gemm1_clamp_limit=7.0,
                )
                self.assertIsInstance(
                    runner.activation,
                    NPUSwigluDeepEPKernel if deepep else NPUSwigluOAI,
                )
                if deepep:
                    self.assertTrue(runner.activation._use_oai)


if __name__ == "__main__":
    unittest.main()
