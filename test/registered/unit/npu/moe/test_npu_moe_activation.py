"""
Unit tests for sglang.srt.hardware_backend.npu.moe.activation.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

# Mock deps pulled in by sglang/__init__ or the source import chain that are
# unavailable on a CPU-only box, so these tests also run locally (CI uses the
# real modules).
if "triton" not in sys.modules:
    try:
        import triton  # noqa: F401
    except ImportError:
        _triton = type(sys)("triton")
        _triton.jit = MagicMock(return_value=lambda f: f)
        _triton.autotune = lambda *a, **kw: (lambda f: f)
        sys.modules["triton"] = _triton
        sys.modules.setdefault("triton.language", MagicMock())
        sys.modules.setdefault("triton.backends", MagicMock())
for _mod in (
    "IPython",
    "IPython.display",
    "aiohttp",
    "zmq",
    "fcntl",
    "sglang.srt.layers.activation",
):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules.setdefault(_mod, MagicMock())

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.activation",
    "sgl_kernel_npu.activation.swiglu_oai_quant",
    "sgl_kernel_npu.activation.swiglu_quant",
    "sgl_kernel_npu.activation.swiglu_oai",
    "sgl_kernel_npu.activation.situ",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.moe.activation import (
    AllGatherActivationWrapper,
    BaseActivation,
    NPUGeluAndMul,
    NPUSitu,
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluQuantWithScales,
    NPUSwigluStepAndMul,
    get_swiglu_variant,
)


# =============================================================================
# BaseActivation
# =============================================================================
class TestBaseActivation(unittest.TestCase):
    def test_is_abstract(self):
        """BaseActivation cannot be instantiated without _apply_activation."""
        with self.assertRaises(TypeError):
            BaseActivation()

    def test_abstract_method_exists(self):
        self.assertTrue(hasattr(BaseActivation, "_apply_activation"))


# =============================================================================
# NPUSwiglu
# =============================================================================
class TestNPUSwiglu(unittest.TestCase):
    @patch("torch.ops")
    def test_calls_npu_swiglu(self, mock_ops):
        """_apply_activation calls torch.ops.npu.npu_swiglu."""
        hidden = torch.randn(4, 16)
        mock_ops.npu.npu_swiglu.return_value = torch.randn(4, 8)

        act = NPUSwiglu()
        out, scale = act._apply_activation(hidden)

        mock_ops.npu.npu_swiglu.assert_called_once_with(hidden)

    @patch("torch.ops")
    def test_returns_none_scale(self, mock_ops):
        """Second return value is always None."""
        mock_ops.npu.npu_swiglu.return_value = torch.randn(4, 8)

        act = NPUSwiglu()
        _, scale = act._apply_activation(torch.randn(4, 16))

        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_returns_op_output(self, mock_ops):
        """First return value is the op output."""
        expected = torch.randn(4, 8)
        mock_ops.npu.npu_swiglu.return_value = expected

        act = NPUSwiglu()
        out, _ = act._apply_activation(torch.randn(4, 16))

        self.assertIs(out, expected)

    @patch("torch.ops")
    def test_preserves_input_passthrough(self, mock_ops):
        """The exact tensor passed in is the exact tensor given to the op."""
        hidden = torch.randn(2, 8)
        mock_ops.npu.npu_swiglu.return_value = torch.randn(2, 4)

        act = NPUSwiglu()
        act._apply_activation(hidden)

        called_arg = mock_ops.npu.npu_swiglu.call_args.args[0]
        self.assertIs(called_arg, hidden)


# =============================================================================
# NPUSwigluQuant
# =============================================================================
class TestNPUSwigluQuant(unittest.TestCase):
    @patch("torch.ops")
    def test_calls_dequant_swiglu_quant(self, mock_ops):
        """torch.ops.npu.npu_dequant_swiglu_quant is called."""
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuant()
        act._apply_activation(torch.randn(4, 16))

        mock_ops.npu.npu_dequant_swiglu_quant.assert_called_once()

    @patch("torch.ops")
    def test_quant_mode_and_activate_left(self, mock_ops):
        """quant_mode=1 and activate_left=True are always passed."""
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuant()
        act._apply_activation(torch.randn(4, 16))

        _, kwargs = mock_ops.npu.npu_dequant_swiglu_quant.call_args
        self.assertEqual(kwargs["quant_mode"], 1)
        self.assertTrue(kwargs["activate_left"])

    @patch("torch.ops")
    def test_returns_hidden_states_and_scale(self, mock_ops):
        """Returns (hidden_states, swiglu_out_scale) from the op."""
        expected_hs = torch.randn(4, 8)
        expected_scale = torch.tensor(0.5)
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            expected_hs,
            expected_scale,
        )

        act = NPUSwigluQuant()
        out, scale = act._apply_activation(torch.randn(4, 16))

        self.assertIs(out, expected_hs)
        self.assertIs(scale, expected_scale)

    @patch("torch.ops")
    def test_input_passed_positionally(self, mock_ops):
        """hidden_states is passed as the first positional argument."""
        hidden = torch.randn(4, 16)
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuant()
        act._apply_activation(hidden)

        args, _ = mock_ops.npu.npu_dequant_swiglu_quant.call_args
        self.assertIs(args[0], hidden)


# =============================================================================
# NPUSwigluQuantWithScales
# =============================================================================
class TestNPUSwigluQuantWithScales(unittest.TestCase):
    def _make_inputs(self):
        return {
            "hidden_states": torch.randn(4, 16),
            "weight_scale": torch.tensor(0.1),
            "activation_scale": torch.tensor(0.2),
            "group_index": torch.tensor([0, 2, 4]),
            "bias": torch.randn(8),
            "quant_scale": torch.tensor(0.3),
            "quant_offset": torch.tensor(1),
        }

    @patch("torch.ops")
    def test_calls_dequant_swiglu_quant(self, mock_ops):
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )
        act = NPUSwigluQuantWithScales()
        act._apply_activation(**self._make_inputs())
        mock_ops.npu.npu_dequant_swiglu_quant.assert_called_once()

    @patch("torch.ops")
    def test_all_kwargs_forwarded(self, mock_ops):
        """Every parameter is forwarded as a keyword argument.

        ``hidden_states`` is forwarded under the key ``x``; all other
        parameters keep their original names.
        """
        inputs = self._make_inputs()
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuantWithScales()
        act._apply_activation(**inputs)

        _, kwargs = mock_ops.npu.npu_dequant_swiglu_quant.call_args
        self.assertIs(kwargs["x"], inputs["hidden_states"])
        for key in (
            "weight_scale",
            "activation_scale",
            "group_index",
            "bias",
            "quant_scale",
            "quant_offset",
        ):
            self.assertIn(key, kwargs)
            self.assertIs(kwargs[key], inputs[key])

    @patch("torch.ops")
    def test_fixed_kwargs(self, mock_ops):
        """activate_left=True and quant_mode=1 are always set."""
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuantWithScales()
        act._apply_activation(**self._make_inputs())

        _, kwargs = mock_ops.npu.npu_dequant_swiglu_quant.call_args
        self.assertTrue(kwargs["activate_left"])
        self.assertEqual(kwargs["quant_mode"], 1)

    @patch("torch.ops")
    def test_optional_args_default_none(self, mock_ops):
        """bias, quant_scale, quant_offset default to None when omitted."""
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            torch.randn(4, 8),
            torch.tensor(0.5),
        )

        act = NPUSwigluQuantWithScales()
        act._apply_activation(
            hidden_states=torch.randn(4, 16),
            weight_scale=torch.tensor(0.1),
            activation_scale=torch.tensor(0.2),
            group_index=torch.tensor([0, 1]),
        )

        _, kwargs = mock_ops.npu.npu_dequant_swiglu_quant.call_args
        self.assertIsNone(kwargs["bias"])
        self.assertIsNone(kwargs["quant_scale"])
        self.assertIsNone(kwargs["quant_offset"])

    @patch("torch.ops")
    def test_returns_hidden_states_and_scale(self, mock_ops):
        expected_hs = torch.randn(4, 8)
        expected_scale = torch.tensor(0.5)
        mock_ops.npu.npu_dequant_swiglu_quant.return_value = (
            expected_hs,
            expected_scale,
        )

        act = NPUSwigluQuantWithScales()
        out, scale = act._apply_activation(**self._make_inputs())

        self.assertIs(out, expected_hs)
        self.assertIs(scale, expected_scale)


# =============================================================================
# NPUSwigluDeepEPKernel
# =============================================================================
class TestNPUSwigluDeepEPKernelInit(unittest.TestCase):
    def test_defaults_use_quant_no_oai(self):
        act = NPUSwigluDeepEPKernel()
        self.assertTrue(act.need_quant)
        self.assertIsNone(act.alpha)
        self.assertIsNone(act.limit)
        self.assertFalse(act._use_oai)

    def test_need_quant_false(self):
        act = NPUSwigluDeepEPKernel(need_quant=False)
        self.assertFalse(act.need_quant)

    def test_oai_when_both_alpha_and_limit(self):
        act = NPUSwigluDeepEPKernel(alpha=0.5, limit=7.0)
        self.assertTrue(act._use_oai)
        self.assertEqual(act.alpha, 0.5)
        self.assertEqual(act.limit, 7.0)

    def test_no_oai_when_only_alpha(self):
        act = NPUSwigluDeepEPKernel(alpha=0.5)
        self.assertFalse(act._use_oai)

    def test_no_oai_when_only_limit(self):
        act = NPUSwigluDeepEPKernel(limit=7.0)
        self.assertFalse(act._use_oai)

    def test_no_oai_when_both_none(self):
        act = NPUSwigluDeepEPKernel(alpha=None, limit=None)
        self.assertFalse(act._use_oai)


class TestNPUSwigluDeepEPKernelApply(unittest.TestCase):
    def _make_kernel(self, need_quant=True, alpha=None, limit=None):
        act = NPUSwigluDeepEPKernel(need_quant=need_quant, alpha=alpha, limit=limit)
        act._kernel = MagicMock(return_value=(torch.randn(4, 8), torch.tensor(0.5)))
        return act

    def test_non_oai_call_args(self):
        """Non-OAI path calls kernel(hidden, group_list, group_list_type, need_quant=...)."""
        act = self._make_kernel(need_quant=True)
        hidden = torch.randn(4, 16)
        group_list = torch.tensor([0, 2, 4])
        group_list_type = 0

        act._apply_activation(hidden, group_list, group_list_type)

        act._kernel.assert_called_once_with(
            hidden, group_list, group_list_type, need_quant=True
        )

    def test_oai_call_args(self):
        """OAI path calls kernel(hidden, alpha, limit, need_quant=..., group_list=..., group_list_type=...)."""
        act = self._make_kernel(need_quant=True, alpha=0.5, limit=7.0)
        hidden = torch.randn(4, 16)
        group_list = torch.tensor([0, 2, 4])
        group_list_type = 1

        act._apply_activation(hidden, group_list, group_list_type)

        act._kernel.assert_called_once_with(
            hidden,
            0.5,
            7.0,
            need_quant=True,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def test_return_with_quant(self):
        """need_quant=True returns (hidden_states, per_token_scale)."""
        act = self._make_kernel(need_quant=True)
        out, scale = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIsNotNone(scale)

    def test_return_without_quant(self):
        """need_quant=False returns (hidden_states, None)."""
        act = self._make_kernel(need_quant=False)
        out, scale = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIsNone(scale)

    def test_oai_return_with_quant(self):
        act = self._make_kernel(need_quant=True, alpha=0.5, limit=7.0)
        out, scale = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIsNotNone(scale)

    def test_oai_return_without_quant(self):
        act = self._make_kernel(need_quant=False, alpha=1.0, limit=5.0)
        out, scale = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIsNone(scale)

    def test_kernel_output_returned(self):
        """The first element of the kernel tuple is returned as out."""
        expected_out = torch.randn(4, 8)
        act = NPUSwigluDeepEPKernel(need_quant=True)
        act._kernel = MagicMock(return_value=(expected_out, torch.tensor(0.5)))
        out, _ = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIs(out, expected_out)


# =============================================================================
# NPUSitu
# =============================================================================
class TestNPUSituInit(unittest.TestCase):
    def test_defaults(self):
        act = NPUSitu(need_quant=True)
        self.assertTrue(act.need_quant)
        self.assertEqual(act.beta, 4.0)
        self.assertEqual(act.linear_beta, 25.0)
        self.assertIsInstance(act.beta, float)
        self.assertIsInstance(act.linear_beta, float)

    def test_need_quant_false(self):
        act = NPUSitu(need_quant=False)
        self.assertFalse(act.need_quant)

    def test_custom_beta(self):
        act = NPUSitu(need_quant=True, beta=2.0, linear_beta=10.0)
        self.assertEqual(act.beta, 2.0)
        self.assertEqual(act.linear_beta, 10.0)

    def test_int_beta_converted_to_float(self):
        act = NPUSitu(need_quant=True, beta=4, linear_beta=25)
        self.assertIsInstance(act.beta, float)
        self.assertIsInstance(act.linear_beta, float)

    def test_linear_beta_none(self):
        act = NPUSitu(need_quant=True, linear_beta=None)
        self.assertIsNone(act.linear_beta)

    def test_situ_attribute_set(self):
        act = NPUSitu(need_quant=True)
        self.assertIsNotNone(act.situ)


class TestNPUSituApply(unittest.TestCase):
    def _make_act(self, need_quant=True, beta=4.0, linear_beta=25.0):
        act = NPUSitu(need_quant=need_quant, beta=beta, linear_beta=linear_beta)
        act.situ = MagicMock(return_value=(torch.randn(4, 8), torch.tensor(0.5)))
        return act

    def test_calls_situ(self):
        act = self._make_act()
        hidden = torch.randn(4, 16)
        group_list = torch.tensor([0, 2, 4])
        group_list_type = 0

        act._apply_activation(hidden, group_list, group_list_type)

        act.situ.assert_called_once()

    def test_positional_args(self):
        act = self._make_act()
        hidden = torch.randn(4, 16)
        group_list = torch.tensor([0, 2, 4])
        group_list_type = 1

        act._apply_activation(hidden, group_list, group_list_type)

        args, _ = act.situ.call_args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], group_list)
        self.assertEqual(args[2], group_list_type)

    def test_kwargs_forwarded(self):
        act = self._make_act(need_quant=True, beta=2.0, linear_beta=10.0)
        act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)

        _, kwargs = act.situ.call_args
        self.assertEqual(kwargs["need_quant"], True)
        self.assertEqual(kwargs["beta"], 2.0)
        self.assertEqual(kwargs["linear_beta"], 10.0)

    def test_linear_beta_none_forwarded(self):
        act = self._make_act(need_quant=True, linear_beta=None)
        act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)

        _, kwargs = act.situ.call_args
        self.assertIsNone(kwargs["linear_beta"])

    def test_returns_situ_output(self):
        expected = (torch.randn(4, 8), torch.tensor(0.5))
        act = NPUSitu(need_quant=True)
        act.situ = MagicMock(return_value=expected)
        result = act._apply_activation(torch.randn(4, 16), torch.tensor([0, 2]), 0)
        self.assertIs(result, expected)


# =============================================================================
# NPUGeluAndMul
# =============================================================================
class TestNPUGeluAndMul(unittest.TestCase):
    @patch("sglang.srt.hardware_backend.npu.moe.activation.GeluAndMul")
    def test_returns_tuple_with_none_scale(self, mock_gelu_cls):
        mock_instance = MagicMock(return_value=torch.randn(4, 8))
        mock_gelu_cls.return_value = mock_instance

        act = NPUGeluAndMul()
        out, scale = act._apply_activation(torch.randn(4, 16))

        self.assertIsNone(scale)

    @patch("sglang.srt.hardware_backend.npu.moe.activation.GeluAndMul")
    def test_calls_gelu(self, mock_gelu_cls):
        mock_instance = MagicMock(return_value=torch.randn(4, 8))
        mock_gelu_cls.return_value = mock_instance

        act = NPUGeluAndMul()
        hidden = torch.randn(4, 16)
        act._apply_activation(hidden)

        mock_instance.assert_called_once_with(hidden)

    @patch("sglang.srt.hardware_backend.npu.moe.activation.GeluAndMul")
    def test_returns_gelu_output(self, mock_gelu_cls):
        expected = torch.randn(4, 8)
        mock_instance = MagicMock(return_value=expected)
        mock_gelu_cls.return_value = mock_instance

        act = NPUGeluAndMul()
        out, _ = act._apply_activation(torch.randn(4, 16))

        self.assertIs(out, expected)

    @patch("sglang.srt.hardware_backend.npu.moe.activation.GeluAndMul")
    def test_gelu_instance_created_once(self, mock_gelu_cls):
        mock_instance = MagicMock(return_value=torch.randn(4, 8))
        mock_gelu_cls.return_value = mock_instance

        act = NPUGeluAndMul()
        mock_gelu_cls.assert_called_once()


# =============================================================================
# NPUSwigluOAI
# =============================================================================
class TestNPUSwigluOAIInit(unittest.TestCase):
    def test_defaults(self):
        act = NPUSwigluOAI()
        self.assertIsNone(act._moe_runner_config)
        self.assertIsNotNone(act._kernel)

    def test_with_config(self):
        config = SimpleNamespace(gemm1_alpha=0.5, gemm1_clamp_limit=7.0)
        act = NPUSwigluOAI(moe_runner_config=config)
        self.assertIs(act._moe_runner_config, config)


class TestNPUSwigluOAIApply(unittest.TestCase):
    def _make_act(self, config=None):
        act = NPUSwigluOAI(moe_runner_config=config)
        act._kernel = MagicMock(return_value=torch.randn(4, 8))
        return act

    def test_default_alpha_and_clamp(self):
        """Without config, alpha=1.0 and clamp=None."""
        act = self._make_act(config=None)
        hidden = torch.randn(4, 16)
        act._apply_activation(hidden)

        args = act._kernel.call_args.args
        self.assertIs(args[0], hidden)
        self.assertEqual(args[1], hidden.shape[-1])
        self.assertEqual(args[2], 1.0)
        self.assertIsNone(args[3])

    def test_alpha_from_config(self):
        config = SimpleNamespace(gemm1_alpha=0.5, gemm1_clamp_limit=7.0)
        act = self._make_act(config=config)
        act._apply_activation(torch.randn(4, 16))

        args = act._kernel.call_args.args
        self.assertEqual(args[2], 0.5)
        self.assertEqual(args[3], 7.0)

    def test_missing_attrs_fallback(self):
        """Config without gemm1_alpha/gemm1_clamp_limit falls back to defaults."""
        config = SimpleNamespace()
        act = self._make_act(config=config)
        act._apply_activation(torch.randn(4, 16))

        args = act._kernel.call_args.args
        self.assertEqual(args[2], 1.0)
        self.assertIsNone(args[3])

    def test_gate_up_dim_from_tensor(self):
        """gate_up dim is derived from hidden_states.shape[-1]."""
        act = self._make_act(config=None)
        hidden = torch.randn(6, 32)
        act._apply_activation(hidden)

        args = act._kernel.call_args.args
        self.assertEqual(args[1], 32)

    def test_returns_output_and_none(self):
        expected = torch.randn(4, 8)
        act = NPUSwigluOAI()
        act._kernel = MagicMock(return_value=expected)
        out, scale = act._apply_activation(torch.randn(4, 16))

        self.assertIs(out, expected)
        self.assertIsNone(scale)

    def test_kernel_called_once(self):
        act = self._make_act(config=None)
        act._apply_activation(torch.randn(4, 16))
        act._kernel.assert_called_once()


# =============================================================================
# NPUSwigluStepAndMul
# =============================================================================
class TestNPUSwigluStepAndMulInit(unittest.TestCase):
    def test_default_clamp_limit_none(self):
        act = NPUSwigluStepAndMul()
        self.assertIsNone(act._clamp_limit)

    def test_custom_clamp_limit(self):
        act = NPUSwigluStepAndMul(clamp_limit=7.0)
        self.assertEqual(act._clamp_limit, 7.0)

    def test_clamp_limit_zero(self):
        act = NPUSwigluStepAndMul(clamp_limit=0.0)
        self.assertEqual(act._clamp_limit, 0.0)


class TestSwiglustepAndMul(unittest.TestCase):
    """Tests for the static method _swiglustep_and_mul (pure torch)."""

    def test_shape(self):
        x = torch.randn(4, 16)
        out = NPUSwigluStepAndMul._swiglustep_and_mul(x)
        self.assertEqual(out.shape, (4, 8))

    def test_default_limit_7(self):
        """Default limit is 7.0."""
        x = torch.randn(4, 16)
        out = NPUSwigluStepAndMul._swiglustep_and_mul(x)

        gate, up = x.chunk(2, dim=-1)
        expected = F.silu(gate).clamp(max=7.0) * up.clamp(min=-7.0, max=7.0)
        self.assertTrue(torch.allclose(out, expected))

    def test_custom_limit(self):
        x = torch.randn(4, 16)
        out = NPUSwigluStepAndMul._swiglustep_and_mul(x, limit=3.0)

        gate, up = x.chunk(2, dim=-1)
        expected = F.silu(gate).clamp(max=3.0) * up.clamp(min=-3.0, max=3.0)
        self.assertTrue(torch.allclose(out, expected))


class TestNPUSwigluStepAndMulApply(unittest.TestCase):
    def test_with_clamp_limit_uses_swiglustep(self):
        """When clamp_limit is set, _swiglustep_and_mul is used."""
        act = NPUSwigluStepAndMul(clamp_limit=7.0)
        hidden = torch.randn(4, 16)
        out, scale = act._apply_activation(hidden)

        gate, up = hidden.chunk(2, dim=-1)
        expected = F.silu(gate).clamp(max=7.0) * up.clamp(min=-7.0, max=7.0)
        self.assertTrue(torch.allclose(out, expected))
        self.assertIsNone(scale)

    def test_with_custom_clamp_limit(self):
        act = NPUSwigluStepAndMul(clamp_limit=3.0)
        hidden = torch.randn(4, 16)
        out, scale = act._apply_activation(hidden)

        gate, up = hidden.chunk(2, dim=-1)
        expected = F.silu(gate).clamp(max=3.0) * up.clamp(min=-3.0, max=3.0)
        self.assertTrue(torch.allclose(out, expected))
        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_without_clamp_calls_npu_swiglu(self, mock_ops):
        """Without clamp_limit, torch.ops.npu.npu_swiglu is called."""
        mock_ops.npu.npu_swiglu.return_value = torch.randn(4, 8)
        act = NPUSwigluStepAndMul(clamp_limit=None)
        hidden = torch.randn(4, 16)

        act._apply_activation(hidden)

        mock_ops.npu.npu_swiglu.assert_called_once_with(hidden)

    @patch("torch.ops")
    def test_without_clamp_returns_none_scale(self, mock_ops):
        mock_ops.npu.npu_swiglu.return_value = torch.randn(4, 8)
        act = NPUSwigluStepAndMul(clamp_limit=None)

        _, scale = act._apply_activation(torch.randn(4, 16))
        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_without_clamp_returns_op_output(self, mock_ops):
        expected = torch.randn(4, 8)
        mock_ops.npu.npu_swiglu.return_value = expected
        act = NPUSwigluStepAndMul(clamp_limit=None)

        out, _ = act._apply_activation(torch.randn(4, 16))
        self.assertIs(out, expected)


# =============================================================================
# AllGatherActivationWrapper
# =============================================================================
class TestAllGatherActivationWrapperInit(unittest.TestCase):
    def test_defaults(self):
        inner = MagicMock()
        wrapper = AllGatherActivationWrapper(inner)
        self.assertIs(wrapper.inner, inner)
        self.assertEqual(wrapper.dim, -1)

    def test_custom_dim(self):
        inner = MagicMock()
        wrapper = AllGatherActivationWrapper(inner, dim=0)
        self.assertEqual(wrapper.dim, 0)

    def test_dim_positive(self):
        inner = MagicMock()
        wrapper = AllGatherActivationWrapper(inner, dim=2)
        self.assertEqual(wrapper.dim, 2)


class TestAllGatherActivationWrapperApply(unittest.TestCase):
    _GET_PARALLEL = "sglang.srt.hardware_backend.npu.moe.activation.get_parallel"
    _ALL_GATHER = (
        "sglang.srt.hardware_backend.npu.moe.activation."
        "tensor_model_parallel_all_gather"
    )

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_tp1_no_gather(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner_out = torch.randn(4, 8)
        inner = MagicMock()
        inner._apply_activation.return_value = (inner_out, None)

        wrapper = AllGatherActivationWrapper(inner)
        out, scale = wrapper._apply_activation(torch.randn(4, 16))

        mock_all_gather.assert_not_called()
        self.assertIs(out, inner_out)
        self.assertIsNone(scale)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_tp2_gathers(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=2)
        gathered = torch.randn(8, 8)
        mock_all_gather.return_value = gathered
        inner_out = torch.randn(4, 8)
        inner_scale = torch.tensor(0.5)
        inner = MagicMock()
        inner._apply_activation.return_value = (inner_out, inner_scale)

        wrapper = AllGatherActivationWrapper(inner, dim=-1)
        out, scale = wrapper._apply_activation(torch.randn(4, 16))

        mock_all_gather.assert_called_once_with(inner_out, dim=-1)
        self.assertIs(out, gathered)
        self.assertIs(scale, inner_scale)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_gather_with_custom_dim(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=4)
        mock_all_gather.return_value = torch.randn(16, 8)
        inner = MagicMock()
        inner._apply_activation.return_value = (
            torch.randn(4, 8),
            None,
        )

        wrapper = AllGatherActivationWrapper(inner, dim=0)
        wrapper._apply_activation(torch.randn(4, 16))

        _, kwargs = mock_all_gather.call_args
        self.assertEqual(kwargs["dim"], 0)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_forwards_args_to_inner(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner = MagicMock()
        inner._apply_activation.return_value = (torch.randn(4, 8), None)

        wrapper = AllGatherActivationWrapper(inner)
        hidden = torch.randn(4, 16)
        wrapper._apply_activation(hidden, extra="kwarg")

        inner._apply_activation.assert_called_once_with(hidden, extra="kwarg")

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_scale_passthrough_tp1(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner_scale = torch.tensor(0.5)
        inner = MagicMock()
        inner._apply_activation.return_value = (
            torch.randn(4, 8),
            inner_scale,
        )

        wrapper = AllGatherActivationWrapper(inner)
        _, scale = wrapper._apply_activation(torch.randn(4, 16))
        self.assertIs(scale, inner_scale)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_scale_passthrough_tp2(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=2)
        mock_all_gather.return_value = torch.randn(8, 8)
        inner_scale = torch.tensor(0.5)
        inner = MagicMock()
        inner._apply_activation.return_value = (
            torch.randn(4, 8),
            inner_scale,
        )

        wrapper = AllGatherActivationWrapper(inner)
        _, scale = wrapper._apply_activation(torch.randn(4, 16))
        self.assertIs(scale, inner_scale)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_tp1_boundary(self, mock_get_parallel, mock_all_gather):
        """tp_size=1 is the boundary; no gather."""
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner = MagicMock()
        inner._apply_activation.return_value = (
            torch.randn(4, 8),
            None,
        )

        wrapper = AllGatherActivationWrapper(inner)
        wrapper._apply_activation(torch.randn(4, 16))
        mock_all_gather.assert_not_called()


# =============================================================================
# get_swiglu_variant factory
# =============================================================================
class TestGetSwigluVariant(unittest.TestCase):
    def test_standard(self):
        act = get_swiglu_variant("standard")
        self.assertIsInstance(act, NPUSwiglu)

    def test_dequant_swiglu_quant(self):
        act = get_swiglu_variant("dequant_swiglu_quant")
        self.assertIsInstance(act, NPUSwigluQuant)

    def test_dequant_swiglu_quant_with_scales(self):
        act = get_swiglu_variant("dequant_swiglu_quant_with_scales")
        self.assertIsInstance(act, NPUSwigluQuantWithScales)

    def test_swiglu_quant_deepep_kernel(self):
        act = get_swiglu_variant("swiglu_quant_deepep_kernel")
        self.assertIsInstance(act, NPUSwigluDeepEPKernel)

    def test_gelu_and_mul(self):
        act = get_swiglu_variant("gelu_and_mul")
        self.assertIsInstance(act, NPUGeluAndMul)

    def test_swiglu_oai(self):
        act = get_swiglu_variant("swiglu_oai")
        self.assertIsInstance(act, NPUSwigluOAI)

    def test_swiglustep_and_mul_default(self):
        act = get_swiglu_variant("swiglustep_and_mul")
        self.assertIsInstance(act, NPUSwigluStepAndMul)
        self.assertIsNone(act._clamp_limit)

    def test_swiglustep_and_mul_with_clamp(self):
        act = get_swiglu_variant("swiglustep_and_mul", clamp_limit=7.0)
        self.assertIsInstance(act, NPUSwigluStepAndMul)
        self.assertEqual(act._clamp_limit, 7.0)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_swiglu_variant("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_unknown_variant_message_format(self):
        with self.assertRaises(ValueError) as ctx:
            get_swiglu_variant("foobar")
        self.assertIn("Unknown SwiGLU variant", str(ctx.exception))

    def test_all_variants_return_base_activation(self):
        """Every factory result is a BaseActivation subclass instance."""
        for method in (
            "standard",
            "dequant_swiglu_quant",
            "dequant_swiglu_quant_with_scales",
            "swiglu_quant_deepep_kernel",
            "gelu_and_mul",
            "swiglu_oai",
            "swiglustep_and_mul",
        ):
            with self.subTest(method=method):
                act = get_swiglu_variant(method)
                self.assertIsInstance(act, BaseActivation)

    def test_swiglustep_and_mul_extra_kwargs_ignored(self):
        """Extra kwargs other than clamp_limit are silently ignored."""
        act = get_swiglu_variant("swiglustep_and_mul", clamp_limit=5.0, extra="ignored")
        self.assertEqual(act._clamp_limit, 5.0)


if __name__ == "__main__":
    unittest.main()
