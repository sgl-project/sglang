"""Unit tests for DeepSeek-V4 MegaMOE TP-attention scatter gating."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.models import deepseek_v4


class _QuantMethod:
    def __init__(self, *, is_fp4_expert: bool):
        self.is_fp4_expert = is_fp4_expert


class _Experts:
    def __init__(self, *, is_fp4_expert: bool):
        self.quant_method = _QuantMethod(is_fp4_expert=is_fp4_expert)


class _Mlp:
    def __init__(self, *, is_fp4_expert: bool):
        self.experts = _Experts(is_fp4_expert=is_fp4_expert)


class TestDeepseekV4MegaMoeScatterGuard(unittest.TestCase):
    def test_megamoe_fp4_fallback_disables_scatter(self):
        mlp = _Mlp(is_fp4_expert=True)
        hidden_states = torch.empty(8, 16)

        with patch.object(
            deepseek_v4,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.MEGAMOE,
        ), patch(
            "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
            return_value=False,
        ) as mock_should_use:
            self.assertTrue(
                deepseek_v4._is_megamoe_fp4_fallback_mlp(mlp, hidden_states)
            )

        mock_should_use.assert_called_once_with(mlp, hidden_states)

    def test_megamoe_fp4_native_path_keeps_scatter_available(self):
        mlp = _Mlp(is_fp4_expert=True)

        with patch.object(
            deepseek_v4,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.MEGAMOE,
        ), patch(
            "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
            return_value=True,
        ):
            self.assertFalse(
                deepseek_v4._is_megamoe_fp4_fallback_mlp(mlp, torch.empty(4, 16))
            )

    def test_non_megamoe_or_non_fp4_paths_keep_scatter_available(self):
        with patch.object(
            deepseek_v4,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.NONE,
        ):
            self.assertFalse(
                deepseek_v4._is_megamoe_fp4_fallback_mlp(
                    _Mlp(is_fp4_expert=True), torch.empty(4, 16)
                )
            )

        with patch.object(
            deepseek_v4,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.MEGAMOE,
        ):
            self.assertFalse(
                deepseek_v4._is_megamoe_fp4_fallback_mlp(
                    _Mlp(is_fp4_expert=False), torch.empty(4, 16)
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
