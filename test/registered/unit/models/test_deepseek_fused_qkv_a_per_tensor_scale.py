"""Per-tensor scales of q_a_proj and kv_a_proj_with_mqa in the DeepSeek loader.

fused_qkv_a_proj_with_mqa holds one per-tensor scale, so the loader has to
merge the two projections' scales (0-d or [1]-shaped, as compressed-tensors
stores them) into one and requantize the weight rows that used the other.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.test_utils import CustomTestCase

Q_LORA_RANK = 2
KV_ROWS = 3
HIDDEN = 4
PREFIX = "model.layers.0.self_attn"
FUSED = f"{PREFIX}.fused_qkv_a_proj_with_mqa"


class _FakeDeepseekLoader(DeepseekV2WeightLoaderMixin):
    def __init__(self, params):
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=Q_LORA_RANK,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.model = SimpleNamespace()
        self.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, nextn_conf
    ):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


def _param(*shape, dtype):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    # The fused layer is a ReplicatedLinear; keep its strict shape check.
    param.weight_loader = partial(ReplicatedLinear.weight_loader, None)
    return param


def _fp8(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn)


class TestFusedQkvAPerTensorScale(CustomTestCase):
    def setUp(self):
        # Every value stays exactly representable after scaling by 1/4.
        self.q_weight = _fp8([[1.0, 2.0, -4.0, 0.5], [8.0, -1.0, 0.0, 4.0]])
        self.kv_weight = _fp8(
            [[1.0, -2.0, 3.0, 0.5], [0.25, 6.0, -8.0, 1.0], [2.0, 2.0, -0.5, 0.0]]
        )
        self.params = {
            f"{FUSED}.weight": _param(
                Q_LORA_RANK + KV_ROWS, HIDDEN, dtype=torch.float8_e4m3fn
            ),
            f"{FUSED}.weight_scale": _param(1, dtype=torch.float32),
            f"{FUSED}.input_scale": _param(1, dtype=torch.float32),
        }

    def _load(self, q_scale, kv_scale, q_input_scale, kv_input_scale):
        # Scales arrive before the weights to show the order does not matter.
        _FakeDeepseekLoader(self.params).do_load_weights(
            [
                (f"{PREFIX}.q_a_proj.weight_scale", q_scale),
                (f"{PREFIX}.kv_a_proj_with_mqa.weight_scale", kv_scale),
                (f"{PREFIX}.q_a_proj.input_scale", q_input_scale),
                (f"{PREFIX}.kv_a_proj_with_mqa.input_scale", kv_input_scale),
                (f"{PREFIX}.q_a_proj.weight", self.q_weight.clone()),
                (f"{PREFIX}.kv_a_proj_with_mqa.weight", self.kv_weight.clone()),
            ]
        )

    def _assert_dequantized_weight_preserved(self, q_scale, kv_scale):
        fused_scale = self.params[f"{FUSED}.weight_scale"]
        expected = torch.cat(
            [self.q_weight.float() * q_scale, self.kv_weight.float() * kv_scale]
        )
        torch.testing.assert_close(
            self.params[f"{FUSED}.weight"].float() * fused_scale,
            expected,
            rtol=0,
            atol=0,
        )

    def test_one_element_scales_take_the_max_and_requantize(self):
        q_scale = torch.tensor([0.5])
        kv_scale = torch.tensor([2.0])
        self._load(q_scale, kv_scale, torch.tensor([3.0]), torch.tensor([1.0]))

        torch.testing.assert_close(
            self.params[f"{FUSED}.weight_scale"].data, torch.tensor([2.0])
        )
        torch.testing.assert_close(
            self.params[f"{FUSED}.input_scale"].data, torch.tensor([3.0])
        )
        self._assert_dequantized_weight_preserved(q_scale, kv_scale)

    def test_scalar_scales_take_the_max_and_requantize(self):
        q_scale = torch.tensor(4.0)
        kv_scale = torch.tensor(1.0)
        self._load(q_scale, kv_scale, torch.tensor(1.0), torch.tensor(2.0))

        torch.testing.assert_close(
            self.params[f"{FUSED}.weight_scale"].data, torch.tensor([4.0])
        )
        torch.testing.assert_close(
            self.params[f"{FUSED}.input_scale"].data, torch.tensor([2.0])
        )
        self._assert_dequantized_weight_preserved(q_scale, kv_scale)

    def test_equal_scales_leave_the_weight_untouched(self):
        scale = torch.tensor([0.5])
        self._load(scale, scale.clone(), torch.tensor([1.0]), torch.tensor([1.0]))

        self.assertTrue(
            torch.equal(
                self.params[f"{FUSED}.weight"].view(torch.uint8),
                torch.cat([self.q_weight, self.kv_weight]).view(torch.uint8),
            )
        )
        torch.testing.assert_close(
            self.params[f"{FUSED}.weight_scale"].data, torch.tensor([0.5])
        )


if __name__ == "__main__":
    unittest.main()
