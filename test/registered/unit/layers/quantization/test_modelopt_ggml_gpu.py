import unittest
from types import SimpleNamespace

import gguf
import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    TopKConfig,
)
from sglang.srt.layers.quantization.modelopt_ggml import (
    ModelOptGGMLLinearMethod,
    ModelOptGGMLMoEMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def packed_weight(rng, shape, algo):
    block, payload = (32, 34) if algo == "Q8_0" else (256, 66)
    raw = rng.integers(
        0, 256, (*shape[:-1], shape[-1] // block, payload), dtype=np.uint8
    )
    raw[..., :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    if algo == "Q8_0":
        raw[..., 2:] = rng.integers(-32, 33, raw[..., 2:].shape, dtype=np.int8).view(
            np.uint8
        )
    dense = gguf.dequantize(
        raw.reshape(*shape[:-1], -1), gguf.GGMLQuantizationType[algo]
    )
    return torch.from_numpy(raw).cuda(), torch.from_numpy(dense).cuda()


def make_moe(rng, experts, *, gated, activation):
    w1, dense1 = packed_weight(rng, (experts, 512 if gated else 256, 256), "IQ2_XXS")
    w2, dense2 = packed_weight(rng, (experts, 256, 256), "IQ2_XXS")
    layer = torch.nn.Module()
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.num_fused_shared_experts = 0
    layer.use_presharded_weights = False
    config = MoeRunnerConfig(is_gated=gated, activation=activation)
    layer.moe_runner_config = config
    method = ModelOptGGMLMoEMethod("IQ2_XXS")
    method.create_weights(layer, experts, 256, 256, torch.bfloat16)
    layer.cuda()
    layer.w13_weight.data.copy_(w1)
    layer.w2_weight.data.copy_(w2)
    method.create_moe_runner(layer, config)
    return layer, method, dense1, dense2


class TestModelOptGGMLGPU(CustomTestCase):
    def assert_relative_error(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        self.assertLess(error.item(), 0.05)

    def test_linear_matches_independently_decoded_weights(self):
        """Packed row strides and vector/batched dispatch must preserve the decoded linear map."""
        rng = np.random.default_rng(123)
        torch.manual_seed(123)
        for algo in ("Q8_0", "IQ2_XXS"):
            packed, dense = packed_weight(rng, (128, 256), algo)
            layer = torch.nn.Module()
            layer.register_parameter(
                "weight", torch.nn.Parameter(packed, requires_grad=False)
            )
            method = ModelOptGGMLLinearMethod(algo)
            bias = torch.randn(128, dtype=torch.bfloat16, device="cuda") * 0.01
            for tokens in (1, 7, 65):
                with self.subTest(algo=algo, tokens=tokens):
                    x = (
                        torch.randn(1, tokens, 256, dtype=torch.bfloat16, device="cuda")
                        * 0.1
                    )
                    expected = F.linear(x.float(), dense, bias.float())
                    self.assert_relative_error(method.apply(layer, x, bias), expected)

    def test_moe_matches_decoded_experts_for_gated_and_ungated_activations(self):
        """ReLU2 must retain the full intermediate width; gated SiLU must keep its existing split."""
        rng = np.random.default_rng(456)
        torch.manual_seed(456)
        for algo in ("IQ2_XXS",):
            for gated, activation in ((False, "relu2"), (True, "silu")):
                layer, method, dense1, dense2 = make_moe(
                    rng, 4, gated=gated, activation=activation
                )
                for tokens in (1, 7, 65):
                    with self.subTest(algo=algo, gated=gated, tokens=tokens):
                        x = (
                            torch.randn(
                                tokens, 256, dtype=torch.bfloat16, device="cuda"
                            )
                            * 0.1
                        )
                        ids = (
                            torch.arange(tokens * 2, device="cuda")
                            .remainder(4)
                            .view(tokens, 2)
                            .int()
                        )
                        weights = torch.rand(
                            tokens, 2, dtype=torch.float32, device="cuda"
                        )
                        weights = weights / weights.sum(-1, keepdim=True) * 1.7
                        if tokens == 65:
                            router_logits = torch.randn(tokens, 4, device="cuda")
                            weights, ids = router_logits.softmax(-1).topk(2, dim=-1)
                            weights = weights / weights.sum(-1, keepdim=True)
                            ids = ids.int()
                        first = torch.einsum(
                            "tkoh,th->tko", dense1[ids.long()], x.float()
                        ).to(x.dtype)
                        if gated:
                            gate, up = first.chunk(2, dim=-1)
                            intermediate = F.silu(gate) * up
                        else:
                            intermediate = F.relu(first).square()
                        per_expert = torch.einsum(
                            "tkhi,tki->tkh", dense2[ids.long()], intermediate.float()
                        ).to(x.dtype)
                        expected = (per_expert * weights.unsqueeze(-1)).sum(1)
                        topk_output = (
                            StandardTopKOutputPacked(weights, ids, None, ids)
                            if tokens == 7
                            else StandardTopKOutput(weights, ids, None)
                        )
                        if tokens == 65:
                            layer.layer_id = 0
                            topk_output = BypassedTopKOutput(
                                x, router_logits, TopKConfig(top_k=2, torch_native=True)
                            )
                        dispatch = SimpleNamespace(
                            hidden_states=x, topk_output=topk_output
                        )
                        self.assert_relative_error(
                            method.apply(layer, dispatch).hidden_states, expected
                        )

    def test_moe_prefill_exceeding_cuda_grid_limit_matches_decoded_experts(self):
        """Large prefills must preserve every token beyond the CUDA grid-z limit."""
        rng = np.random.default_rng(789)
        torch.manual_seed(789)
        tokens, experts, topk = 7282, 16, 9
        layer, method, dense1, dense2 = make_moe(
            rng, experts, gated=False, activation="relu2"
        )
        x = torch.randn(tokens, 256, dtype=torch.bfloat16, device="cuda") * 0.1
        ids = torch.arange(tokens * topk, device="cuda").remainder(experts)
        ids = ids.view(tokens, topk).int()
        weights = torch.rand(tokens, topk, device="cuda").softmax(-1)
        expected = torch.zeros(tokens, 256, dtype=torch.float32, device="cuda")
        for expert in range(experts):
            rows, slots = torch.where(ids == expert)
            first = F.linear(x[rows].float(), dense1[expert]).to(x.dtype)
            intermediate = F.relu(first).square()
            output = F.linear(intermediate.float(), dense2[expert]).to(x.dtype)
            expected.index_add_(0, rows, output.float() * weights[rows, slots, None])
        dispatch = SimpleNamespace(
            hidden_states=x, topk_output=StandardTopKOutput(weights, ids, None)
        )
        actual = method.apply(layer, dispatch).hidden_states
        self.assert_relative_error(actual, expected)
        self.assert_relative_error(actual[-1:], expected[-1:])


if __name__ == "__main__":
    unittest.main()
