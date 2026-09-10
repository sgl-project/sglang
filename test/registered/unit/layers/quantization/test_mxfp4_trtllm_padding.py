import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")


def make_layer(weights):
    layer = torch.nn.Module()
    names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )
    for name, tensor in zip(names, weights):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.num_experts = weights[0].shape[0]
    layer.num_local_experts = layer.num_experts
    layer.moe_ep_rank = 0
    return layer


def make_weights(intermediate, hidden=256, device="cpu"):
    experts = 8
    w13 = torch.randint(
        -128,
        128,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device=device,
    )
    w2 = torch.randint(
        -128,
        128,
        (experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device=device,
    )
    s13 = (
        torch.randint(-6, -3, (experts, 2 * intermediate, hidden // 32), device=device)
        .float()
        .exp2()
    )
    s2 = (
        torch.randint(-6, -3, (experts, hidden, intermediate // 32), device=device)
        .float()
        .exp2()
    )
    return w13, w2, s13, s2


class TestMxfp4TrtllmPadding(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "Requires Blackwell",
    )
    def test_tp4_matches_unsharded_experts(self):
        torch.manual_seed(42)
        weights = make_weights(2304, hidden=5120, device="cuda")

        def prepare(tensors):
            layer = make_layer(tensors)
            method = object.__new__(mxfp4.Mxfp4FlashinferTrtllmMoEMethod)
            method._fp8 = Mock()
            method.prefix = "test.experts"
            method.flashinfer_mxfp4_moe_precision = "default"
            method.process_weights_after_loading(layer)
            method.create_moe_runner(layer, SimpleNamespace(swiglu_limit=10.0))
            return method, layer

        full = prepare([tensor.clone() for tensor in weights])
        shards = []
        for rank in range(4):
            start = rank * 576
            end = start + 576
            w13, w2, s13, s2 = weights
            shard = (
                torch.cat(
                    (w13[:, start:end], w13[:, 2304 + start : 2304 + end]), dim=1
                ),
                w2[..., start // 2 : end // 2].contiguous(),
                torch.cat(
                    (s13[:, start:end], s13[:, 2304 + start : 2304 + end]), dim=1
                ),
                s2[..., start // 32 : end // 32].contiguous(),
            )
            shards.append(prepare(shard))

        with (
            patch.object(mxfp4, "get_tp_group", return_value=None),
            patch.object(mxfp4, "is_allocation_symmetric", return_value=False),
            patch.object(mxfp4, "use_symmetric_memory", return_value=nullcontext()),
        ):
            for tokens in (1, 16, 64):
                with self.subTest(tokens=tokens):
                    x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="cuda")
                    logits = torch.randn(tokens, 8, device="cuda")
                    scores, ids = logits.softmax(-1).topk(6, dim=-1)
                    topk = StandardTopKOutput(scores, ids.to(torch.int32), logits)
                    dispatch = StandardDispatchOutput(x, None, topk)
                    reference = full[0].apply(full[1], dispatch).hidden_states.float()
                    actual = sum(
                        method.apply(layer, dispatch).hidden_states.float()
                        for method, layer in shards
                    )
                    relative_rmse = (
                        (
                            (actual - reference).square().mean()
                            / reference.square().mean()
                        )
                        .sqrt()
                        .item()
                    )
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(relative_rmse, 0.01)
                    print(
                        f"tokens={tokens} TP4 vs unsharded relative_rmse={relative_rmse:.6f}",
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
