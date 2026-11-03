"""Bitwise tests for the SM120 fused NVFP4 SwiGLU MoE against FlashInfer CUTLASS.

Turns red if the fused kernel's quantization, GEMM accumulation order, SwiGLU
epilogue or top-k finalize drifts from ``cutlass_fused_moe`` with the unfused
finalize, or if routing (empty experts, skipped expert ids, unit boundaries at
multiples of 32 tokens) or CUDA-graph replay goes wrong.
"""

import unittest

import torch

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# (num_experts, top_k, hidden, intermediate): Qwen3.8-Flash-Next, Qwen3-30B-A3B.
SHAPES = [(512, 10, 2560, 640), (128, 8, 2048, 768)]
TOKENS = [1, 2, 7, 31, 32, 33, 128, 333, 512]


def _make_weights(num_experts, hidden, inter, gen):
    def fp4(rows, cols):
        return torch.randint(
            0, 256, (num_experts, rows, cols // 2), dtype=torch.uint8, generator=gen
        ).cuda()

    def scales(rows, cols):
        # E4M3 bytes 0x28..0x47 are the normal values 0.125 .. 3.75.
        sf = torch.randint(
            0x28,
            0x48,
            (num_experts, rows, cols // 16),
            dtype=torch.uint8,
            generator=gen,
        )
        return sf.cuda().view(torch.float8_e4m3fn)

    def alphas(scale):
        return (scale * (0.5 + torch.rand(num_experts, generator=gen))).cuda()

    gs1 = torch.tensor(64.0, device="cuda")
    gs2 = torch.tensor(32.0, device="cuda")
    return dict(
        w13=fp4(2 * inter, hidden),
        w13_sf=scales(2 * inter, hidden),
        w2=fp4(hidden, inter),
        w2_sf=scales(hidden, inter),
        alpha1=alphas(1.0 / (64.0 * 256.0)),
        alpha2=alphas(1.0 / (32.0 * 128.0)),
        gs1=gs1,
        gs2=gs2,
    )


def _make_inputs(num_tokens, num_experts, top_k, hidden, gen, x_scale=1.0):
    x = (torch.randn(num_tokens, hidden, generator=gen) * x_scale).to(torch.bfloat16)
    if num_tokens > 2:
        x[1].zero_()
    logits = torch.randn(num_tokens, num_experts, generator=gen)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return x.cuda(), ids.to(torch.int32).cuda(), weights.float().cuda()


def _scale_list(w):
    return [
        w["gs1"],
        w["w13_sf"].view(torch.int32),
        w["alpha1"],
        w["gs2"],
        w["w2_sf"].view(torch.int32),
        w["alpha2"],
    ]


def _cutlass(x, ids, weights, w):
    from flashinfer.fused_moe import cutlass_fused_moe

    return cutlass_fused_moe(
        input=x,
        token_selected_experts=ids,
        token_final_scales=weights,
        fc1_expert_weights=w["w13"].view(torch.long),
        fc2_expert_weights=w["w2"].view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=_scale_list(w),
        use_fused_finalize=False,
    )[0]


def _fused(x, ids, weights, w, out=None):
    from sglang.kernels.ops.gemm.sm120_nvfp4_moe import nvfp4_moe_swiglu

    return nvfp4_moe_swiglu(
        x,
        w["w13"],
        w["w13_sf"],
        w["w2"],
        w["w2_sf"],
        w["alpha1"],
        w["alpha2"],
        w["gs1"],
        w["gs2"],
        ids,
        weights,
        out=out,
    )


def _assert_bitwise(test, ref, got, msg):
    neq = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
    test.assertEqual(neq, 0, f"{msg}: {neq}/{ref.numel()} elements differ")


@unittest.skipIf(
    not torch.cuda.is_available() or get_device_sm() != 120,
    "SM120 fused NVFP4 MoE requires an SM120 GPU",
)
class TestSm120Nvfp4MoeSwiglu(CustomTestCase):
    def test_bitwise_vs_cutlass(self):
        for num_experts, top_k, hidden, inter in SHAPES:
            gen = torch.Generator().manual_seed(num_experts + hidden)
            w = _make_weights(num_experts, hidden, inter, gen)
            for num_tokens in TOKENS:
                for x_scale in (1.0, 8.0):
                    x, ids, weights = _make_inputs(
                        num_tokens, num_experts, top_k, hidden, gen, x_scale
                    )
                    msg = f"E={num_experts} T={num_tokens} x_scale={x_scale}"
                    with self.subTest(msg):
                        ref = _cutlass(x, ids, weights, w)
                        got = _fused(x, ids, weights, w)
                        self.assertFalse(torch.isnan(ref.float()).any().item(), msg)
                        _assert_bitwise(self, ref, got, msg)

    def test_skipped_experts_and_hot_expert(self):
        num_experts, top_k, hidden, inter = SHAPES[0]
        gen = torch.Generator().manual_seed(7)
        w = _make_weights(num_experts, hidden, inter, gen)
        x, ids, weights = _make_inputs(96, num_experts, top_k, hidden, gen)
        ids[:, 0] = 3  # 96 tokens on one expert: three full units
        ids[5, 2:] = -1
        ids[9, :] = -1  # every expert skipped: the output row is zero
        _assert_bitwise(
            self, _cutlass(x, ids, weights, w), _fused(x, ids, weights, w), "masked"
        )
        self.assertEqual(int(_fused(x, ids, weights, w)[9].abs().sum()), 0)

    def test_cuda_graph_replay(self):
        num_experts, top_k, hidden, inter = SHAPES[0]
        gen = torch.Generator().manual_seed(11)
        w = _make_weights(num_experts, hidden, inter, gen)
        x, ids, weights = _make_inputs(64, num_experts, top_k, hidden, gen)
        out = torch.empty_like(x)
        _fused(x, ids, weights, w, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fused(x, ids, weights, w, out=out)
        for seed in range(3):
            g = torch.Generator().manual_seed(100 + seed)
            nx, nids, nweights = _make_inputs(64, num_experts, top_k, hidden, g)
            x.copy_(nx)
            ids.copy_(nids)
            weights.copy_(nweights)
            out.fill_(float("nan"))
            graph.replay()
            _assert_bitwise(self, _cutlass(x, ids, weights, w), out, f"replay {seed}")

    def test_dispatch_declines_unsupported_inputs(self):
        from sglang.kernels.ops.gemm.sm120_nvfp4_moe import (
            SWIGLU_MAX_TOKENS,
            try_nvfp4_moe_swiglu,
        )

        num_experts, top_k, hidden, inter = SHAPES[1]
        gen = torch.Generator().manual_seed(3)
        w = _make_weights(num_experts, hidden, inter, gen)
        scales = [
            w["gs1"],
            w["w13_sf"],
            w["alpha1"],
            w["gs2"],
            w["w2_sf"],
            w["alpha2"],
        ]
        x, ids, weights = _make_inputs(
            SWIGLU_MAX_TOKENS + 1, num_experts, top_k, hidden, gen
        )
        args = (w["w13"], w["w2"], scales, ids, weights)
        self.assertIsNone(try_nvfp4_moe_swiglu(x, *args, output_dtype=torch.bfloat16))
        self.assertIsNone(
            try_nvfp4_moe_swiglu(x[:4].half(), *args, output_dtype=torch.float16)
        )
        per_expert = list(scales)
        per_expert[0] = w["gs1"].expand(num_experts).contiguous()
        self.assertIsNone(
            try_nvfp4_moe_swiglu(
                x[:4],
                w["w13"],
                w["w2"],
                per_expert,
                ids[:4],
                weights[:4],
                output_dtype=torch.bfloat16,
            )
        )


if __name__ == "__main__":
    unittest.main()
