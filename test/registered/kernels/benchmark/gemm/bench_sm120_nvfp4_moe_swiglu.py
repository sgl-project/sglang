"""Fused SM120 NVFP4 SwiGLU MoE vs FlashInfer's autotuned CUTLASS MoE (unfused finalize).

Random NVFP4 weights and softmax top-k routing over all experts.
"""

import sys

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.sm120_nvfp4_moe import nvfp4_moe_swiglu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="nightly", runner_config="1-gpu-small")

# (num_experts, top_k, hidden, intermediate): Qwen3.8-Flash-Next, Qwen3-30B-A3B.
SHAPES = [(512, 10, 2560, 640), (128, 8, 2048, 768)]
_WEIGHTS = {}
_TUNED = set()


def _weights(num_experts, hidden, inter):
    key = (num_experts, hidden, inter)
    if key not in _WEIGHTS:

        def fp4(rows, cols):
            return torch.randint(
                0, 256, (num_experts, rows, cols // 2), dtype=torch.uint8, device="cuda"
            )

        def sf(rows, cols):
            return torch.randint(
                0x28, 0x48, (num_experts, rows, cols // 16), device="cuda"
            ).to(torch.uint8)

        alpha = torch.full((num_experts,), 1e-4, device="cuda")
        scale = torch.ones((), device="cuda")
        _WEIGHTS[key] = [
            fp4(2 * inter, hidden),
            sf(2 * inter, hidden),
            fp4(hidden, inter),
            sf(hidden, inter),
            alpha,
            alpha,
            scale * 64,
            scale * 32,
        ]
    return _WEIGHTS[key]


def _fused(x, ids, weights, w13, w13_sf, w2, w2_sf, a1, a2, g1, g2):
    return nvfp4_moe_swiglu(x, w13, w13_sf, w2, w2_sf, a1, a2, g1, g2, ids, weights)


def _cutlass(x, ids, weights, w13, w13_sf, w2, w2_sf, a1, a2, g1, g2):
    from flashinfer.fused_moe import cutlass_fused_moe

    return cutlass_fused_moe(
        input=x,
        token_selected_experts=ids,
        token_final_scales=weights,
        fc1_expert_weights=w13.view(torch.long),
        fc2_expert_weights=w2.view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=[
            g1,
            w13_sf.view(torch.int32),
            a1,
            g2,
            w2_sf.view(torch.int32),
            a2,
        ],
        use_fused_finalize=False,
        tune_max_num_tokens=1024,
    )[0]


@marker.parametrize("shape", SHAPES, SHAPES[:1])
@marker.parametrize(
    "num_tokens", [1, 8, 32, 64, 128, 256, 384, 512, 768, 1024], [1, 64, 512]
)
@marker.benchmark("provider", ["fused", "cutlass"])
def benchmark(shape, num_tokens, provider):
    num_experts, top_k, hidden, inter = shape
    w = _weights(num_experts, hidden, inter)
    x = torch.randn(num_tokens, hidden, device="cuda").to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    args = (x, ids.to(torch.int32), weights.float(), *w)
    if provider == "cutlass" and shape not in _TUNED:
        from flashinfer.autotuner import autotune

        with autotune(True):
            _cutlass(*args)
        _TUNED.add(shape)
    fn = _fused if provider == "fused" else _cutlass
    return marker.do_bench(fn, input_args=args)


if __name__ == "__main__":
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    ):
        print("[skip] fused NVFP4 SwiGLU MoE benchmark requires CUDA SM120")
        sys.exit(0)
    benchmark.run()
