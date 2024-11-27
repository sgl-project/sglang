import numbers
from typing import Optional
import torch
import triton
from torch.nn.parameter import Parameter
from torch.nn import init
from sglang.srt.layers.fused_moe_triton.fused_moe  import fused_moe as fused_moe_sglang
from sglang.srt.layers.fused_moe_triton.fused_moe  import get_moe_configs as get_moe_configs_sglang
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe as fused_moe_vllm
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs as get_moe_configs_vllm

def fused_moe_vllm_api(x, w1, w2, input_gating, topk, w1_scale, w2_scale, a1_scale, a2_scale):
    return fused_moe_vllm(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )

def fused_moe_sglang_api(x, w1, w2, input_gating, topk, w1_scale, w2_scale, a1_scale, a2_scale):
    return fused_moe_sglang(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'], 
        x_vals=list(range(1, 513)),
        line_arg='provider',
        line_vals=['vllm fused moe', 'sglang fused moe',],
        line_names=["vllm fused moe", 'sglang fused moe',],
        styles=[('blue', '-'), ('green', '-'),],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance", 
        args={},
    )
)
def benchmark(batch_size, provider):
    print(f'benchmark for {batch_size}.')
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    dtype = torch.bfloat16
    init_dtype = torch.float16
    num_tokens = batch_size
    num_experts = 64
    hidden_size = 3584
    shard_intermediate_size = 1280
    topk = 8
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts,
                         shard_intermediate_size,
                         hidden_size,
                         dtype=init_dtype)
    w2 = torch.randn(num_experts,
                        hidden_size,
                        shard_intermediate_size // 2,
                        dtype=init_dtype)
    input_gating = torch.randn(num_tokens,
                                num_experts,
                                dtype=torch.float32)
    w1_scale = torch.randn(num_experts, dtype=torch.float32)
    w2_scale = torch.randn(num_experts, dtype=torch.float32)
    a1_scale = torch.randn(1, dtype=torch.float32)
    a2_scale = torch.randn(1, dtype=torch.float32)

    w1 = w1.to(torch.float8_e4m3fn)
    w2 = w2.to(torch.float8_e4m3fn)

    # warmup for fused_moe_vllm
    for _ in range(10):
        y = fused_moe_vllm_api(x, w1, w2, input_gating, topk, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale)
    torch.cuda.synchronize()
    # warmup for fused_moe_sglang
    for _ in range(10):
        y = fused_moe_sglang_api(x, w1, w2, input_gating, topk, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale)
    torch.cuda.synchronize()


    quantiles = [0.5, 0.2, 0.8]

    if provider == 'vllm fused moe':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_moe_vllm_api(x, w1, w2, input_gating, topk, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale)[0], quantiles=quantiles)
    elif provider == 'sglang fused moe':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_moe_sglang_api(x, w1, w2, input_gating, topk, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale)[0], quantiles=quantiles)
    return ms, min_ms, max_ms

benchmark.run(show_plots=True, print_data=True, save_path='./configs/benchmark_ops/vllm_sglang_fused_moe/')
