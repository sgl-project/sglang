from __future__ import annotations

import statistics

import torch
from flashinfer.testing.utils import bench_gpu_time_with_cupti

from sglang.jit_kernel.fused_q_quant_kv_write import fused_q_quant_kv_write
from sglang.srt.layers.attention.triton_ops.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
TOTAL_SLOTS = 16384
K_SCALE, V_SCALE, SCALING = 0.3, 0.5, 1.0 / (HEAD_DIM**0.5)


def _make_inputs(num_tokens, device):
    q = torch.randn(
        num_tokens, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    cache_loc = torch.randperm(TOTAL_SLOTS, device=device)[:num_tokens].to(torch.int64)
    k_cache = torch.zeros(
        TOTAL_SLOTS, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    k_scale_t = torch.tensor(K_SCALE, device=device, dtype=torch.float32)
    v_scale_t = torch.tensor(V_SCALE, device=device, dtype=torch.float32)
    return q, k, v, cache_loc, k_cache, v_cache, k_scale_t, v_scale_t


def _fused_fn(q, k, v, cache_loc, k_cache, v_cache, *_):
    def run():
        fused_q_quant_kv_write(
            q,
            k,
            v,
            k_cache,
            v_cache,
            cache_loc,
            inv_k_scale=1.0 / K_SCALE,
            inv_v_scale=1.0 / V_SCALE,
            bmm1_extra=K_SCALE * SCALING,
        )

    return run


def _unfused_fn(q, k, v, cache_loc, k_cache, v_cache, k_scale_t, v_scale_t):
    cache_loc_i32 = cache_loc.int()

    def run():
        fused_fp8_set_kv_buffer(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc_i32,
            k_scale=k_scale_t,
            v_scale=v_scale_t,
            page_size=1,
        )
        q_fp8, q_scale = scaled_fp8_quant(q.reshape(-1, q.shape[-1]).contiguous(), None)
        _ = q_scale * K_SCALE * SCALING

    return run


def _bench(run, repeat_ms=200):
    times = bench_gpu_time_with_cupti(run, repeat_time_ms=repeat_ms, cold_l2_cache=True)
    return statistics.median(times)


def main():
    device = "cuda"
    print(
        f"{'num_tokens':>10} | {'fused(us)':>12} {'unfused(us)':>12} | {'speedup':>8}"
    )
    print("-" * 56)
    for num_tokens in [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        inp = _make_inputs(num_tokens, device)
        fused_us = _bench(_fused_fn(*inp))
        unfused_us = _bench(_unfused_fn(*inp))
        print(
            f"{num_tokens:>10} | {fused_us:>12.4f} {unfused_us:>12.4f} | "
            f"{unfused_us / fused_us:>7.2f}x"
        )


if __name__ == "__main__":
    main()
