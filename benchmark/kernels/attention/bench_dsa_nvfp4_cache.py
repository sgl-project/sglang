"""Compare basic NVFP4 DSA gather with the FlashMLA packed-FP8 gather.

This benchmark measures only cache materialization, not sparse attention.  The
comparison path is the 656-byte/token layout used by the DeepSeekMLA/FlashMLA
implementation: 512 FP8 latent values, four FP32 scales, and 64 BF16 RoPE
values.  The NVFP4 path stores a 324-byte/token row and materializes selected
rows into compact FP8 for TRTLLM-GEN.
"""

from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    gather_dequant_requant_fp8_paged,
)
from sglang.kernels.ops.attention.dsa.nvfp4_mla_cache import (
    gather_dequant_nvfp4_mla_cache_generation,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-tokens", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    head_dim = 576
    page_size = 64
    source = torch.randn(
        args.pool_tokens, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    global_scale = torch.ones(1, dtype=torch.float32, device=device)

    nv_data, nv_scales, _ = NVFP4KVQuantizeUtil.quantize(source, global_scale)
    flashmla_cache = quantize_k_cache(source.view(-1, page_size, 1, head_dim)).view(
        args.pool_tokens, 1, 656
    )
    fp8_cache = source.to(torch.float8_e4m3fn)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Persistent bytes/token: "
        f"NVFP4={nv_data[0].nbytes + nv_scales[0].nbytes}, "
        f"FlashMLA-packed-FP8={flashmla_cache[0].nbytes}, raw-FP8={fp8_cache[0].nbytes}"
    )
    print("batch topk selected  nvfp4_ms  flashmla_ms  raw_fp8_ms  nv/flash")

    for batch in args.batch_sizes:
        physical = torch.randint(
            1,
            args.pool_tokens,
            (batch, args.topk),
            dtype=torch.int32,
            device=device,
        )
        flat = physical.reshape(-1)
        flat_long = flat.long()
        padded_rows = ((physical.numel() + page_size - 1) // page_size) * page_size
        nv_output = torch.empty(
            (padded_rows, 1, head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        nv_indices = torch.empty_like(physical)

        def run_nvfp4(selected=physical, output=nv_output, compact_indices=nv_indices):
            return gather_dequant_nvfp4_mla_cache_generation(
                nv_data.view(torch.uint8),
                nv_scales.view(torch.uint8),
                selected,
                global_scale,
                head_dim=head_dim,
                page_size=page_size,
                output=output,
                compact_indices=compact_indices,
            )

        def run_flashmla(selected=flat):
            return gather_dequant_requant_fp8_paged(flashmla_cache, selected)

        def run_raw_fp8(selected=flat_long):
            return fp8_cache.index_select(0, selected)

        nv_ms = bench_ms(run_nvfp4, args.warmup, args.iters)
        flash_ms = bench_ms(run_flashmla, args.warmup, args.iters)
        raw_ms = bench_ms(run_raw_fp8, args.warmup, args.iters)
        print(
            f"{batch:5d} {args.topk:4d} {batch * args.topk:8d} "
            f"{nv_ms:10.3f} {flash_ms:12.3f} {raw_ms:11.3f} "
            f"{nv_ms / flash_ms:9.2f}x"
        )

    # Report a simple reconstruction error for the final shape.
    compact, remapped = run_nvfp4()
    reconstructed = compact.view(-1, 1, head_dim).index_select(
        0, remapped.reshape(-1).long()
    )
    reference = source.index_select(0, flat_long).to(torch.float8_e4m3fn)
    rel = (
        reconstructed.float() - reference.float()
    ).abs().mean() / reference.float().abs().mean()
    print(f"NVFP4->FP8 mean relative absolute error: {rel.item():.4f}")


if __name__ == "__main__":
    main()
