"""Microbench for #24938: dst.fill_(0) + memcpy_triton vs memcpy_triton_with_zero_fill."""

from typing import Callable

import torch

from sglang.srt.layers.dp_attention import (
    memcpy_triton,
    memcpy_triton_with_zero_fill,
)


def _bench(fn: Callable[[], None], iters: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def run_shape(
    label: str,
    dst_rows: int,
    src_rows: int,
    hidden: int,
    offset: int,
    sz: int,
    offset_src: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    device = "cuda"
    src = torch.randn((src_rows, hidden), dtype=dtype, device=device)
    dst = torch.empty((dst_rows, hidden), dtype=dtype, device=device)
    offset_t = torch.tensor([offset], dtype=torch.int32, device=device)
    sz_t = torch.tensor([sz], dtype=torch.int32, device=device)

    def unfused():
        dst.fill_(0)
        memcpy_triton(
            dst=dst, src=src, dim=0, offset=offset_t, sz=sz_t, offset_src=offset_src
        )

    def fused():
        memcpy_triton_with_zero_fill(
            dst=dst, src=src, dim=0, offset=offset_t, sz=sz_t, offset_src=offset_src
        )

    t_unfused = _bench(unfused)
    t_fused = _bench(fused)

    dst_bytes = dst.numel() * dst.element_size()
    src_copy_bytes = sz * hidden * src.element_size()
    unfused_traffic = dst_bytes + 2 * src_copy_bytes
    fused_traffic = dst_bytes + src_copy_bytes

    speedup_pct = (t_unfused - t_fused) / t_unfused * 100
    print(
        f"[{label}] dst={dst_rows}x{hidden} sz={sz} ({sz/dst_rows*100:.1f}% coverage) | "
        f"unfused={t_unfused*1000:7.2f}us  fused={t_fused*1000:7.2f}us  "
        f"delta={speedup_pct:+5.1f}% | "
        f"BW unfused={unfused_traffic/1e9/(t_unfused/1000):6.1f}GB/s fused={fused_traffic/1e9/(t_fused/1000):6.1f}GB/s"
    )


def main():
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    # Common LLM hidden sizes:
    #   2048 = small LLMs, 4096 = Llama-7B / Qwen3 / Pixtral,
    #   5120 = Llama-13B / Llama-4-Scout / DeepSeek-V2,
    #   7168 = DeepSeek-V3 / Kimi-K2, 8192 = Llama-70B / Qwen-72B.
    tp = 8
    for hidden in (2048, 4096, 5120, 7168, 8192):
        for padded_per_rank in (128, 256, 512, 1024, 2048, 4096):
            global_rows = tp * padded_per_rank
            sz = padded_per_rank
            run_shape(
                f"gather  hidden={hidden:5d} per={padded_per_rank:5d}",
                dst_rows=global_rows,
                src_rows=padded_per_rank,
                hidden=hidden,
                offset=(tp // 2) * padded_per_rank,
                sz=sz,
                offset_src=False,
            )
            run_shape(
                f"scatter hidden={hidden:5d} per={padded_per_rank:5d}",
                dst_rows=padded_per_rank,
                src_rows=global_rows,
                hidden=hidden,
                offset=(tp // 2) * padded_per_rank,
                sz=sz,
                offset_src=True,
            )
        print()


if __name__ == "__main__":
    main()
