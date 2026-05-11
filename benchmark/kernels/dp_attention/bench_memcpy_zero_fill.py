"""Microbenchmark: dst.fill_(0) + memcpy_triton(...) vs memcpy_triton_with_zero_fill(...)

Measures the kernel-level speedup for #24938 at shapes representative of
DP-attention's gather/scatter buffers (DeepSeek-V3-ish: hidden=7168,
global = `tp_size` DP ranks * `padded_per_rank`).
"""

import torch

from sglang.srt.layers.dp_attention import (
    memcpy_triton,
    memcpy_triton_with_zero_fill,
)


def _bench(fn, iters=200, warmup=20):
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


def run_shape(label, dst_rows, src_rows, hidden, offset, sz, offset_src, dtype=torch.bfloat16):
    device = "cuda"
    src = torch.randn((src_rows, hidden), dtype=dtype, device=device)
    dst = torch.empty((dst_rows, hidden), dtype=dtype, device=device)
    offset_t = torch.tensor([offset], dtype=torch.int32, device=device)
    sz_t = torch.tensor([sz], dtype=torch.int32, device=device)

    def unfused():
        dst.fill_(0)
        memcpy_triton(dst, src, 0, offset_t, sz_t, offset_src)

    def fused():
        memcpy_triton_with_zero_fill(dst, src, 0, offset_t, sz_t, offset_src)

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

    # DeepSeek-V3 hidden=7168, tp=8. Padded DP batch sizes that show up in practice.
    hidden = 7168
    for tp in (8,):
        for padded_per_rank in (256, 512, 1024, 2048):
            global_rows = tp * padded_per_rank
            for fill_ratio in (1.0, 0.5, 0.25, 0.125):
                sz = max(1, int(padded_per_rank * fill_ratio))
                run_shape(
                    f"gather  tp={tp} per={padded_per_rank} fill={fill_ratio:.3f}",
                    dst_rows=global_rows, src_rows=padded_per_rank, hidden=hidden,
                    offset=(tp // 2) * padded_per_rank, sz=sz, offset_src=False,
                )
                run_shape(
                    f"scatter tp={tp} per={padded_per_rank} fill={fill_ratio:.3f}",
                    dst_rows=padded_per_rank, src_rows=global_rows, hidden=hidden,
                    offset=(tp // 2) * padded_per_rank, sz=sz, offset_src=True,
                )
            print()


if __name__ == "__main__":
    main()
