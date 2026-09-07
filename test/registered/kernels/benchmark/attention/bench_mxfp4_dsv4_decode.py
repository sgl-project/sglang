"""Benchmark: JIT MXFP4 DSV4 decode vs the FP8 sparse decode (FlashMLA).

Same query shapes, same indices, same load; only the cache format differs.
Each case covers one layer kind: C0 (SWA only), C4 (SWA + 64-token compressed
cache) and C128 (SWA + 1024-token compressed cache), at two batch sizes.
Timed with CUDA events over the full call (scheduler + main + combine).
"""

import math

import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta as Fp8SchedMeta
from sgl_kernel.flash_mla import flash_mla_with_kvcache as fp8_decode

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import (
    FlashMLASchedMeta as Mxfp4SchedMeta,
)
from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import (
    flash_mla_with_kvcache_dsv4_mxfp4 as mxfp4_decode,
)
from sglang.kernels.ops.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    quantize_dsv4_mxfp4_k_cache_into,
)
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_HEAD_DIM = 512
_FP8_BYTES_PER_TOKEN = 584


def _pack_fp8(k_bf16: torch.Tensor) -> torch.Tensor:
    """BF16 [N, 512] -> FlashMLA fp8 row [N, 584] uint8."""
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
    n = k_bf16.shape[0]
    row = torch.empty(
        (n, _FP8_BYTES_PER_TOKEN), dtype=torch.uint8, device=k_bf16.device
    )
    row[:, :448] = pack.k_nope_fp8.view(torch.uint8)
    row[:, 448:576] = pack.k_rope_bf16.view(torch.uint8)
    row[:, 576:583] = pack.scale_k_nope_ue8m0
    return row


def _make_caches(num_pages: int, page_size: int, gen):
    dev = torch.device("cuda")
    cap = num_pages * page_size
    src = (
        torch.randn((cap, _HEAD_DIM), dtype=torch.bfloat16, device=dev, generator=gen)
        / 10
    )
    loc = torch.arange(cap, dtype=torch.int32, device=dev)

    mxfp4_raw = torch.zeros(
        (num_pages, page_size * MXFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=dev
    )
    quantize_dsv4_mxfp4_k_cache_into(
        cache_k=src, kv_buffer=mxfp4_raw, loc=loc, page_size=page_size
    )
    mxfp4_4d = mxfp4_raw.view(num_pages, page_size, 1, MXFP4_BYTES_PER_TOKEN)

    fp8_row = _pack_fp8(src)
    fp8_4d = fp8_row.view(num_pages, page_size, 1, _FP8_BYTES_PER_TOKEN)
    return mxfp4_4d, fp8_4d


def _make_indices(b: int, width: int, capacity: int, gen):
    dev = torch.device("cuda")
    rows, lengths = [], []
    for i in range(b):
        begin = i * width
        row = torch.arange(begin, begin + width, dtype=torch.int32, device=dev)
        row = row[torch.randperm(width, device=dev, generator=gen)]
        rows.append(row)
        lengths.append(width)
    return torch.stack(rows).unsqueeze(1).contiguous(), torch.tensor(
        lengths, dtype=torch.int32, device=dev
    )


def _make_case(h_q: int, b: int, swa_pages: int, extra_cfg):
    """Build (mxfp4 args, fp8 args) for one layer configuration."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    sm = 1.0 / math.sqrt(_HEAD_DIM)

    swa_page_size = 256
    swa_topk = 128
    mxfp4_swa, fp8_swa = _make_caches(swa_pages, swa_page_size, gen)
    swa_idx, swa_len = _make_indices(b, swa_topk, swa_pages * swa_page_size, gen)
    q = torch.randn((b, 1, h_q, _HEAD_DIM), dtype=torch.bfloat16, device=dev) / 10
    attn_sink = torch.zeros(h_q, dtype=torch.float32, device=dev)

    ex_mx = ex_f8 = ex_idx = ex_len = None
    if extra_cfg is not None:
        e_page, e_topk = extra_cfg
        e_pages = max((b * e_topk + e_page - 1) // e_page, 1)
        ex_mx, ex_f8 = _make_caches(e_pages, e_page, gen)
        ex_idx, ex_len = _make_indices(b, e_topk, e_pages * e_page, gen)

    mx_meta = Mxfp4SchedMeta()
    f8_meta = Fp8SchedMeta()
    return dict(
        mxfp4=lambda: mxfp4_decode(
            q=q,
            k_cache=mxfp4_swa,
            indices=swa_idx,
            topk_length=swa_len,
            attn_sink=attn_sink,
            tile_scheduler_metadata=mx_meta,
            softmax_scale=sm,
            extra_k_cache=ex_mx,
            extra_indices_in_kvcache=ex_idx,
            extra_topk_length=ex_len,
        ),
        fp8=lambda: fp8_decode(
            q=q,
            k_cache=fp8_swa,
            head_dim_v=_HEAD_DIM,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=f8_meta,
            softmax_scale=sm,
            is_fp8_kvcache=True,
            indices=swa_idx,
            topk_length=swa_len,
            attn_sink=attn_sink,
            extra_k_cache=ex_f8,
            extra_indices_in_kvcache=ex_idx,
            extra_topk_length=ex_len,
        ),
    )


_CASES = {
    "c0_b8": dict(h_q=64, b=8, swa_pages=8, extra_cfg=None),
    "c4_b8": dict(h_q=64, b=8, swa_pages=8, extra_cfg=(64, 512)),
    "c128_b8": dict(h_q=128, b=8, swa_pages=8, extra_cfg=(2, 1024)),
    "c4_b32": dict(h_q=64, b=32, swa_pages=16, extra_cfg=(64, 512)),
}


@marker.parametrize("case", list(_CASES), ["c4_b8", "c4_b32"])
@marker.benchmark("impl", ["mxfp4", "fp8"], unit="us")
def benchmark(case: str, impl: str):
    fns = _make_case(**_CASES[case])
    return marker.do_bench(
        fns[impl],
        # Closure-based fns take no args; the JIT scheduler holds its own
        # buffers, so memory accounting is skipped (both impls are compared
        # under identical L2 conditions).
        memory_args=(),
        memory_output=(),
        graph_clone_args=(),
    )


if __name__ == "__main__":
    benchmark.run()
