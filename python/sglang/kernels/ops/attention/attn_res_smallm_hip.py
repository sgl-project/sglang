"""attn_res_hip for decode/verify token counts on ROCm, as one HIP kernel
(jit/csrc/kimi_k3/attn_res_smallm.cuh) without the Triton kernel's per-row LDS
round trips. Outputs match attn_res_hip up to rare 1-ulp bf16 flips (fp32
reduction order); prefix_out and the bank snapshot are bit-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_NVB = 8


def _launch_shape(hidden_size: int) -> tuple[int, int]:
    """(vectors per thread, threads per block) for H = hidden_size."""
    n_vecs = hidden_size // 8
    for threads in (448, 512, 384, 256, 320, 576, 640, 704, 768, 832, 896, 960, 1024):
        if n_vecs % threads == 0 and n_vecs // threads <= 4:
            return n_vecs // threads, threads
    threads = 512
    return -(-n_vecs // threads), threads


@cache_once
def _module(vpt: int, threads: int) -> Module:
    args = make_cpp_args(vpt, threads)
    return load_jit(
        f"kimi_k3_attn_res_smallm_{vpt}_{threads}",
        *args,
        cuda_files=["kimi_k3/attn_res_smallm.cuh"],
        cuda_wrappers=[("run", f"AttnResSmallMKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def supported(hidden_size: int, nvb: int) -> bool:
    return (
        1 <= nvb <= MAX_NVB
        and hidden_size % 8 == 0
        and _launch_shape(hidden_size)[0] <= 4
    )


def attn_res_smallm_hip(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
    out_fp8: Optional[torch.Tensor] = None,
    stream_out: Optional[torch.Tensor] = None,
) -> None:
    """attn_res_hip; stream_out (optional [T, H], row-strided ok) also receives
    the bf16 pre-norm mixture (a dspark aux capture)."""
    T, H = prefix_sum.shape
    assert supported(H, nvb), (H, nvb)
    assert addend is None or prefix_out is not None, "addend requires prefix_out"
    assert cw.dtype == torch.float32 and prefix_sum.dtype == torch.bfloat16
    vpt, threads = _launch_shape(H)
    flags = (
        (1 if addend is not None else 0)
        | (2 if write_prefix else 0)
        | (4 if ow is not None else 0)
        | (8 if out_fp8 is not None else 0)
        | (16 if stream_out is not None else 0)
    )
    # Unused operands still need a real tensor; the flags gate every access.
    _module(vpt, threads).run(
        prefix_sum,
        addend if addend is not None else prefix_sum,
        prefix_out if addend is not None else prefix_sum,
        bank,
        cw,
        ow if ow is not None else prefix_sum,
        out,
        out_fp8.view(torch.uint8) if out_fp8 is not None else out,
        stream_out if stream_out is not None else out,
        nvb,
        flags,
        float(score_eps),
        float(out_eps),
    )
