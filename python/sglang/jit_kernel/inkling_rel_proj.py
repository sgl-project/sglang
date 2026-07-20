"""CUDA-JIT latency-lean rel_logits projection for SMALL token counts, with
the optional log-scaling tau prescale folded in registers. See
csrc/tml/inkling_rel_proj.cuh; cuBLAS keeps everything above the measured
small-t band (an earlier bandwidth-oriented custom kernel lost to it at every
size)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    empty_sentinel,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rel_proj_module(d_rel: int, use_pdl: bool) -> Module:
    args = make_cpp_args(d_rel, use_pdl)
    return load_jit(
        "inkling_rel_proj",
        *args,
        cuda_files=["inkling/inkling_rel_proj.cuh"],
        cuda_wrappers=[("run", f"rel_proj_small_t<{args}>")],
    )


def rel_proj_small_t(
    r: torch.Tensor,
    proj: torch.Tensor,
    tau: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``r``: [t, h, d_rel] bf16, token rows possibly strided ((h*d_rel)-
    contiguous inner, 16B-aligned); ``proj``: [d_rel, e] bf16 contiguous;
    ``tau``: optional fp32 [t] prescale (rounds r*tau to bf16 before the dot,
    the shipped prescale semantics). Returns contiguous [t, h, e]."""
    if out is None:
        out = torch.empty(
            (r.shape[0], r.shape[1], proj.shape[1]), dtype=r.dtype, device=r.device
        )
    module = _jit_rel_proj_module(r.shape[2], is_arch_support_pdl())
    sh = tau if tau is not None else empty_sentinel(r.device, torch.float32)
    module.run(r, sh, proj, out)
    return out
