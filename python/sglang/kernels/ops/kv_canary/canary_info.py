from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.kv_canary._dispatch import use_torch_reference

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def launch_canary_info_kernel(
    *,
    canary_buf: torch.Tensor,
    out_cache_loc: torch.Tensor,
    next_token_ids: torch.Tensor,
    accept_lens: Optional[torch.Tensor],
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    min_ps: torch.Tensor,
    out: torch.Tensor,
) -> None:
    assert out_cache_loc.ndim == next_token_ids.ndim == out.ndim == 2
    assert out_cache_loc.shape == next_token_ids.shape == out.shape
    num_rows = out_cache_loc.shape[0]
    assert accept_lens is None or accept_lens.shape == (num_rows,)
    for name, tensor in (
        ("temperatures", temperatures),
        ("top_ps", top_ps),
        ("top_ks", top_ks),
        ("min_ps", min_ps),
    ):
        if tensor.numel() != num_rows:
            raise ValueError(
                f"kv-canary: {name} must hold {num_rows} rows, got {tensor.numel()}"
            )

    assert accept_lens is None or accept_lens.dtype == torch.int32
    assert temperatures.dtype == top_ps.dtype == min_ps.dtype == torch.float32
    assert top_ks.dtype == torch.int32

    out_cache_loc = out_cache_loc.to(torch.int64).contiguous()
    next_token_ids = next_token_ids.to(torch.int32).contiguous()
    accept_lens = None if accept_lens is None else accept_lens.contiguous()
    temperatures = temperatures.reshape(-1).contiguous()
    top_ps = top_ps.reshape(-1).contiguous()
    top_ks = top_ks.reshape(-1).contiguous()
    min_ps = min_ps.reshape(-1).contiguous()

    if use_torch_reference(canary_buf.device):
        from sglang.kernels.ops.kv_canary.canary_info_ref import (
            launch_canary_info_kernel_torch_reference,
        )

        launch_canary_info_kernel_torch_reference(
            canary_buf=canary_buf,
            out_cache_loc=out_cache_loc,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            out=out,
        )
        return

    _jit_canary_info_module().canary_info_cuda(
        canary_buf,
        out_cache_loc,
        next_token_ids,
        accept_lens if accept_lens is not None else next_token_ids,
        int(accept_lens is not None),
        temperatures,
        top_ps,
        top_ks,
        min_ps,
        out,
    )


@cache_once
def _jit_canary_info_module() -> Module:
    return load_jit(
        "kv_canary_info",
        cuda_files=["kv_canary/canary_info.cuh"],
        cuda_wrappers=[("canary_info_cuda", "canary::canary_info_cuda")],
    )
