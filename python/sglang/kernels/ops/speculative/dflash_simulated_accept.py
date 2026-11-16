from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_dflash_simulated_accept_module() -> Module:
    return load_jit(
        "speculative_dflash_simulated_accept",
        cuda_files=["speculative/dflash_simulated_accept.cuh"],
        cuda_wrappers=[("dflash_simulated_accept", "dflash_simulated_accept")],
    )


@debug_kernel_api
def dflash_simulated_accept(
    *,
    candidates: torch.Tensor,
    target_bonus_tokens: Optional[torch.Tensor],
    forced_num_accept_tokens: int,
    num_correct_drafts_out: torch.Tensor,
    num_accept_tokens_out: torch.Tensor,
    bonus_tokens_out: torch.Tensor,
    out_tokens: torch.Tensor,
    fixed_token: int,
    prefix_lens: Optional[torch.Tensor] = None,
    new_seq_lens_out: Optional[torch.Tensor] = None,
) -> None:
    """Writes a forced DFlash acceptance of ``n = forced_num_accept_tokens``
    (bonus included) for every request in one launch.

    ``target_bonus_tokens`` is the target token at the forced row, ``[bs]``;
    None selects fixed mode, which writes ``fixed_token`` to the bonus and to
    every ``out_tokens`` column. Otherwise ``out_tokens`` holds the drafts
    ``candidates[:, 1:n]``, the bonus at column ``n - 1`` and zeros after it.
    ``new_seq_lens_out = prefix_lens + n`` is written when it is given.
    Every tensor is int32 or int64 with any strides; the launcher checks them.
    """
    _jit_dflash_simulated_accept_module().dflash_simulated_accept(
        candidates,
        target_bonus_tokens,
        num_correct_drafts_out,
        num_accept_tokens_out,
        bonus_tokens_out,
        out_tokens,
        prefix_lens,
        new_seq_lens_out,
        forced_num_accept_tokens,
        fixed_token,
    )
