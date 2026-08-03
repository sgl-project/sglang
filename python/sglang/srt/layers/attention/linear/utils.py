from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    NV_CUTEDSL = "nv_cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    NVIDIA_KDA = "nvidia_kda"
    PTX_KDA = "ptx_kda"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_nv_cutedsl(self):
        return self == LinearAttnKernelBackend.NV_CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_nvidia_kda(self):
        return self == LinearAttnKernelBackend.NVIDIA_KDA

    def is_ptx_kda(self):
        return self == LinearAttnKernelBackend.PTX_KDA

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


_BACKENDS: Dict[str, Optional[LinearAttnKernelBackend]] = {
    "decode": None,
    "prefill": None,
    "verify": None,
}


def initialize_linear_attn_config(
    server_args: ServerArgs, prefill_default: Optional[str] = None
):
    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or prefill_default or base

    _BACKENDS["decode"] = LinearAttnKernelBackend(decode)
    _BACKENDS["prefill"] = LinearAttnKernelBackend(prefill)

    # Verify backend. Unset -> follow decode (flashinfer -> its recurrent kernel,
    # else triton), preserving historical behavior.
    verify = server_args.linear_attn_verify_backend
    if verify is None:
        verify = decode if _BACKENDS["decode"].is_flashinfer() else "triton"
    _BACKENDS["verify"] = LinearAttnKernelBackend(verify)

    rank0_log(
        f"Linear attention kernel backend: decode={decode}, prefill={prefill}, "
        f"verify={verify}"
    )


def _get_backend(phase: str) -> LinearAttnKernelBackend:
    backend = _BACKENDS[phase]
    if backend is None:
        logger.warning(
            "linear-attn %s backend is not initialized, using triton backend", phase
        )
        backend = _BACKENDS[phase] = LinearAttnKernelBackend.TRITON
    return backend


def get_linear_attn_decode_backend() -> LinearAttnKernelBackend:
    return _get_backend("decode")


def get_linear_attn_prefill_backend() -> LinearAttnKernelBackend:
    return _get_backend("prefill")


def get_linear_attn_verify_backend() -> LinearAttnKernelBackend:
    return _get_backend("verify")


def build_verify_intermediate_state_indices(
    pool_size: int, server_args: ServerArgs, device
):
    """Per-request row index into the speculative intermediate scratch
    (`intermediate_ssm` / `intermediate_conv_window`) for the MTP /
    target_verify path: request slot i owns scratch row i.

    The scratch is allocated with one extra padding row (the `+1` in
    MambaPool.SpeculativeState, index `pool_size`). Warmup and MLP-sync
    batches can be padded past the pool capacity — under DP attention
    `get_eager_max_batch_size` ceil-aligns the eager warmup bs to attn_tp —
    and the verify kernels index this table positionally up to that padded
    bs. Size the table to the padded maximum and clamp every out-of-pool row
    onto the padding row: pad rows race onto one discard row, which is
    value-irrelevant (same convention as the ragged-verify ghost row).
    """
    import torch

    from sglang.srt.utils.common import get_eager_max_batch_size

    padded_bs = max(get_eager_max_batch_size(server_args, pool_size), pool_size)
    indices = torch.arange(pool_size, dtype=torch.int32, device=device)
    if padded_bs > pool_size:
        indices = torch.cat(
            [
                indices,
                torch.full(
                    (padded_bs - pool_size,),
                    pool_size,
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        )
    return indices
