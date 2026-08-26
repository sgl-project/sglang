from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    NV_CUTEDSL = "nv_cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    NVIDIA_KDA = "nvidia_kda"
    PTX_KDA = "ptx_kda"
    HELION = "helion"
    INTEL_XPU = "intel_xpu"
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

    def is_helion(self):
        return self == LinearAttnKernelBackend.HELION

    def is_intel_xpu(self):
        return self == LinearAttnKernelBackend.INTEL_XPU

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


class LinearAttnBackends(msgspec.Struct, frozen=True):
    """One runner's linear-attn kernel choice, per phase.

    Per runner, not per process: a target and its draft coexist and can want
    different kernels (only the runner whose model is GDN gets the SM100
    FlashInfer prefill default, and an explicit flag applies to whichever runner
    was launched with it).
    """

    decode: LinearAttnKernelBackend
    prefill: LinearAttnKernelBackend
    verify: LinearAttnKernelBackend


def resolve_linear_attn_backends(
    prefill_default: Optional[str] = None,
) -> LinearAttnBackends:
    """This runner's kernel choice from the published leaves.

    ``prefill_default`` is the caller's own auto-default (the SM100 GDN
    domain); an explicitly configured ``--linear-attn-prefill-backend`` wins.
    """
    mamba = get_exec().mamba
    base = mamba.linear_attn_backend
    decode = LinearAttnKernelBackend(mamba.linear_attn_decode_backend or base)
    prefill = LinearAttnKernelBackend(
        mamba.linear_attn_prefill_backend or prefill_default or base
    )

    # Unset verify follows decode (flashinfer -> its recurrent kernel, else triton).
    verify = mamba.linear_attn_verify_backend
    if verify is None:
        verify = decode.value if decode.is_flashinfer() else "triton"

    backends = LinearAttnBackends(
        decode=decode, prefill=prefill, verify=LinearAttnKernelBackend(verify)
    )
    rank0_log(
        f"Linear attention kernel backend: decode={backends.decode.value}, "
        f"prefill={backends.prefill.value}, verify={backends.verify.value}"
    )
    return backends


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
