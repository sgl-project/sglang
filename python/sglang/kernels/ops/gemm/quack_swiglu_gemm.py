"""SM120 gate_up GEMM with SwiGLU fused into its epilogue (quack warp-MMA GEMM).

Computes ``silu(x @ W_gate.T) * (x @ W_up.T)`` from a merged ``[gate; up]`` weight
``[2 * I, K]``, read in place through quack's concat layout: bf16 in and out, fp32
accumulation. The activation reads the fp32 accumulator, so gate and up are not
rounded to bf16 first as they are when ``SiluAndMul`` follows a separate GEMM.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils.common import get_device_core_count

# quack-kernels 0.6.4 launches SM120 GEMMs with programmatic dependent launch but
# never waits on the prior grid (0.6.5 adds the wait); drop once quack >= 0.6.5.
_POST_INIT_ATTRS = (("use_pdl", False),)
_TILE_M = 128
# quack 0.6.4's SM120 cooperative tile widths, widest first so ties keep it.
_TILE_N_CHOICES = (192, 160, 128)


@cache_once
def _swiglu_epilogue_mod():
    from quack.epilogue.library import linear_act_mod

    return linear_act_mod(
        "swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False
    )


@cache_once
def _num_sms(device_index: int) -> int:
    return get_device_core_count(device_index)


@cache_once
def _tile_n(m: int, n: int, num_sms: int) -> int:
    # Least wave-quantized work per SM. On RTX PRO 6000 (188 SMs) for the
    # Qwen3-0.6B gate_up (N=6144), this is within 3% of the best quack config
    # at every M from 256 to 16384.
    def work_per_sm(tile_n: int) -> int:
        tiles = -(-m // _TILE_M) * -(-n // tile_n)
        return -(-tiles // num_sms) * tile_n

    return min(_TILE_N_CHOICES, key=work_per_sm)


def quack_swiglu_gemm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        # TMA needs 16-byte rows; the epilogue splits each tile into gate/up halves.
        and x.shape[1] % 8 == 0
        and weight.shape[0] % 32 == 0
    )


def quack_swiglu_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.shape[0], weight.shape[0]
    out = x.new_empty(m, n // 2)
    _swiglu_epilogue_mod().gemm(
        x,
        weight,
        None,
        None,
        epi_args={"mAuxOut": out},
        tile_M=_TILE_M,
        tile_N=_tile_n(m, n, _num_sms(x.device.index)),
        cluster_M=1,
        cluster_N=1,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=True,
        concat_layout=("B",),
        post_init_attrs=_POST_INIT_ATTRS,
    )
    return out
