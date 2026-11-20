"""SM120 gate_up GEMM with SwiGLU fused into its epilogue (quack warp-MMA GEMM).

Computes ``silu(x @ W_gate.T) * (x @ W_up.T)`` from a merged ``[gate; up]`` weight
``[2 * I, K]``, read in place through quack's concat layout: bf16 in and out, fp32
accumulation. The activation reads the fp32 accumulator, so gate and up are not
rounded to bf16 first as they are when ``SiluAndMul`` follows a separate GEMM.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils.common import ceil_div
from sglang.srt.utils.device import get_device_core_count

# quack 0.6.4's SM120 cooperative tile widths, widest first so ties keep it; 0.6.5
# also accepts other multiples of 32, which were not measured here.
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
def _tile_shape(m: int, n: int, num_sms: int) -> tuple[int, int]:
    # Least wave-quantized work per SM; ties keep the taller, then wider tile.
    # 64-row tiles only within one wave; measured slower past it (Qwen3-0.6B, SM120).
    def num_tiles(tile_m: int, tile_n: int) -> int:
        return ceil_div(m, tile_m) * ceil_div(n, tile_n)

    shapes = [(128, tile_n) for tile_n in _TILE_N_CHOICES]
    shapes += [
        (64, tile_n) for tile_n in _TILE_N_CHOICES if num_tiles(64, tile_n) <= num_sms
    ]

    def work_per_sm(shape: tuple[int, int]) -> int:
        return ceil_div(num_tiles(*shape), num_sms) * shape[0] * shape[1]

    return min(shapes, key=work_per_sm)


def quack_swiglu_gemm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        # 0 rows kill the process in quack's launcher (SIGFPE, quack-kernels 0.6.5).
        and x.shape[0] > 0
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
    tile_m, tile_n = _tile_shape(m, n, _num_sms(x.device.index))
    _swiglu_epilogue_mod().gemm(
        x,
        weight,
        None,
        None,
        epi_args={"mAuxOut": out},
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=True,
        concat_layout=("B",),
    )
    return out
