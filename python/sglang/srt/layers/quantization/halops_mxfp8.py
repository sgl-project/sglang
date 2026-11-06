# SPDX-License-Identifier: Apache-2.0

"""HALOps MXFP8 (group-32) dense-GEMM weight helpers.

The load-time half of the HALOps dense GEMM: the optional-package loader, the
CUTLASS SfAtom 128x4 scale layout, and the online weight quantizer that feeds it.
"""

from functools import lru_cache
from typing import Tuple

import torch

HALOPS_MODE = "w8a8-mxfp8"

# HALOps w8a8 needs K % 128 == 0.
HALOPS_K_ALIGN = 128

# Row alignment of the CUTLASS SfAtom 128x4 scale layout.
HALOPS_SFATOM_ROW_ALIGN = 128

# HALOps needs M align to 256 when use bias.
HALOPS_BIAS_M_ALIGN = 256


@lru_cache(maxsize=1)
def load_halops_gemm():
    """Return HALOps' ``(dense_gemm, group_wise_quant)``.

    Imported lazily so the wheel is only required once a layer actually needs it,
    and raises with an install hint rather than a bare ImportError so a
    ``--fp8-gemm-backend=halops`` run fails at load, not at first token.
    """
    try:
        from halops import dense_gemm, group_wise_quant
    except ImportError as exc:
        raise RuntimeError(
            "HALOps GEMM backend requested, but the `halops` package is not installed. "
            "The official Docker image ships it; outside Docker install it with: "
            "pip install halops==0.1.0 "
            "-f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/"
            "or build from source"
        ) from exc
    return dense_gemm, group_wise_quant


def weight_supported(*, n: int, k: int) -> bool:
    return k % HALOPS_K_ALIGN == 0 and n % HALOPS_SFATOM_ROW_ALIGN == 0


def swizzle_scale_to_sfatom(scale_u8: torch.Tensor) -> torch.Tensor:
    """Row-major UE8M0 ``[M, K // 32]`` -> the CUTLASS SfAtom 128x4 layout.

    Byte-identical to ``group_wise_quant(use_swizzle=True)`` (verified against it),
    so a scale that is already quantized still reaches ``dense_gemm``. Needs
    ``M % 128 == 0`` and ``K % 128 == 0``.
    """
    m, k_groups = scale_u8.shape
    return (
        scale_u8.contiguous()
        .view(m // 128, 128, k_groups // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(m // 128, k_groups // 4, 4, 32, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(m, k_groups)
        .contiguous()
    )


def quantize_weight(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a bf16 ``[N, K]`` weight with HALOps' ``group_wise_quant``.

    Returns ``(fp8 weight, SfAtom scale, row-major scale)``. SfAtom feeds
    ``dense_gemm``'s ``sfb``; row-major stays in ``weight_scale_inv`` for raw
    readers and doubles as the FlashInfer fallback scale.

    Requires ``N % 128 == 0`` and ``K % 128 == 0`` -- ``weight_supported`` gates
    the only caller on exactly that, because the SfAtom 128-row atoms and this
    swizzle both need it and apply() cannot slice padded output columns.

    One quantization pass: ``use_swizzle=False`` yields the row-major scale, and
    the SfAtom layout is the same bytes permuted, so swizzle it here rather than
    asking HALOps to quantize the weight a second time (which would also allocate
    and throw away a second full fp8 tensor). ``swizzle_scale_to_sfatom`` is
    verified byte-identical to ``group_wise_quant(use_swizzle=True)``, and feeding
    either one to ``dense_gemm`` gives bit-identical output.
    """
    _, group_wise_quant = load_halops_gemm()
    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)
    fp8_weight, sf_rowmajor = group_wise_quant(
        weight.contiguous(), mode=HALOPS_MODE, use_swizzle=False
    )
    sf_sfatom = swizzle_scale_to_sfatom(sf_rowmajor.view(torch.uint8))
    return fp8_weight, sf_sfatom, sf_rowmajor
