"""
Utilities to manage the dequantization of weights.
"""

from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.mxfp8_block_convert import (
    dequant_mxfp8_2d_to_bf16,
)
from sglang.srt.utils import set_weight_attrs

NVFP4_BLOCK_SIZE = 16
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

# The checkpoint tensors each dequantized recipe needs, by name after the layer prefix
BF16_RECIPE_PARTS = {
    "MXFP8": ("weight", "weight_scale"),
    "W4A16_NVFP4": ("weight", "weight_scale", "weight_scale_2"),
}

# The name the MXFP8 block-scale rename gives `weight_scale` before load_weights
MXFP8_RENAMED_SCALE = "weight_scale_inv"


def copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


def dequantize_fp8(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    block_size: list[int],
    format_ue8m0: bool = False,
) -> torch.Tensor:
    """
    Dequantizes `w_q` to bfloat16.
    """
    if format_ue8m0:
        # TODO this is only needed for Blackwell
        w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])

    w_dequant = block_quant_dequant(
        w_q,
        w_s,
        block_size=block_size,
        dtype=torch.bfloat16,
    )

    return w_dequant


def dequantize_nvfp4(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_s2: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16,
    high_nibble_first: bool = False,
) -> torch.Tensor:
    """NVFP4 -> ``out_dtype``. ``w_q``: uint8 [..., out, in/2] packed e2m1
    (low nibble = even idx unless ``high_nibble_first``). ``w_s``: fp8 e4m3
    [..., out, in/16] per-block.
    ``w_s2``: optional fp32 per-tensor scalar that multiplies the per-block
    scale (ModelOpt / AMD Quark NVFP4)."""
    device = w_q.device
    *batch, out_dim, half_in = w_q.shape
    in_dim = half_in * 2

    low = (w_q & 0xF).to(torch.int64)
    high = (w_q >> 4).to(torch.int64)
    first, second = (high, low) if high_nibble_first else (low, high)
    lut = _FP4_E2M1_LUT.to(device=device, dtype=torch.float32)
    deq = torch.empty(*batch, out_dim, in_dim, dtype=torch.float32, device=device)
    deq[..., 0::2] = lut[first]
    deq[..., 1::2] = lut[second]

    scale = w_s.to(torch.float32)
    if w_s2 is not None:
        scale = scale * w_s2.to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)
    return (deq * scale).to(out_dtype)


def bf16_weight_of(recipe: str, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The BF16 weight of a layer from the PARTS its RECIPE stores."""
    if recipe == "MXFP8":
        return dequant_mxfp8_2d_to_bf16(parts["weight"], parts["weight_scale"])
    return dequantize_nvfp4(
        parts["weight"], parts["weight_scale"], parts["weight_scale_2"]
    )


def dequantize_layers_to_bf16(
    weights: Iterable[Tuple[str, torch.Tensor]],
    prefix: str,
    quant_algo_of: Callable[[str], Optional[str]],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """WEIGHTS with each MXFP8 or W4A16_NVFP4 layer under PREFIX as one BF16 `weight`.

    For a model that builds those layers in BF16. Every other tensor, biases
    included, passes through unchanged. A layer missing a part at the end raises.
    """
    waiting: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        layer, _, part = name.rpartition(".")
        recipe = quant_algo_of(layer) if name.startswith(prefix) else None
        if part == MXFP8_RENAMED_SCALE and recipe == "MXFP8":
            part = "weight_scale"
        if recipe not in BF16_RECIPE_PARTS or part not in BF16_RECIPE_PARTS[recipe]:
            yield name, tensor
            continue

        parts = waiting.setdefault(layer, {})
        parts[part] = tensor
        if len(parts) < len(BF16_RECIPE_PARTS[recipe]):
            continue
        del waiting[layer]
        yield f"{layer}.weight", bf16_weight_of(recipe, parts)

    if waiting:
        incomplete = [
            f"{layer} ({', '.join(parts)})" for layer, parts in waiting.items()
        ]
        raise ValueError(
            f"{len(waiting)} quantized layer(s) under {prefix!r} are missing parts, "
            f"e.g. {incomplete[:4]}; the checkpoint is incomplete."
        )
