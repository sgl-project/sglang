from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import flashinfer
import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_packbit_module() -> Module:
    flashinfer_include_path = str(
        (pathlib.Path(flashinfer.__file__).parent / "data" / "include").resolve()
    )
    return load_jit(
        "packbit",
        cuda_files=["speculative/packbit.cuh"],
        cuda_wrappers=[
            ("segment_packbits", "segment_packbits"),
        ],
        extra_include_paths=[flashinfer_include_path],
    )


@register_custom_op(
    op_name="segment_packbits_out",
    mutates_args=["y"],
)
def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> None:
    """
    Pack boolean bits into bytes, segment by segment (little-endian bit order).

    Args:
        x:             [sum(input_indptr)] bool — input bits
        input_indptr:  [batch_size + 1] int32 — segment start offsets for input
        output_indptr: [batch_size + 1] int32 — segment start offsets for output
        y:             [sum(output_indptr)] uint8 — packed output (mutated in-place)
        batch_size:    number of segments
    """
    module = _jit_packbit_module()
    module.segment_packbits(x, input_indptr, output_indptr, y, batch_size)
