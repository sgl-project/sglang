"""Reversible NVFP4 layout changes with one layer of additional storage.

The original parameter pointers and the decode graph pointers are both stable.
Forward transposes run in layer order into [scratch, raw[0], ..., raw[-2]];
inverse transposes run in reverse order, restoring every original byte.
All transforms must be outside graph capture and ordered on the forward stream.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _transpose_module(rows: int, cols: int):
    return load_jit(
        "sm120_nvfp4_weight_transpose",
        rows,
        cols,
        cuda_files=["gemm/sm120_nvfp4_weight_transpose.cuh"],
        cuda_wrappers=[("run", f"sglang::Nvfp4WeightTranspose<{rows},{cols}>::run")],
        extra_cuda_cflags=["-O3"],
    )


class Nvfp4LayoutCycle:
    """Own the two views of a fixed, serial set of [512,1024,1344] weights."""

    def __init__(self, weights):
        if not weights:
            raise ValueError("A layout cycle requires at least one expert weight")
        first = weights[0]
        if first.device.type != "cuda":
            raise ValueError("The layout cycle requires CUDA weights")
        for weight in weights:
            if (
                weight.shape != (512, 1024, 1344)
                or weight.dtype != torch.uint8
                or not weight.is_contiguous()
                or weight.device != first.device
            ):
                raise ValueError("Incompatible NVFP4 FC2 weight layout")
        if len({weight.data_ptr() for weight in weights}) != len(weights):
            raise ValueError("Expert weights in a layout cycle must not alias")
        self.original = tuple(weights)
        self.scratch = torch.empty_like(first)
        self.pair = tuple(
            weight.view(512, 1344, 1024)
            for weight in (self.scratch, *self.original[:-1])
        )
        self.is_pair = False
        self.switches = 0
        # Compile before any graph capture can reach the state transition.
        self._forward = _transpose_module(1024, 1344)
        self._inverse = _transpose_module(1344, 1024)

    def ensure(self, pair: bool):
        if pair == self.is_pair:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 layout transitions must run before graph capture")
        if pair:
            for original, paired in zip(self.original, self.pair):
                self._forward.run(original, paired)
        else:
            for original, paired in reversed(tuple(zip(self.original, self.pair))):
                self._inverse.run(paired, original)
        self.is_pair = pair
        self.switches += 1
