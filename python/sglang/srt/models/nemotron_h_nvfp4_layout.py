from __future__ import annotations

import torch

from sglang.kernels.ops.gemm.sm120_nvfp4_moe import _transpose_module


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
        self.scratch = None
        self.pair = ()
        self.is_pair = False
        # Compile before any graph capture can reach the state transition.
        self._forward = _transpose_module(1024, 1344)
        self._inverse = _transpose_module(1344, 1024)

    def prepare_workspace(self):
        if self.scratch is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 workspace must be prepared before graph capture")
        # Allocate from runtime headroom, not before Mamba/KV pool sizing, where it
        # would count as weights and could leave the state pool under one request.
        self.scratch = torch.empty_like(self.original[0])
        self.pair = tuple(
            weight.view(512, 1344, 1024)
            for weight in (self.scratch, *self.original[:-1])
        )

    def ensure(self, pair: bool):
        if pair == self.is_pair:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 layout transitions must run before graph capture")
        if pair:
            self.prepare_workspace()
            for original, paired in zip(self.original, self.pair):
                self._forward.run(original, paired)
        else:
            for original, paired in reversed(tuple(zip(self.original, self.pair))):
                self._inverse.run(paired, original)
        self.is_pair = pair
