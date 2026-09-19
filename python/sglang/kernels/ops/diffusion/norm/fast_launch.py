# SPDX-License-Identifier: Apache-2.0
"""Rust-backed launch path for hot Triton kernels.

At low resolution a diffusion step is a long tail of small kernels: Z-Image-Turbo
at 512x512 runs 2108 kernels averaging 26.6 us with 45.5 us of GPU idle behind
each one. The device is not the constraint there. Python and the Triton launcher
are, and describing a launch costs more than performing it.

Neither the kernel nor its argument layout changes between steps, only the
buffers do. So record one launch per signature, then rebind the pointers and
replay through ``cuLaunchKernel`` from Rust. Same kernel, same grid, same
arguments, same order, so results are bit-identical: this removes host latency
and nothing else.

Unlike a CUDA graph it survives a shape change -- a new signature records a new
entry rather than forcing a recapture of everything -- which matters for models
whose sequence length depends on the prompt.
"""

from __future__ import annotations

import torch

try:
    import sgl_launch

    _HAVE_RUST_LAUNCH = True
except ImportError:  # pragma: no cover - extension not built
    sgl_launch = None
    _HAVE_RUST_LAUNCH = False


def available() -> bool:
    return _HAVE_RUST_LAUNCH


class CachedLaunch:
    """One Triton kernel's launches, keyed by signature.

    ``record`` is called on the first sighting of a signature, after the
    ordinary Triton path has produced the result and the ``CompiledKernel``.
    ``lookup`` plus ``replay`` is the steady state and touches no dispatcher.
    """

    def __init__(self) -> None:
        self._chains: dict[tuple, object] = {}

    def lookup(self, signature: tuple):
        return self._chains.get(signature)

    def record(
        self,
        signature: tuple,
        compiled,
        grid: tuple[int, int, int],
        pointer_args: list[int],
        scalar_args: list[int],
    ) -> None:
        if not _HAVE_RUST_LAUNCH or signature in self._chains:
            return
        chain = sgl_launch.LaunchChain()
        threads = compiled.metadata.num_warps * compiled.metadata.warp_size
        # Triton appends scratch pointers to every kernel's parameter list. They
        # are null when unused, but the slots exist and the driver rejects a
        # short argument array.
        scratch = [0, 0]
        chain.record(
            compiled.function,
            grid,
            (threads, 1, 1),
            compiled.metadata.shared,
            list(pointer_args) + list(scalar_args) + scratch,
        )
        self._chains[signature] = chain

    def replay(self, chain, pointer_args: list[int]) -> None:
        for index, value in enumerate(pointer_args):
            chain.rebind(0, index, value)
        chain.replay(torch.cuda.current_stream().cuda_stream)
