# SPDX-License-Identifier: Apache-2.0
"""Signature-cached launches for small Triton kernels.

A small kernel's wrapper costs more host time than its GPU time: operand checks,
stride math, the device guard and Triton's JIT dispatch take ~28 us per call against
a ~13 us cached launch (rmsnorm_scale, Z-Image 512^2, H200). All of it is fixed by
the operands' signature, so ``SignatureLaunch`` keeps Triton's compiled launcher and
the derived scalars per signature and relaunches with the same kernel, grid and
arguments: results are bit-identical.

``SGLANG_DIFFUSION_TRITON_FAST_LAUNCH=0`` turns it off.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Hashable

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get(
    "SGLANG_DIFFUSION_TRITON_FAST_LAUNCH", "1"
).strip().lower() not in {"0", "false", "off", "no"}


def launch_signature(*operands: Any) -> tuple | None:
    """Everything a compiled launch depends on, or None when the fast path is off.

    Pointer alignment is included because Triton specializes the compiled kernel on
    ``data_ptr() % 16``. A signature is recorded only after the wrapper validated its
    operands, so a hit needs no revalidation. Under torch.compile the wrapper must stay
    traceable, so no signature is taken there (checked first: Dynamo would otherwise
    guard on ``_ENABLED``).
    """
    if torch.compiler.is_compiling() or not _ENABLED:
        return None
    return tuple(
        (
            (op.shape, op.stride(), op.dtype, op.device, op.data_ptr() % 16)
            if isinstance(op, torch.Tensor)
            else op
        )
        for op in operands
    )


class SignatureLaunch:
    """Compiled launches of one Triton kernel, keyed by ``launch_signature``."""

    def __init__(self) -> None:
        self._plans: dict[Hashable, tuple[Callable[..., None], int]] = {}
        # Whether this kernel's launcher reproduces the JIT launch; None until checked.
        self._exact: bool | None = None

    def get(self, signature: tuple | None) -> Callable[..., None] | None:
        """Launch for ``signature``, called with the kernel's leading tensor arguments."""
        if signature is None:
            return None
        plan = self._plans.get(signature)
        if plan is None or torch.cuda.current_device() != plan[1]:
            return None
        return plan[0]

    def record(
        self,
        signature: tuple | None,
        compiled: Any,
        grid: tuple[int, ...],
        args: tuple,
    ) -> None:
        """Keep ``compiled``'s launcher for ``signature``.

        ``args`` are all kernel parameters in order, as the JIT launch just received
        them; the leading tensors are the per-call operands, ``args[0]`` the output.
        Call under the operands' device. The first record of each kernel checks the
        launcher's calling convention by relaunching into a scratch output, which
        syncs, so that check never runs during graph capture.
        """
        global _ENABLED
        if signature is None or signature in self._plans or compiled is None:
            return
        if self._exact is False:
            return
        num_tensors = next(
            (i for i, arg in enumerate(args) if not isinstance(arg, torch.Tensor)),
            len(args),
        )
        scalars = args[num_tensors:]
        try:
            launcher = compiled[tuple(grid) + (1,) * (3 - len(grid))]
            if self._exact is None:
                if torch.cuda.is_current_stream_capturing():
                    return
                scratch = torch.empty_like(args[0])
                launcher(scratch, *args[1:])
                self._exact = torch.equal(
                    scratch.view(torch.uint8), args[0].view(torch.uint8)
                )
        except torch.cuda.OutOfMemoryError:
            return
        except (AttributeError, TypeError) as exc:  # Triton's launcher API moved
            logger.warning(
                "Triton fast launch disabled (%s: %s); using the JIT path",
                type(exc).__name__,
                exc,
            )
            _ENABLED = False
            return
        if not self._exact:
            return

        def launch(*tensors: torch.Tensor) -> None:
            launcher(*tensors, *scalars)

        self._plans[signature] = (launch, torch.cuda.current_device())
