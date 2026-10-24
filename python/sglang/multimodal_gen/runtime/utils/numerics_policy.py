# SPDX-License-Identifier: Apache-2.0
"""Explicit control of the two approximate numerics PyTorch enables by default.

``torch.backends.cudnn.allow_tf32`` truncates fp32 convolution inputs to 10
mantissa bits, and
``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` lets a
bf16 GEMM accumulate split-K partials below fp32. Both change what is
computed, so both sit at the ``high`` tier of ``QUALITY_LEVELS`` while the
runtime serves ``exact`` and ``lossless`` requests through the same process.

They cannot follow the request: they are process-global and a server mixes
tiers across concurrent batches. The runtime therefore sets them once per
worker from the server args and states the result, so a tier's numerical
contract is auditable instead of inherited from whichever PyTorch build is
installed.

The defaults keep PyTorch's own values. Turning TF32 off makes fp32
convolutions -- the VAE decoders that default to ``vae_precision="fp32"`` --
run on fp32 CUDA cores rather than tensor cores, so flipping the default would
slow the exact tier down. A deployment that needs the strict contract sets
both to false and pays that cost knowingly.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def apply_numerics_policy(
    *, allow_cudnn_tf32: bool, allow_bf16_reduced_precision_reduction: bool
) -> None:
    """Set both approximate-numerics switches and log what this worker runs."""
    torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        allow_bf16_reduced_precision_reduction
    )
    approximate = [
        name
        for name, enabled in (
            ("cudnn TF32", allow_cudnn_tf32),
            (
                "bf16 reduced-precision reduction",
                allow_bf16_reduced_precision_reduction,
            ),
        )
        if enabled
    ]
    if approximate:
        logger.info_once(
            "Numerics policy: %s enabled. These lower precision below the "
            "reference, so quality='exact' is bit-reproducible on this server "
            "but not free of approximate library defaults. Pass "
            "--allow-cudnn-tf32 false --allow-bf16-reduced-precision-reduction "
            "false for the strict contract (slower fp32 convolutions).",
            " and ".join(approximate),
        )
    else:
        logger.info_once(
            "Numerics policy: cudnn TF32 and bf16 reduced-precision reduction "
            "are both disabled; fp32 and bf16 math keeps reference precision."
        )
