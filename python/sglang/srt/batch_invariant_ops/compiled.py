# SPDX-License-Identifier: Apache-2.0
"""Keep deterministic linear arithmetic opaque to Inductor lowering."""

import torch
import torch.nn.functional as F


@torch.library.custom_op("sglang::batch_invariant_linear", mutates_args=())
def batch_invariant_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Run the same CUDA dispatcher overrides used by uncompiled prefill.

    Lowering aten.linear directly lets Inductor select or fuse a GEMM instead
    of using the registered batch-invariant CUDA implementations.
    """
    return F.linear(x, weight, bias)


@batch_invariant_linear.register_fake
def _batch_invariant_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
