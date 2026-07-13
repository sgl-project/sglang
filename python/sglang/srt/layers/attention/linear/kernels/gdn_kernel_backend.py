from abc import abstractmethod
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def unwrap_direct_write_out(
    out: Optional[torch.Tensor], *, expected_shape: Tuple[int, ...]
) -> Optional[torch.Tensor]:
    """Validate and unwrap a caller-supplied direct-write output buffer."""
    if out is None:
        return None
    assert (
        out.shape == expected_shape
    ), f"direct-write out buffer {tuple(out.shape)} != expected {expected_shape}"
    return out.squeeze(0)


class GDNKernelBase(LinearAttnKernelBase):
    """GDN-only kernel contract.

    Public ``extend`` callers always pass raw q/k and log-space decay ``g``.
    Backend-specific representations such as FlashInfer's multiplicative
    ``alpha = exp(g)`` remain private to the concrete implementation.
    """

    @abstractmethod
    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        no_prefix: bool = False,
        **kwargs,
    ) -> tuple:
        """Run GDN extend from raw q/k and log-space decay ``g``."""
        ...
