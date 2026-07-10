from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


def unwrap_direct_write_out(
    out: Optional[torch.Tensor], *, expected_shape: Tuple[int, ...]
) -> Optional[torch.Tensor]:
    """Validate and unwrap a caller-supplied direct-write output buffer.

    Returns the ``[T, Hv, V]`` view the kernel writes into, or None when the
    caller did not supply a buffer. A mis-shaped buffer must fail loud here —
    silently falling back would hide a broken direct-write contract.
    """
    if out is None:
        return None
    assert (
        out.shape == expected_shape
    ), f"direct-write out buffer {tuple(out.shape)} != expected {expected_shape}"
    return out.squeeze(0)


class LinearAttnKernelBase(ABC):
    """Abstract base class for linear attention kernel implementations.

    Each concrete implementation wraps a specific kernel (Triton, CuTe DSL, etc.)
    and provides decode/extend/target_verify methods with a unified interface.
    """

    # extend()'s `g` argument: False = log-space decay from fused_gdn_gating;
    # True = exp(g) (multiplicative alpha) computed by fused_gdn_gating(exp_gate=True).
    extend_expects_exp_gate: bool = False

    # Whether this kernel implements extend_prenormed() — the glue-kernel entry
    # point that receives ALREADY-L2-NORMALIZED q/k. Kernels whose extend()
    # normalizes internally must leave this False so the caller keeps the
    # unfused split+gating+norm chain.
    supports_prenormed_extend: bool = False

    def build_extend_prep(
        self,
        *,
        head_k_dim: int,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        ssm_states: torch.Tensor,
        total_seq_len: int,
    ) -> Optional[tuple]:
        """Layer-invariant extend metadata, built once per forward and passed
        back to every per-layer extend() via ``prep=`` (opaque, backend-specific
        contents). None = this kernel takes no prep; extend() must accept and
        ignore ``prep=None``."""
        return None

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor: ...

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
        # Optional perf hints; implementations MAY ignore them (ignoring is
        # behavior-identical): out = direct-write output buffer, prep = this
        # kernel's build_extend_prep result, no_prefix = no request in the
        # batch has a prefix (initial state is all-zero).
        out: Optional[torch.Tensor] = None,
        prep: Optional[tuple] = None,
        no_prefix: bool = False,
        **kwargs,
    ) -> tuple: ...

    def extend_prenormed(
        self,
        q_normed: torch.Tensor,
        k_normed: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """extend() variant for glue-kernel callers: q/k arrive L2-normalized
        and ``g`` in the form declared by :attr:`extend_expects_exp_gate`.
        Implementations must NOT normalize q/k or transform g again. Only
        kernels declaring ``supports_prenormed_extend = True`` implement this."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support extend_prenormed"
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support target_verify"
        )
