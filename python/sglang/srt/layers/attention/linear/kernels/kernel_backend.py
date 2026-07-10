from abc import ABC, abstractmethod

import torch


class LinearAttnKernelBase(ABC):
    """Abstract base class for linear attention kernel implementations.

    Each concrete implementation wraps a specific kernel (Triton, CuTe DSL, etc.)
    and provides decode/extend/target_verify methods with a unified interface.
    """

    # Whether extend() accepts a hoisted layer-invariant prep tuple built once
    # per forward via build_extend_prep() (opaque, backend-specific contents).
    # Kernels that set this True must define build_extend_prep with the common
    # keyword signature (head_k_dim, query_start_loc, cache_indices, ssm_states,
    # total_seq_len).
    supports_extend_prep: bool = False

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
        **kwargs,
    ) -> tuple: ...

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
