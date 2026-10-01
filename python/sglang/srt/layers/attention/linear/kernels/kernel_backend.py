from abc import ABC, abstractmethod

import torch


class LinearAttnKernelBase(ABC):
    """Abstract base class for linear attention kernel implementations.

    Each concrete implementation wraps a specific kernel (Triton, CuTe DSL, etc.)
    and provides decode/extend/target_verify methods with a unified interface.
    """

    uses_state_checkpoints: bool = False
    supports_fused_chain_verify: bool = False
    # Opt in only when target-verify kernels honor non-unit token strides.
    supports_strided_target_verify_qkv: bool = False

    # True when extend() honors the fp32 track snapshot (track_state /
    # track_chunk_idx), natively or by routing tracked batches to a kernel
    # that does. KDAAttnBackend asserts this before allocating the snapshot
    # buffer: a kernel that silently ignores those arguments leaves the buffer
    # unwritten and corrupts prefix-cache restores. Kernels that reject
    # tracked batches loudly (NotImplementedError) keep the default False.
    supports_track_state_snapshot: bool = False

    # True when decode() implements the bounded ("safe") gate, i.e. honors a
    # finite ``lower_bound`` (Kimi-K3: gate_lower_bound=-5.0). The dispatcher
    # refuses bounded-gate decode on kernels that keep the default False
    # instead of letting them silently run the unbounded softplus gate.
    supports_bounded_gate_decode: bool = False

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
