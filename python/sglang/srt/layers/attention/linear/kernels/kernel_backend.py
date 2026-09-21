from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Literal

import torch

LinearAttnKind = Literal["gdn", "kda"]
LinearAttnPhase = Literal["decode", "extend"]
LinearAttnKernelFactory = Callable[
    ["LinearAttnKernelBase"], "LinearAttnKernelBase"
]


class LinearAttnKernelBase(ABC):
    """Abstract base class for linear attention kernel implementations.

    Each concrete implementation wraps a specific kernel (Triton, CuTe DSL, etc.)
    and provides decode/extend/target_verify methods with a unified interface.
    """

    uses_state_checkpoints: bool = False
    supports_fused_chain_verify: bool = False

    # True when extend() honors the fp32 track snapshot (track_state /
    # track_chunk_idx), natively or by routing tracked batches to a kernel
    # that does. KDAAttnBackend asserts this before allocating the snapshot
    # buffer: a kernel that silently ignores those arguments leaves the buffer
    # unwritten and corrupts prefix-cache restores. Kernels that reject
    # tracked batches loudly (NotImplementedError) keep the default False.
    supports_track_state_snapshot: bool = False

    _oot_kernel_registry: ClassVar[
        dict[
            str,
            dict[tuple[LinearAttnKind, LinearAttnPhase], LinearAttnKernelFactory],
        ]
    ] = {}

    @classmethod
    def register_oot_kernel(
        cls,
        kind: LinearAttnKind,
        phase: LinearAttnPhase,
        factory: LinearAttnKernelFactory,
        platform_key: str,
    ) -> None:
        """Register an OOT factory for one linear-attention kernel phase."""
        registry = cls._oot_kernel_registry.setdefault(platform_key, {})
        registry[(kind, phase)] = factory

    @classmethod
    def resolve_oot_kernel(
        cls,
        kind: LinearAttnKind,
        phase: LinearAttnPhase,
        fallback: "LinearAttnKernelBase",
    ) -> "LinearAttnKernelBase":
        """Return the active OOT kernel or the selected in-tree fallback."""
        from sglang.srt.platforms import current_platform

        if not current_platform.is_out_of_tree():
            return fallback
        factory = cls._oot_kernel_registry.get(
            current_platform.get_dispatch_key_name(), {}
        ).get((kind, phase))
        return fallback if factory is None else factory(fallback)

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
