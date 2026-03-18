from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MambaSSUBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used for logging."""

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based selective-state-update backend."""

    def __init__(self) -> None:
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based selective-state-update backend."""

    def __init__(self) -> None:
        from flashinfer.mamba import selective_state_update

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        if retrieve_parent_token is not None:
            raise ValueError(
                "FlashInfer backend does not support retrieve_parent_token. "
                "Use --mamba-backend triton for EAGLE tree attention."
            )
        # FlashInfer expects cache_steps as an int (0 when unused).
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=0 if cache_steps is None else cache_steps,
            intermediate_state_indices=intermediate_state_indices,
        )


_BACKEND_REGISTRY: dict[str, type[MambaSSUBackend]] = {
    "triton": TritonSSUBackend,
    "flashinfer": FlashInferSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_selective_state_update_backend(server_args: ServerArgs) -> None:
    """Instantiate the selective-state-update backend from server config.

    This should be called once during scheduler initialization.

    Args:
        server_args: Server arguments containing ``mamba_backend`` setting.

    Raises:
        ValueError: If the requested backend is unavailable or cannot be imported.
    """
    global _mamba_ssu_backend

    requested = server_args.mamba_backend or "triton"

    backend_cls = _BACKEND_REGISTRY.get(requested)
    if backend_cls is None:
        raise ValueError(
            f"Unknown mamba backend '{requested}'. "
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
        )

    try:
        _mamba_ssu_backend = backend_cls()
    except ImportError:
        raise ValueError(
            f"Mamba backend '{requested}' requested but its dependencies are not "
            f"available. Install the required package or use a different "
            f"--mamba-backend value."
        )

    logger.info(
        "Mamba selective_state_update backend initialized: %s",
        _mamba_ssu_backend.name,
    )


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    out: torch.Tensor | None = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    cache_steps: int | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
) -> None:
    """Dispatch selective-state-update to the configured backend.

    This function provides a unified interface regardless of the underlying
    backend. Backend-specific argument adaptation is handled inside each
    :class:`MambaSSUBackend` subclass.

    Args:
        state: SSM state tensor (batch, nheads, dim, dstate)
        x: Input tensor
        dt: Delta time tensor
        A: A matrix
        B: B matrix
        C: C matrix
        D: Optional D vector
        z: Optional z tensor for gating
        dt_bias: Optional dt bias
        dt_softplus: Whether to apply softplus to dt
        state_batch_indices: Optional batch indices for state
        out: Preallocated output tensor (in-place updated)
        disable_state_update: If True, don't write back to state (for speculative verify)
        intermediate_states_buffer: Buffer to cache intermediate states
        cache_steps: Total number of steps in the buffer
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.
            If provided, uses these indices instead of state_batch_indices for the buffer.
    """
    assert _mamba_ssu_backend is not None, (
        "Mamba selective_state_update backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
    )

    _mamba_ssu_backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        pad_slot_id=pad_slot_id,
        out=out,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        intermediate_state_indices=intermediate_state_indices,
    )
