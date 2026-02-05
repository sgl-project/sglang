from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MambaSelectiveStateUpdateBackend(Enum):
    """Backend for selective-state-update kernel."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"


_mamba_selective_state_fn = None
_mamba_selective_state_update_backend = None


def initialize_mamba_selective_state_update_backend(server_args: ServerArgs) -> None:
    """
    Initialize the selective_state_update backend based on server configuration.
    This should be called once during scheduler initialization.

    Args:
        server_args: Server arguments containing mamba_backend setting.

    Raises:
        ValueError: If requested backend is not available.
    """
    global _mamba_selective_state_fn
    global _mamba_selective_state_update_backend

    requested = server_args.mamba_backend or "triton"

    if requested == "flashinfer":
        try:
            from flashinfer.mamba import selective_state_update  # noqa: F401

            _mamba_selective_state_fn = selective_state_update
            _mamba_selective_state_update_backend = (
                MambaSelectiveStateUpdateBackend.FLASHINFER
            )
            logger.info("Mamba selective_state_update backend initialized: flashinfer")
        except ImportError:
            raise ValueError(
                "FlashInfer mamba backend requested but flashinfer.mamba module "
                "is not available. Install FlashInfer with mamba support or use "
                "--mamba-backend triton"
            )
    else:
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        _mamba_selective_state_fn = selective_state_update
        _mamba_selective_state_update_backend = MambaSelectiveStateUpdateBackend.TRITON
        logger.info("Mamba selective_state_update backend initialized: triton")


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
    out: torch.Tensor | None = None,
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
) -> None:
    """Dispatch selective-state-update to the configured backend.

    This function provides a unified interface for both Triton and FlashInfer
    implementations of the selective-state-update kernel.

    Note: For speculative decoding (which requires additional parameters like
    intermediate_states_buffer), use the triton kernel directly via
    `from sglang.srt.layers.attention.mamba.ops.mamba_ssm import selective_state_update`

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
    assert (
        _mamba_selective_state_fn is not None
    ), "Mamba selective_state_update function not initialized. Call initialize_mamba_selective_state_update_backend() first."

    
    # TODO smor before merger- fix the kwargs so you won't need this call duplication
    # TODO smor before merger- add assert for retrieve_parent_token, if FI then it must be one, otherwise fail. Do it in server_args!!
    if (
        _mamba_selective_state_update_backend
        == MambaSelectiveStateUpdateBackend.FLASHINFER
    ):
        print(f"SMOR: FLASHINFER selective_state_update")
        cache_steps = 0 if cache_steps is None else cache_steps
        _mamba_selective_state_fn(
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
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            intermediate_state_indices=intermediate_state_indices,
        )
    else:
        print(f"SMOR: TRITON selective_state_update")
        _mamba_selective_state_fn(
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
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )

def _is_specdec_call(
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
) -> bool:
    return (
        (disable_state_update)
        or (intermediate_states_buffer is not None)
        or (cache_steps is not None)
        or (retrieve_parent_token is not None)
        or (intermediate_state_indices is not None)
    )
