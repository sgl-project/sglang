from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactMambaPrefillCheckpoints:
    """Selected SSM chunk-boundary states in radix-track destination order."""

    states: torch.Tensor


class MambaSSUBackend(ABC):
    prefill_requires_chunk_metadata = False

    def prefill_metadata_chunk_size(self, chunk_size: int) -> int:
        """Physical chunk size used to construct this backend's metadata."""
        return chunk_size

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

    def chunk_scan_combined(self, *args, **kwargs):
        """Run the prefill SSDCombined operation for this backend."""
        raise NotImplementedError(
            f"Mamba backend '{self.name}' does not implement SSDCombined prefill"
        )


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based selective-state-update backend."""

    def __init__(
        self,
        *,
        enable_stochastic_rounding: bool = False,
        cache_philox_rounds: int = 0,
    ) -> None:
        from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import (
            selective_state_update,
        )
        from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (
            mamba_chunk_scan_combined,
        )

        self._kernel = selective_state_update
        self._prefill_kernel = mamba_chunk_scan_combined
        self._enable_stochastic_rounding = enable_stochastic_rounding
        self._cache_philox_rounds = cache_philox_rounds

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
            enable_stochastic_rounding=self._enable_stochastic_rounding,
            cache_philox_rounds=self._cache_philox_rounds,
        )

    def chunk_scan_combined(self, *args, **kwargs):
        return self._prefill_kernel(*args, **kwargs)


class FlashInferSSUBackend(MambaSSUBackend):
    """Established Triton prefill and FlashInfer selective-state-update decode."""

    def __init__(
        self,
        *,
        enable_stochastic_rounding: bool = False,
        cache_philox_rounds: int = 0,
    ) -> None:
        from flashinfer.mamba import selective_state_update

        from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (
            mamba_chunk_scan_combined,
        )

        self._kernel = selective_state_update
        self._enable_stochastic_rounding = enable_stochastic_rounding
        self._cache_philox_rounds = cache_philox_rounds
        # Preserve the established `flashinfer` option: FlashInfer decode with
        # Triton prefill. `flashinfer_ssd` and `cake` opt into strict public
        # SSDCombined prefill below.
        self._prefill_kernel = mamba_chunk_scan_combined
        self._prefill_backend = None
        self._prefill_runners = {}
        self._zero_initial_states = {}

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
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self._enable_stochastic_rounding
            else None
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
            rand_seed=rand_seed,
            philox_rounds=self._cache_philox_rounds or 10,
        )

    @staticmethod
    def _pad_sequence(tensor: torch.Tensor, pad: int, value: float) -> torch.Tensor:
        if pad == 0:
            return tensor
        padding = torch.full(
            (tensor.shape[0], pad, *tensor.shape[2:]),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, padding), dim=1)

    def _get_prefill_runner(
        self,
        *,
        x: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor | None,
        z: torch.Tensor | None,
        initial_states: torch.Tensor,
        seq_idx: torch.Tensor,
        chunk_size: int,
    ):
        from flashinfer.mamba import SSDCombined

        key = (
            x.device.index,
            chunk_size,
            x.shape[2],
            x.shape[3],
            B.shape[2],
            B.shape[3],
            x.dtype,
            initial_states.dtype,
            D is not None,
            D is not None and D.ndim == 2,
            z is not None,
            seq_idx.dtype,
        )
        runner = self._prefill_runners.get(key)
        if runner is None:
            runner = SSDCombined(
                chunk_size=chunk_size,
                nheads=x.shape[2],
                headdim=x.shape[3],
                dstate=B.shape[3],
                ngroups=B.shape[2],
                io_dtype=x.dtype,
                state_dtype=initial_states.dtype,
                has_d=D is not None,
                d_has_hdim=D is not None and D.ndim == 2,
                has_initial_states=True,
                has_varlen=True,
                has_z=z is not None,
                seq_idx_dtype=seq_idx.dtype,
                backend=self._prefill_backend,
            )
            self._prefill_runners[key] = runner
        return runner

    def _get_zero_initial_states(
        self,
        *,
        x: torch.Tensor,
        B: torch.Tensor,
        num_sequences: int,
        state_dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (
            x.device.index,
            num_sequences,
            x.shape[2],
            x.shape[3],
            B.shape[3],
            state_dtype,
        )
        states = self._zero_initial_states.get(key)
        if states is None:
            states = torch.zeros(
                (num_sequences, x.shape[2], x.shape[3], B.shape[3]),
                dtype=state_dtype,
                device=x.device,
            )
            self._zero_initial_states[key] = states
        return states

    def _run_compact_checkpoints(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None,
        z: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        dt_softplus: bool,
        dt_limit,
        initial_states: torch.Tensor,
        checkpoint_seq_indices: tuple[int, ...],
        checkpoint_seq_starts: tuple[int, ...],
        checkpoint_lengths: tuple[int, ...],
        chunk_size: int,
    ) -> CompactMambaPrefillCheckpoints | None:
        """Re-run only prefixes needed for non-final radix checkpoints.

        SSDCombined does not expose Triton's potentially multi-GB dense ``h``
        tensor.  Radix tracking needs at most one earlier chunk boundary per
        selected request, so pack just those prefixes and return their final
        states in destination order.  This stays on the selected SSD backend;
        it is not a Triton fallback.
        """

        if not checkpoint_seq_indices:
            return None
        if not (
            len(checkpoint_seq_indices)
            == len(checkpoint_seq_starts)
            == len(checkpoint_lengths)
        ):
            raise ValueError("compact checkpoint metadata lengths must match")
        if any(length <= 0 or length % chunk_size for length in checkpoint_lengths):
            raise ValueError(
                "compact checkpoint lengths must be positive chunk multiples"
            )

        def select_prefixes(value: torch.Tensor | None):
            if value is None:
                return None
            return torch.cat(
                tuple(
                    value[:, start : start + length]
                    for start, length in zip(
                        checkpoint_seq_starts, checkpoint_lengths, strict=True
                    )
                ),
                dim=1,
            )

        checkpoint_x = select_prefixes(x)
        checkpoint_dt = select_prefixes(dt)
        checkpoint_B = select_prefixes(B)
        checkpoint_C = select_prefixes(C)
        checkpoint_z = select_prefixes(z)
        assert checkpoint_x is not None and checkpoint_dt is not None
        assert checkpoint_B is not None and checkpoint_C is not None
        checkpoint_initial = initial_states[
            torch.tensor(
                checkpoint_seq_indices,
                dtype=torch.long,
                device=initial_states.device,
            )
        ]
        num_checkpoints = len(checkpoint_seq_indices)
        total_tokens = sum(checkpoint_lengths)
        checkpoint_seq_idx = torch.repeat_interleave(
            torch.arange(
                num_checkpoints,
                dtype=torch.int32,
                device=x.device,
            ),
            torch.tensor(
                checkpoint_lengths,
                dtype=torch.long,
                device=x.device,
            ),
            output_size=total_tokens,
        ).unsqueeze(0)
        num_chunks = total_tokens // chunk_size
        checkpoint_chunk_indices = torch.arange(
            num_chunks, dtype=torch.int32, device=x.device
        )
        checkpoint_chunk_offsets = torch.zeros_like(checkpoint_chunk_indices)
        runner = self._get_prefill_runner(
            x=checkpoint_x,
            B=checkpoint_B,
            D=D,
            z=checkpoint_z,
            initial_states=checkpoint_initial,
            seq_idx=checkpoint_seq_idx,
            chunk_size=chunk_size,
        )
        _, checkpoint_states = runner.run(
            checkpoint_x,
            checkpoint_dt,
            A,
            checkpoint_B,
            checkpoint_C,
            D=D,
            z=checkpoint_z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=checkpoint_initial,
            seq_idx=checkpoint_seq_idx,
            chunk_indices=checkpoint_chunk_indices,
            chunk_offsets=checkpoint_chunk_offsets,
            out=None,
            return_final_states=True,
        )
        # SSDCombined runners reuse final-state workspace.  Clone the selected
        # states before the main full-input invocation overwrites that storage.
        return CompactMambaPrefillCheckpoints(checkpoint_states.clone())

    def chunk_scan_combined(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        dt_softplus: bool = False,
        dt_limit=(0.0, float("inf")),
        out: torch.Tensor | None = None,
        return_final_states: bool = False,
        return_varlen_states: bool = False,
        return_intermediate_states: bool = False,
        state_dtype=None,
        checkpoint_seq_indices: tuple[int, ...] = (),
        checkpoint_seq_starts: tuple[int, ...] = (),
        checkpoint_lengths: tuple[int, ...] = (),
    ):
        if self._prefill_backend is None:
            return self._prefill_kernel(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                cu_seqlens=cu_seqlens,
                dt_softplus=dt_softplus,
                dt_limit=dt_limit,
                out=out,
                return_final_states=return_final_states,
                return_varlen_states=return_varlen_states,
                return_intermediate_states=return_intermediate_states,
                state_dtype=state_dtype,
            )
        # SGLang only consumes the packed-varlen form below. Refuse other
        # combinations instead of silently changing their return semantics.
        if (
            not return_varlen_states
            or return_final_states
            or not return_intermediate_states
        ):
            raise ValueError(
                f"Mamba backend '{self.name}' only supports SGLang's "
                "return_intermediate_states=True, return_varlen_states=True, "
                "return_final_states=False prefill form"
            )
        if cu_seqlens is None or seq_idx is None:
            raise ValueError(
                f"Mamba backend '{self.name}' requires packed sequence metadata"
            )
        if chunk_indices is None or chunk_offsets is None:
            raise ValueError(
                f"Mamba backend '{self.name}' requires logical chunk metadata"
            )
        if x.shape[0] != 1:
            raise ValueError(
                f"Mamba backend '{self.name}' requires packed batch=1 input"
            )
        if float(dt_limit[0]) != 0.0:
            raise ValueError(
                f"Mamba backend '{self.name}' requires dt_limit[0]=0 so "
                "tail padding is an exact identity transition"
            )
        if out is None:
            raise ValueError(
                f"Mamba backend '{self.name}' requires SGLang's caller-owned "
                "token-major output buffer"
            )
        if state_dtype is None:
            state_dtype = x.dtype
        num_sequences = len(cu_seqlens) - 1
        if initial_states is None:
            initial_states = self._get_zero_initial_states(
                x=x,
                B=B,
                num_sequences=num_sequences,
                state_dtype=state_dtype,
            )

        seqlen = x.shape[1]
        padded_seqlen = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
        pad = padded_seqlen - seqlen
        # Padding is an exact identity transition: softplus(-inf) == 0, and
        # zero x/B/C contributes no state or output. The final sequence id is
        # extended through the padding, so existing logical chunk metadata
        # remains valid.
        x_padded = self._pad_sequence(x, pad, 0.0)
        B_padded = self._pad_sequence(B, pad, 0.0)
        C_padded = self._pad_sequence(C, pad, 0.0)
        dt_padded = self._pad_sequence(dt, pad, -float("inf"))
        z_padded = self._pad_sequence(z, pad, 0.0) if z is not None else None
        if pad:
            seq_idx = torch.cat((seq_idx, seq_idx[:, -1:].expand(-1, pad)), dim=1)

        runner = self._get_prefill_runner(
            x=x_padded,
            B=B_padded,
            D=D,
            z=z_padded,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_size=chunk_size,
        )
        compact_checkpoints = self._run_compact_checkpoints(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=initial_states,
            checkpoint_seq_indices=checkpoint_seq_indices,
            checkpoint_seq_starts=checkpoint_seq_starts,
            checkpoint_lengths=checkpoint_lengths,
            chunk_size=chunk_size,
        )
        output, varlen_states = runner.run(
            x_padded,
            dt_padded,
            A,
            B_padded,
            C_padded,
            D=D,
            z=z_padded,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            out=None,
            return_final_states=True,
        )
        output = output[:, :seqlen]
        if out is not None:
            out.copy_(output)
        return compact_checkpoints, varlen_states


class FlashInferSSDCombinedSSUBackend(FlashInferSSUBackend):
    """FlashInfer CuTe SSDCombined prefill and selective-state-update decode."""

    prefill_requires_chunk_metadata = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._prefill_backend = "cute"

    @property
    def name(self) -> str:
        return "flashinfer_ssd"

    def prefill_metadata_chunk_size(self, chunk_size: int) -> int:
        if chunk_size not in (128, 256):
            raise ValueError(
                "SSDCombined Mamba prefill requires logical chunk_size 128 or 256"
            )
        return 128

    def chunk_scan_combined(self, *args, **kwargs):
        if "chunk_size" in kwargs:
            kwargs["chunk_size"] = self.prefill_metadata_chunk_size(
                int(kwargs["chunk_size"])
            )
        elif len(args) >= 6:
            args = (
                *args[:5],
                self.prefill_metadata_chunk_size(int(args[5])),
                *args[6:],
            )
        else:
            raise ValueError(
                "SSDCombined Mamba prefill requires an explicit chunk_size"
            )
        return super().chunk_scan_combined(*args, **kwargs)


class CakeSSUBackend(FlashInferSSDCombinedSSUBackend):
    """Cake SSDCombined prefill with FlashInfer selective-state-update decode."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._prefill_backend = "cake"

    @property
    def name(self) -> str:
        return "cake"


_BACKEND_REGISTRY: dict[str, type[MambaSSUBackend]] = {
    "triton": TritonSSUBackend,
    "flashinfer": FlashInferSSUBackend,
    "flashinfer_ssd": FlashInferSSDCombinedSSUBackend,
    "cake": CakeSSUBackend,
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
        _mamba_ssu_backend = backend_cls(
            enable_stochastic_rounding=(
                server_args.enable_mamba_cache_stochastic_rounding
            ),
            cache_philox_rounds=server_args.mamba_cache_philox_rounds,
        )
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


def mamba_chunk_scan_combined(*args, **kwargs):
    """Dispatch Mamba prefill to the configured backend without fallback."""
    assert _mamba_ssu_backend is not None, (
        "Mamba backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
    )
    return _mamba_ssu_backend.chunk_scan_combined(*args, **kwargs)


def mamba_prefill_requires_chunk_metadata() -> bool:
    """Whether the selected prefill backend needs logical chunks unconditionally."""
    return bool(getattr(_mamba_ssu_backend, "prefill_requires_chunk_metadata", False))


def mamba_prefill_metadata_chunk_size(chunk_size: int) -> int:
    """Resolve logical model chunks to the selected backend's physical chunks."""
    assert _mamba_ssu_backend is not None, (
        "Mamba backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
    )
    return _mamba_ssu_backend.prefill_metadata_chunk_size(chunk_size)
