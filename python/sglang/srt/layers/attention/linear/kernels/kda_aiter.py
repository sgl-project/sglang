import inspect
from typing import Optional

import torch

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_gfx95_supported

_CHUNK_SIZE = 64
_MIN_TOKENS = 64
_MAX_TOKENS = 16384
_MAX_SEQUENCES = 8
_TRACKED_MIN_TOKENS = {16: 8192, 8: 14336}


def _load_chunk_kda():
    if not is_gfx95_supported():
        return None, False
    try:
        from aiter.ops.triton.attention import chunk_kda as module
    except (ImportError, ModuleNotFoundError):
        return None, False
    # ROCm/aiter#5866 provides chunk_kda, but only ROCm/aiter#6672 lets
    # gfx950 load/store the BF16 state cache used by GLM-5.3.
    try:
        supports_bf16_state = "torch.bfloat16" in inspect.getsource(
            module.chunk_kda_walk
        )
    except (OSError, TypeError):
        supports_bf16_state = False
    return module.chunk_kda, supports_bf16_state


_AITER_CHUNK_KDA, _AITER_SUPPORTS_BF16_STATE = _load_chunk_kda()


def aiter_kda_prefill_available() -> bool:
    return _AITER_CHUNK_KDA is not None and _AITER_SUPPORTS_BF16_STATE


class AiterKDAPrefillKernel(LinearAttnKernelBase):
    """gfx950 AITER chunk-KDA prefill with a Triton fallback."""

    supports_track_state_snapshot: bool = True
    supports_safe_gate: bool = True

    def __init__(self, triton_fallback: Optional[TritonKDAKernel] = None):
        if not is_gfx95_supported():
            raise ValueError("AITER KDA prefill requires gfx950")
        if _AITER_CHUNK_KDA is None:
            raise ValueError("AITER KDA prefill requires merged ROCm/aiter#5866")
        if not _AITER_SUPPORTS_BF16_STATE:
            raise ValueError(
                "AITER KDA prefill with GLM-5.3 BF16 state requires merged "
                "ROCm/aiter#6672"
            )
        self._triton = triton_fallback or TritonKDAKernel()
        self._dispatches = 0
        self._aiter_dispatches = 0
        self._logged_rejections = set()

    def decode(self, *args, **kwargs):
        raise NotImplementedError("AiterKDAPrefillKernel is prefill-only")

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("AiterKDAPrefillKernel is prefill-only")

    def _fallback(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        **kwargs,
    ):
        self._record_dispatch(False)
        return self._triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def _record_dispatch(self, use_aiter: bool) -> None:
        self._dispatches += 1
        self._aiter_dispatches += int(use_aiter)
        if self._dispatches & (self._dispatches - 1) == 0:
            from sglang.srt.utils.common import rank0_log

            rank0_log(
                "AITER KDA prefill dispatch: "
                f"hits={self._aiter_dispatches}/{self._dispatches}"
            )

    def _log_rejection(
        self,
        *,
        q,
        ssm_states,
        num_sequences,
        beta_is_raw,
        lower_bound,
        return_intermediate_states,
        track_state,
        track_chunk_idx,
        track_chunk_idx_cpu,
        is_spec_decode,
    ) -> None:
        signature = (
            q.shape[1],
            q.shape[-2],
            num_sequences,
            str(q.dtype),
            str(ssm_states.dtype),
            beta_is_raw,
            lower_bound,
            return_intermediate_states,
            track_state is not None,
            track_chunk_idx is not None,
            track_chunk_idx_cpu is not None,
            bool(is_spec_decode),
        )
        if signature in self._logged_rejections or len(self._logged_rejections) >= 8:
            return
        self._logged_rejections.add(signature)
        from sglang.srt.utils.common import rank0_log

        rank0_log(f"AITER KDA prefill fallback contract: {signature}")

    @staticmethod
    def _single_sequence_metadata(
        length: int, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_chunks = (length + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        query_start_loc = source.new_tensor([0, length])
        chunk_indices = torch.stack(
            (
                torch.zeros(num_chunks, device=source.device, dtype=source.dtype),
                torch.arange(num_chunks, device=source.device, dtype=source.dtype),
            ),
            dim=1,
        )
        chunk_offsets = source.new_tensor([0, num_chunks])
        return query_start_loc, chunk_indices, chunk_offsets

    def _run_aiter(
        self,
        *,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        lower_bound,
        query_start_loc,
        chunk_indices,
        chunk_offsets,
        initial_state=None,
        output_final_state=False,
        state_cache=None,
        cache_indices=None,
    ):
        return _AITER_CHUNK_KDA(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            cu_seqlens=query_start_loc,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            scale=k.shape[-1] ** -0.5,
            out=v,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_cache=state_cache,
            state_indices=cache_indices,
            has_initial_state=(
                cache_indices >= 0 if cache_indices is not None else None
            ),
        )

    def _extend_single_tracked(
        self,
        *,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        lower_bound,
        ssm_states,
        cache_indices,
        track_state,
        track_chunk_idx_cpu,
    ):
        track_chunk = track_chunk_idx_cpu[0]
        split = track_chunk * _CHUNK_SIZE
        total = q.shape[1]
        if track_chunk < 0 or split >= total:
            return None

        safe_index = cache_indices.clamp_min(0).long()
        valid = cache_indices >= 0
        initial_state = ssm_states.index_select(0, safe_index).float()
        initial_state = torch.where(
            valid.view(-1, 1, 1, 1), initial_state, torch.zeros_like(initial_state)
        )
        output = v
        state = initial_state

        if split:
            qsl, chunk_indices, chunk_offsets = self._single_sequence_metadata(
                split, cache_indices
            )
            _, state = self._run_aiter(
                q=q[:, :split],
                k=k[:, :split],
                v=output[:, :split],
                g=g[:, :split],
                beta=beta[:, :split],
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                query_start_loc=qsl,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                initial_state=state,
                output_final_state=True,
            )

        track_state.copy_(state)
        suffix = total - split
        if suffix:
            qsl, chunk_indices, chunk_offsets = self._single_sequence_metadata(
                suffix, cache_indices
            )
            _, state = self._run_aiter(
                q=q[:, split:],
                k=k[:, split:],
                v=output[:, split:],
                g=g[:, split:],
                beta=beta[:, split:],
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                query_start_loc=qsl,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                initial_state=state,
                output_final_state=True,
            )

        current = ssm_states.index_select(0, safe_index)
        committed = torch.where(
            valid.view(-1, 1, 1, 1), state.to(ssm_states.dtype), current
        )
        ssm_states.index_copy_(0, safe_index, committed)
        return output

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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        num_tokens = q.shape[1]
        num_sequences = query_start_loc.numel() - 1
        track_state = kwargs.get("track_state")
        track_chunk_idx = kwargs.get("track_chunk_idx")
        track_chunk_idx_cpu = kwargs.pop("track_chunk_idx_cpu", None)
        tracked = (
            return_intermediate_states
            or track_state is not None
            or track_chunk_idx is not None
        )
        supported = (
            not kwargs.get("is_spec_decode")
            and beta_is_raw
            and A_log is not None
            and A_log.is_contiguous()
            and dt_bias is not None
            and dt_bias.is_contiguous()
            and lower_bound is not None
            and -5.5 <= lower_bound < 0
            and _MIN_TOKENS <= num_tokens <= _MAX_TOKENS
            and 1 <= num_sequences <= _MAX_SEQUENCES
            and ssm_states.dtype in (torch.float32, torch.bfloat16)
            and cache_indices.dtype in (torch.int32, torch.int64)
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and g.dtype == torch.bfloat16
            and q.ndim == 4
            and q.shape[0] == 1
            and q.shape[-2] in (8, 16)
            and q.shape[-1] == 128
            and k.shape == q.shape
            and g.shape == q.shape
            and v.shape == q.shape
            and q.stride()[2:] == (128, 1)
            and k.stride()[2:] == (128, 1)
            and v.stride()[2:] == (128, 1)
            and g.stride()[2:] == (128, 1)
            and beta.shape == q.shape[:-1]
            and beta.stride(-1) == 1
            and A_log.numel() == q.shape[-2]
            and dt_bias.numel() == q.shape[-2] * 128
            and (
                not tracked
                or (
                    return_intermediate_states
                    and (
                        (track_state is None and track_chunk_idx is None)
                        or (
                            track_state is not None
                            and track_chunk_idx is not None
                            and track_chunk_idx_cpu is not None
                            and num_sequences == 1
                            and num_tokens >= _TRACKED_MIN_TOKENS[q.shape[-2]]
                        )
                    )
                )
            )
        )
        if not supported:
            self._log_rejection(
                q=q,
                ssm_states=ssm_states,
                num_sequences=num_sequences,
                beta_is_raw=beta_is_raw,
                lower_bound=lower_bound,
                return_intermediate_states=return_intermediate_states,
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
                track_chunk_idx_cpu=track_chunk_idx_cpu,
                is_spec_decode=kwargs.get("is_spec_decode"),
            )
            return self._fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_chunk_idx_cpu=track_chunk_idx_cpu,
                **kwargs,
            )

        if tracked:
            if track_state is None and track_chunk_idx is None:
                chunk_indices = kwargs.get("chunk_indices")
                if chunk_indices is None:
                    chunk_indices = prepare_chunk_indices(query_start_loc, _CHUNK_SIZE)
                chunk_offsets = kwargs.get("chunk_offsets")
                if chunk_offsets is None:
                    chunk_offsets = prepare_chunk_offsets(query_start_loc, _CHUNK_SIZE)
                output, final_state = self._run_aiter(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                    query_start_loc=query_start_loc,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    state_cache=ssm_states,
                    cache_indices=cache_indices,
                )
                assert final_state is None
                self._record_dispatch(True)
                return output, None
            output = self._extend_single_tracked(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                track_state=track_state,
                track_chunk_idx_cpu=track_chunk_idx_cpu,
            )
            if output is None:
                return self._fallback(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                    beta_is_raw=beta_is_raw,
                    return_intermediate_states=return_intermediate_states,
                    track_state=track_state,
                    track_chunk_idx=track_chunk_idx,
                    **kwargs,
                )
            self._record_dispatch(True)
            return output, None

        chunk_indices = kwargs.get("chunk_indices")
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(query_start_loc, _CHUNK_SIZE)
        chunk_offsets = kwargs.get("chunk_offsets")
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(query_start_loc, _CHUNK_SIZE)

        output, final_state = self._run_aiter(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            query_start_loc=query_start_loc,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_cache=ssm_states,
            cache_indices=cache_indices,
        )
        assert final_state is None
        self._record_dispatch(True)
        return output
