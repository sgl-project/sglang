import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.gdn_aiter_prefill import (
    FUSED as AITER_GDN_PREFILL_FUSED,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_hip, is_npu, is_xpu

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_gdn_replayssm_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized import (
        fused_sigmoid_gating_delta_rule_update_decode_npu as fused_sigmoid_gating_delta_rule_update,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )
elif is_xpu():
    from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )


_AITER_GDN_PREFILL = None
_AITER_GDN_PREFILL_UNAVAILABLE = False


def _aiter_gdn_prefill_requested() -> bool:
    return envs.SGLANG_USE_AITER.get() and envs.SGLANG_AITER_GDN_PREFILL.get()


def _aiter_gdn_prefill():
    """AITER's chunked prefill, or None if it cannot be used.

    Opt-in: SGLANG_USE_AITER plus SGLANG_AITER_GDN_PREFILL. It replaces the five
    Triton FLA prefill kernels with either the opt_vk stages (fused FlyDSL
    prepare, a tuned hidden-state kernel, the VK output kernel) or, with
    SGLANG_AITER_GDN_PREFILL_H=fused, one FlyDSL kernel for the whole chunk loop.
    """
    global _AITER_GDN_PREFILL, _AITER_GDN_PREFILL_UNAVAILABLE
    if _AITER_GDN_PREFILL is not None or _AITER_GDN_PREFILL_UNAVAILABLE:
        return _AITER_GDN_PREFILL
    if not _aiter_gdn_prefill_requested():
        _AITER_GDN_PREFILL_UNAVAILABLE = True
        return None
    try:
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter_prefill import (
            AiterGDNPrefill,
        )

        _AITER_GDN_PREFILL = AiterGDNPrefill(envs.SGLANG_AITER_GDN_PREFILL_H.get())
    except ImportError as exc:
        logger.info("aiter opt_vk GDN prefill unavailable (%s); keeping Triton", exc)
        _AITER_GDN_PREFILL_UNAVAILABLE = True
    return _AITER_GDN_PREFILL


def _fill_track_state_from_h(
    h: torch.Tensor,
    track_state: torch.Tensor,
    track_chunk_idx: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Copy each sequence's tracked chunk-start state out of the per-chunk ``h``.

    ``h`` is [1, total_chunks, H, V, K] with each sequence's chunks contiguous;
    rows with ``track_chunk_idx < 0`` receive an arbitrary chunk and are never
    read. Index math stays on the device so the copy does not sync the stream.
    """
    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_chunks = (seq_lens + 63) // 64
    first_chunk = torch.cumsum(num_chunks, 0) - num_chunks
    src = (first_chunk + track_chunk_idx.clamp(min=0)).clamp(max=h.shape[1] - 1)
    track_state.copy_(h[0].index_select(0, src.long()))


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()
    supports_strided_target_verify_qkv: bool = True

    def __init__(self):
        # The AITER fused prefill writes the fp32 track snapshot in-kernel; any
        # batch it declines falls back to a kernel whose h fills the snapshot.
        self.supports_track_state_snapshot = (
            _is_hip
            and _aiter_gdn_prefill_requested()
            and envs.SGLANG_AITER_GDN_PREFILL_H.get() == AITER_GDN_PREFILL_FUSED
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # GDN ReplaySSM buffered decode (slice 1a). Drop-in for the packed
        # decode: same args plus the three per-layer ring caches and the
        # per-row write cursor. When any ring tensor / cursor is None (flag
        # off) we fall through to the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        # GDN ReplaySSM (slice 2b): optional per-row force-flush (radix track
        # boundary). None when radix tracking is off / flag off; the kernel
        # treats None as "no forced flush" (byte-identical to slice 1a/1b).
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
            )
            return out.transpose(0, 1)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

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
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

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
        inplace_update: bool = True,
        seq_lens_cpu: list[int] | None = None,
        track_state: torch.Tensor | None = None,
        track_chunk_idx: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple:
        o, last_state, h = self._extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            inplace_update=inplace_update,
            seq_lens_cpu=seq_lens_cpu,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )
        if track_state is not None and h is not None:
            _fill_track_state_from_h(h, track_state, track_chunk_idx, query_start_loc)
        return o, last_state, h

    def _extend(
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
        inplace_update: bool,
        seq_lens_cpu: list[int] | None,
        track_state: torch.Tensor | None,
        track_chunk_idx: torch.Tensor | None,
    ) -> tuple:
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        inplace_update_args = {"inplace_update": inplace_update}
        if is_cpu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the CPU chunk kernel"
                )
            inplace_update_args = {}
        elif is_npu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the NPU chunk kernel"
                )
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
            # The external NPU kernel does not expose the optional write-back
            # control. Its existing behavior is equivalent to True.
            inplace_update_args = {}
        elif (
            _is_hip
            and inplace_update
            and seq_lens_cpu is not None
            and len(seq_lens_cpu) == query_start_loc.numel() - 1
            and ssm_states.is_contiguous()
            and _aiter_gdn_prefill() is not None
        ):
            out = _aiter_gdn_prefill().extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                cu_seqlens=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
            )
            if out is not None:
                o, h = out
                return o, None, h

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **inplace_update_args,
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
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
