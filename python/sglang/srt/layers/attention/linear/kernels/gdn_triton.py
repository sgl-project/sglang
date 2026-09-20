import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_gdn_replayssm_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
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


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

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
        **kwargs,
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

        # --- AscendC prefill switch (patch_gdn_prefill_ascendc.py) ---
        # #747 flipped the pool, the decode kernel and the verify operator to
        # (nv, dv, dk); the triton path here still returns (nv, dk, dv), so the
        # state written back to the pool is transposed -- silently, because
        # dk == dv == 128 makes the shapes identical. Use the operator #747
        # added, which is native to the unified layout.
        if is_npu():
            import torch as _torch

            _op = getattr(_torch.ops.npu, "chunk_gated_delta_rule", None)
            if _op is None:
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "chunk_gated_delta_rule is not registered; falling back to the "
                    "triton prefill, whose state layout does not match the pool."
                )
            else:
                from sgl_kernel_npu.fla.l2norm import l2norm_fwd as _l2norm

                _t, _nk, _dk = q.shape[-3], q.shape[-2], q.shape[-1]
                _nv, _dv = v.shape[-2], v.shape[-1]
                # q/k/v arrive as strided views of mixed_qkv, so one reshape each
                # is the copy the operator needs; l2norm_fwd returns contiguous.
                _q = _l2norm(q.reshape(-1, _dk)).view(_t, _nk, _dk)
                _k = _l2norm(k.reshape(-1, _dk)).view(_t, _nk, _dk)
                _lens = _torch.diff(query_start_loc).to(_torch.int32)
                # The operator's chunk grid is sum_b ceil(len_b / 64), the same
                # grid _init_track_ssm_indices builds for GDN, and it writes each
                # chunk's entering state -- so chunk_state is exactly the `h` the
                # mamba page tracking reads. Sizing it costs one host sync.
                _chunks = int(((_lens + 63) // 64).sum())
                _h = _torch.empty(
                    _chunks,
                    _nv,
                    _dv,
                    _dk,
                    dtype=recurrent_state.dtype,
                    device=recurrent_state.device,
                )
                _out, _state = _op(
                    _q,
                    _k,
                    v.reshape(_t, _nv, _dv),
                    beta=beta.reshape(_t, _nv),
                    initial_state=recurrent_state,
                    actual_seq_lengths=_lens,
                    scale=_dk**-0.5,
                    g=g.reshape(_t, _nv).to(_torch.float32),
                    chunk_state=_h,
                )
                return _out.unsqueeze(0), _state, _h.unsqueeze(0)
        # --- end switch ---

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
