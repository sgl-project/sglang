"""Ascend NPU backend for Kimi-K3 KDA (Key-Delta Attention).

Decode uses the triton fused_recurrent_kda_packed_decode kernel; prefill uses
the pypto chunk_kda wrapper. Causal conv1d is a torch-native fallback.
"""

import logging
from typing import Optional, Tuple, Union

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)


def _l2norm_fp32(x: torch.Tensor) -> torch.Tensor:
    """Per-head L2 normalization in fp32; returns the input dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _compute_kda_gate(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Per-K log-decay ``gk`` [1, T, H, K] fp32 from raw forget-gate ``a``.

    Matches the in-kernel gate of fused_recurrent_kda_packed_decode / chunk_kda.
    """
    _, T, H, K = a.shape
    x = a.float() + dt_bias.view(H, K).view(1, 1, H, K)
    alog = A_log.view(1, 1, H, 1).float()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(torch.exp(alog) * x)
    softplus = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)
    return -torch.exp(alog) * softplus


# Causal conv1d torch-native fallback (sgl_kernel_npu has dtype/shape bugs on 910C).


def _causal_conv1d_fallback_decode(
    x: torch.Tensor,  # [batch, dim]
    conv_state: torch.Tensor,  # [N, K-1, D] (transposed state)
    weight: torch.Tensor,  # [out_dim, 1, kernel]
    bias: torch.Tensor | None,  # [out_dim]
    cache_indices: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step causal conv1d with state update (decode path)."""
    batch_size = cache_indices.shape[0]
    kernel_size = weight.shape[-1]
    dim = x.shape[-1]
    out_dim = weight.shape[0]

    # .contiguous() is required: transpose creates a strided view on NPU.
    state = conv_state[cache_indices].transpose(1, 2).to(x.dtype).contiguous()  # [B, D, K-1]

    w = weight.squeeze(1).to(x.dtype)  # [out_dim, K]
    b = bias.to(x.dtype) if bias is not None else None

    # Build full K-element window: state (K-1 past) + current input
    window = torch.cat([state, x.unsqueeze(-1)], dim=-1)  # [B, D, K]

    # Convolve (K=4 is tiny, loop is fine)
    out = torch.zeros(batch_size, out_dim, dtype=x.dtype, device=x.device)
    for i in range(kernel_size):
        wi = w[:, i].view(dim, -1).contiguous()  # [D, out_dim/D]
        win_i = window[:, :, i].contiguous()  # [B, D]
        out = out + (win_i.unsqueeze(-1) * wi.unsqueeze(0)).reshape(batch_size, out_dim)

    if bias is not None:
        out = out + b
    if activation == "silu":
        out = torch.nn.functional.silu(out)

    # Store updated state: last K-1 elements of window → [B, K-1, D]
    conv_state[cache_indices] = window[..., 1:].transpose(1, 2).to(conv_state.dtype)
    return out, conv_state


def _causal_conv1d_fallback_prefill(
    x: torch.Tensor,  # [dim, total_seq]
    conv_state: torch.Tensor,  # [batch, K-1, dim] (transposed state)
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    weight: torch.Tensor,  # [out_dim, 1, kernel]
    bias: torch.Tensor | None,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-step causal conv1d with state update (prefill path)."""
    w = weight.squeeze(1).to(x.dtype)  # [out_dim, kernel]
    b = bias.to(x.dtype) if bias is not None else None
    kernel_size = weight.shape[-1]
    dim = x.shape[0]
    out_dim = w.shape[0]

    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    batch_size = seqlens.shape[0]
    # states: [B, K-1, dim] → transpose to [B, dim, K-1]
    states = conv_state[cache_indices].transpose(1, 2).to(x.dtype)  # [B, dim, K-1]

    outputs = []
    for b in range(batch_size):
        start = query_start_loc[b].item()
        end = query_start_loc[b + 1].item()
        seq_len = end - start
        x_b = x[:, start:end]  # [dim, seq_len]

        if has_initial_state[b]:
            prefix = states[b]  # [dim, K-1]
            full = torch.cat([prefix, x_b], dim=-1)  # [dim, K-1 + seq_len]
        else:
            prefix = torch.zeros(dim, kernel_size - 1, dtype=x.dtype, device=x.device)
            full = torch.cat([prefix, x_b], dim=-1)  # [dim, K-1 + seq_len]

        # Unfold into sliding windows: [dim, T] → [dim, seq_len, K]
        # .contiguous() is required: unfold creates a strided view on NPU.
        windows = full.unfold(-1, kernel_size, 1)[:, :seq_len, :].contiguous()  # [dim, seq_len, K]

        # Convolve (K=4 is tiny, loop is fine)
        out_b = torch.zeros(seq_len, out_dim, dtype=x.dtype, device=x.device)
        for i in range(kernel_size):
            wi = w[:, i].view(dim, -1).contiguous()  # [dim, O_per_dim]
            win_i = windows[:, :, i].t().contiguous()  # [seq_len, dim]
            out_b = out_b + (win_i.unsqueeze(-1) * wi.unsqueeze(0)).reshape(seq_len, out_dim)

        out_b = out_b.transpose(0, 1).contiguous()  # [out_dim, seq_len]

        if bias is not None:
            out_b = out_b + b.unsqueeze(1)

        if activation == "silu":
            out_b = torch.nn.functional.silu(out_b)

        outputs.append(out_b)

        # Update state: last K-1 elements of sequence
        full_seq = x[:, start:end]  # [dim, seq_len]
        if seq_len >= kernel_size - 1:
            states[b] = full_seq[:, -(kernel_size - 1):]
        else:
            prev = states[b] if has_initial_state[b] else torch.zeros(dim, kernel_size - 1, dtype=x.dtype, device=x.device)
            combined = torch.cat([prev, full_seq], dim=-1)
            states[b] = combined[:, -(kernel_size - 1):]

    # Write back: transpose [B, dim, K-1] → [B, K-1, dim]
    conv_state[cache_indices] = states.transpose(1, 2).to(conv_state.dtype)
    out = torch.cat(outputs, dim=-1)  # [out_dim, total_seq]
    return out, conv_state


class AscendKDAAttnBackend(AscendMambaAttnBackendBase):
    """NPU backend for KDA (Kimi Delta Attention) layers on Ascend 910C."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # KDA conv states are stored transposed vs Mamba2 convention; expose the
        # transposed shape so the track path reads conv_states_shape[-1] as the
        # conv window length.
        self.conv_states_shape = torch.Size(
            (
                *model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
                .transpose(-1, -2)
                .shape,
            )
        )
        self.decode_backend = "cann_recurrent"
        self.prefill_backend = "pto_chunk"

    # metadata

    def _prepare_kda_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Pre-compute per-forward metadata tensors for CANN operators."""
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accept_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            self.ssm_state_indices = torch.arange(
                cache_indices.shape[0] * seq_len,
                dtype=torch.int32,
                device=cache_indices.device,
            )
        else:
            self.ssm_state_indices = cache_indices

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self._prepare_kda_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata(forward_batch)
        self._prepare_kda_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self.graph_mode = False

    # decode

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Decode path: one token per sequence, recurrent update."""
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]  # [N, D, kernel-1]
        ssm_states = layer_cache.temporal  # [N, heads, Dv, Dk]
        cache_indices = self.forward_metadata.mamba_cache_indices

        # causal conv1d
        assert isinstance(mixed_qkv, torch.Tensor)
        conv_states_tmp = conv_states.transpose(1, 2).clone()  # [N, K-1, D]
        qkv, conv_states_out = _causal_conv1d_fallback_decode(
            mixed_qkv,
            conv_states_tmp,
            layer.conv_weights,
            layer.bias,
            cache_indices,
            activation="silu",
        )
        # conv_states_out is [N, K-1, D] → transpose back
        conv_states[:] = conv_states_out.transpose(1, 2).to(conv_states.dtype)

        # KDA recurrent decode via the triton fused_recurrent_kda_packed_decode
        # kernel: computes gk/beta/q-k L2norm/scale in-kernel and reads/writes
        # the recurrent state at cache_indices in place.
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_kda_packed_decode,
        )
        N = qkv.shape[0]
        hv = layer.num_v_heads
        out = qkv.new_empty(N, 1, hv, layer.head_v_dim)
        a_2d = a if a.dim() == 2 else a.reshape(N, -1)
        b_2d = b if b.dim() == 2 else b.reshape(N, -1)
        core_attn_out, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=qkv.contiguous(),
            a=a_2d.contiguous(),
            b=b_2d.contiguous(),
            A_log=layer.A_log.reshape(-1).contiguous(),
            dt_bias=layer.dt_bias.reshape(-1).contiguous(),
            scale=layer.head_k_dim ** -0.5,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=layer.lower_bound,
        )
        # out [N, 1, HV, V] → [1, N, HV, V] (model does .squeeze(0)).
        core_attn_out = core_attn_out.transpose(0, 1)

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        return core_attn_out

    def _cann_recurrent_decode(
        self,
        mix_qkv_packed: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_value_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        gk: torch.Tensor,
        scale: float,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Wrap CANN aclnnRecurrentGatedDeltaRule via torch.ops.npu."""
        beta = beta.to(torch.bfloat16)

        if self.graph_mode:
            num_accept_tokens = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            actual_seq_lengths = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            ssm_state_indices = self.forward_metadata.mamba_cache_indices
        else:
            num_accept_tokens = self.num_accept_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.ssm_state_indices

        # CANN aclnnRecurrentGatedDeltaRule requires an EVEN head count: an odd
        # num_heads leaves a dangling half-block and raises ADDR_MISALIGN. Pad
        # the head dim up to the next even number, run the op, then slice back.
        # The padded head carries zeroed Q/K/V; state slot indexing is remapped
        # onto a gathered copy of the touched slots.
        nv = num_value_heads
        head_pad = nv & 1  # 1 when odd -> pad to even
        nk_pad = num_heads + head_pad
        nv_pad = nv + head_pad
        ssm_state_indices = ssm_state_indices.view(batch_size, -1)
        if head_pad == 0:
            attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
                mix_qkv_packed.contiguous(),
                recurrent_state.contiguous(),
                beta=beta.contiguous(),
                scale=scale,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=ssm_state_indices.contiguous(),
                nk=num_heads,
                nv=nv,
                intermediate_state=None,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accept_tokens,
                g=g.contiguous(),
                gk=gk.contiguous(),
            )
            return attn_core_out.contiguous()

        # --- odd-head path: pad head dim, gather touched state slots, remap ---
        N = mix_qkv_packed.shape[1]
        N = mix_qkv_packed.shape[1]
        qk_dim = num_heads * head_k_dim
        v_dim = nv * head_v_dim
        qk_part = mix_qkv_packed[..., : 2 * qk_dim].view(1, N, 2, num_heads, head_k_dim)
        v_part = mix_qkv_packed[..., 2 * qk_dim :].view(1, N, nv, head_v_dim)
        qk_pad = torch.nn.functional.pad(qk_part, (0, 0, 0, head_pad))  # [1,N,2,nk_pad,hk]
        v_pad = torch.nn.functional.pad(v_part, (0, 0, 0, head_pad))  # [1,N,nv_pad,hv]
        mix_qkv_pad = torch.cat(
            [qk_pad.reshape(1, N, 2 * nk_pad * head_k_dim), v_pad.reshape(1, N, nv_pad * head_v_dim)],
            dim=-1,
        ).contiguous()

        beta_pad = torch.nn.functional.pad(beta, (0, head_pad))  # pad nv -> nv_pad on last dim
        g_pad = torch.nn.functional.pad(g, (0, head_pad))
        # gk is [T, nv, Dk]; pad the head (nv) axis, leave Dk unchanged.
        gk_pad = torch.nn.functional.pad(gk, (0, 0, 0, head_pad))

        # Gather the touched state slots, pad head dim, remap indices via searchsorted.
        all_idx = torch.cat([ssm_state_indices.reshape(-1), cache_indices.reshape(-1)])
        unique_idx = torch.unique(all_idx)  # sorted ascending
        state_local = recurrent_state[unique_idx]  # [U, nv, hk, hv]
        state_pad = torch.nn.functional.pad(state_local, (0, 0, 0, 0, 0, head_pad))  # [U, nv_pad, hk, hv]
        local_ssm = torch.searchsorted(unique_idx, ssm_state_indices.reshape(-1)).reshape(ssm_state_indices.shape).to(torch.int32)
        local_cache = torch.searchsorted(unique_idx, cache_indices.reshape(-1)).reshape(cache_indices.shape).to(torch.int32)

        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv_pad,
            state_pad,
            beta=beta_pad.contiguous(),
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=local_ssm.contiguous(),
            nk=nk_pad,
            nv=nv_pad,
            intermediate_state=None,
            cache_indices=local_cache,
            num_accepted_tokens=num_accept_tokens,
            g=g_pad.contiguous(),
            gk=gk_pad.contiguous(),
        )
        # Slice real heads back; scatter updated slots to the global cache.
        attn_core_out = attn_core_out[..., :nv, :].contiguous()
        recurrent_state[unique_idx] = state_pad[:, :nv]
        return attn_core_out

    # prefill (extend)

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Prefill path: multi-token chunked KDA."""
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        has_initial_states = forward_batch.extend_prefix_lens > 0

        # causal conv1d (prefill / chunked)
        mixed_qkv_t = mixed_qkv.transpose(0, 1)  # [dim, seq_len]
        kernel_size = layer.conv_weights.shape[-1]
        conv_states_tmp = conv_states.transpose(1, 2).contiguous()  # [N, K-1, D]

        mixed_qkv_t, conv_states_tmp = _causal_conv1d_fallback_prefill(
            mixed_qkv_t,
            conv_states_tmp,
            query_start_loc,
            cache_indices,
            has_initial_states,
            layer.conv_weights,
            layer.bias,
            activation="silu",
        )
        mixed_qkv_t = mixed_qkv_t.transpose(0, 1)[:seq_len]

        conv_states[:] = conv_states_tmp.transpose(1, 2).contiguous().to(conv_states.dtype)

        # Zero ssm state for new sequences (no prefix): reused slots are not
        # zeroed by the allocator and would leak the previous request's state.
        if not bool(has_initial_states.all()):
            new_indices = cache_indices[~has_initial_states]
            if new_indices.numel() > 0:
                ssm_states[new_indices] = 0

        # KDA chunked prefill via the pypto chunk_kda wrapper (1 launch, chunked).
        # Same KDA recurrence / state ABI / gate / L2norm / scale as forward_decode,
        # so the state it writes is exactly what decode reads. chunk_kda takes
        # pre-activated g=gk and sigmoid'd beta; it returns a fresh s_out (not
        # inplace), so scatter it back to ssm_states. Imported lazily.
        from sglang.srt.hardware_backend.npu.attention.pto.kda_flash.chunk_kda.chunk_kda_impl import (
            chunk_kda_wrapper,
        )
        hv = layer.num_v_heads
        kd = layer.head_k_dim
        vv = layer.head_v_dim
        A_log = layer.A_log.reshape(-1).contiguous()
        dt_bias = layer.dt_bias.reshape(-1).contiguous()
        scale = layer.head_k_dim ** -0.5

        # mixed_qkv_t: [seq_len, packed_dim] (post-conv, Q|K|V), bf16 -> [1,T,H,K/V]
        qk_dim = hv * kd
        q = mixed_qkv_t[:, :qk_dim].reshape(1, seq_len, hv, kd).contiguous()
        k = mixed_qkv_t[:, qk_dim : 2 * qk_dim].reshape(1, seq_len, hv, kd).contiguous()
        v = mixed_qkv_t[:, 2 * qk_dim :].reshape(1, seq_len, hv, vv).contiguous()
        g = _compute_kda_gate(a.reshape(1, seq_len, hv, kd), A_log, dt_bias, layer.lower_bound)
        beta = b.reshape(1, seq_len, hv).float()

        init_state = ssm_states[cache_indices].contiguous()  # [N, hv, vv, kd]
        core_attn_out, s_out = chunk_kda_wrapper(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=init_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )  # core_attn_out [1, T, hv, vv]; s_out [N, hv, vv, kd]
        ssm_states[cache_indices] = s_out

        return core_attn_out

    def _recurrent_prefill_fallback(
        self,
        Q, K, V, beta, g, gk, scale,
        ssm_states, cache_indices, seq_len, batch_size,
    ) -> torch.Tensor:
        """Prefill fallback: call recurrent_gated_delta_rule one token at a time."""
        CHUNK_SIZE = 1  # 1 token/op-call: no race, no OOB, correct handoff
        # Save mutable state to restore after this fallback
        saved_num_accept = self.num_accept_tokens
        saved_seq_lengths = self.actual_seq_lengths
        saved_ssm_indices = self.ssm_state_indices
        device = cache_indices.device

        outputs = []
        for start in range(0, seq_len, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, seq_len)
            chunk_size = end - start

            chunk_qkv = torch.cat([
                Q[:, start:end].squeeze(0).reshape(-1, Q.shape[-2] * Q.shape[-1]),
                K[:, start:end].squeeze(0).reshape(-1, K.shape[-2] * K.shape[-1]),
                V[:, start:end].squeeze(0).reshape(-1, V.shape[-2] * V.shape[-1]),
            ], dim=-1).unsqueeze(0)  # [1, chunk, 3*D]

            chunk_beta = beta[:, start:end] if beta.dim() == 3 else beta[start:end]
            chunk_g = g[start:end]
            chunk_gk = gk[start:end]

            # Per-chunk: update state for _cann_recurrent_decode
            self.num_accept_tokens = torch.full(
                [batch_size], chunk_size, dtype=torch.int32, device=device,
            )
            self.actual_seq_lengths = self.num_accept_tokens
            # CHUNK_SIZE=1: ssm_state_indices = cache_indices; state chains across calls.
            self.ssm_state_indices = cache_indices.repeat_interleave(
                chunk_size
            ).to(torch.int32)

            out = self._cann_recurrent_decode(
                mix_qkv_packed=chunk_qkv,
                batch_size=batch_size,
                num_heads=Q.shape[-2],
                num_value_heads=V.shape[-2],
                head_k_dim=Q.shape[-1],
                head_v_dim=V.shape[-1],
                recurrent_state=ssm_states,
                beta=chunk_beta.view(batch_size, -1, V.shape[-2]).contiguous(),
                g=chunk_g.reshape(-1, V.shape[-2]),
                gk=chunk_gk.reshape(-1, V.shape[-2], Q.shape[-1]).contiguous(),
                scale=scale,
                cache_indices=cache_indices,
            )
            outputs.append(out)

        # Restore state
        self.num_accept_tokens = saved_num_accept
        self.actual_seq_lengths = saved_seq_lengths
        self.ssm_state_indices = saved_ssm_indices

        return torch.cat(outputs, dim=1)  # concat along seq_len for chunks [1, s1, D] + [1, s2, D]
