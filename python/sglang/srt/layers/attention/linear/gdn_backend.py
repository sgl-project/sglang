from typing import Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        self.verify_kernel = triton_kernel

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}"
        )

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
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
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
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
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

    def target_verify(
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
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
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


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        # GQA expansion: tile Q/K from num_k_heads to num_v_heads (tiled pattern)
        if layer.num_k_heads != layer.num_v_heads:
            gqa_ratio = layer.num_v_heads // layer.num_k_heads
            query = query.repeat(1, 1, gqa_ratio, 1)
            key = key.repeat(1, 1, gqa_ratio, 1)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        _trace_gdn = getattr(self, '_trace_gdn_count', 0) < 1  # Enable GDN trace
        if _trace_gdn:
            self._trace_gdn_count = getattr(self, '_trace_gdn_count', 0) + 1

        if _trace_gdn:
            # Conv output: mixed_qkv is [seq_len, 8192] after conv+silu
            cv = mixed_qkv.float()
            # Flatten all to compare with ik_llama conv_output_silu-0 (which is flattened [81920])
            flat = cv.flatten()
            vals = ", ".join(f"{v:.6f}" for v in flat[:10].tolist())
            print(f"TRACE GDN_conv_silu-0 shape={list(cv.shape)} first10=[{vals}] mean={flat.mean():.6f} std={flat.std():.6f}", flush=True)
            # Token 0 only (first 8192 values)
            t0 = cv[0]
            vals0 = ", ".join(f"{v:.6f}" for v in t0[:10].tolist())
            print(f"TRACE GDN_conv_silu_t0-0 shape=[{t0.shape[0]}] first10=[{vals0}] mean={t0.mean():.6f} std={t0.std():.6f}", flush=True)

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        if _trace_gdn:
            # Q: [seq, q_dim=2048], K: [seq, k_dim=2048], V: [seq, v_dim=4096]
            qf = query.float()[0]
            vals_q = ", ".join(f"{v:.6f}" for v in qf[:10].tolist())
            print(f"TRACE GDN_q_t0-0 shape=[{qf.shape[0]}] first10=[{vals_q}] mean={qf.mean():.6f} std={qf.std():.6f}", flush=True)
            kf = key.float()[0]
            vals_k = ", ".join(f"{v:.6f}" for v in kf[:10].tolist())
            print(f"TRACE GDN_k_t0-0 shape=[{kf.shape[0]}] first10=[{vals_k}] mean={kf.mean():.6f} std={kf.std():.6f}", flush=True)
            vf = value.float()[0]
            vals_v = ", ".join(f"{v:.6f}" for v in vf[:10].tolist())
            print(f"TRACE GDN_v_t0-0 shape=[{vf.shape[0]}] first10=[{vals_v}] mean={vf.mean():.6f} std={vf.std():.6f}", flush=True)
            # ik_llama k_conv-0: shape=[128,16,6,1] first10=[0.000125, -0.000384, ...]
            # That's [head_k_dim, num_k_heads, seq, 1] - already head-major. Check our first head.
            kf_h0 = kf[:128]  # first 128 = head 0 of K
            vals_kh = ", ".join(f"{v:.6f}" for v in kf_h0[:10].tolist())
            print(f"TRACE GDN_k_h0_t0-0 first10=[{vals_kh}] mean={kf_h0.mean():.6f} std={kf_h0.std():.6f}", flush=True)

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        # GQA expansion: tile Q/K from num_k_heads to num_v_heads.
        # Must use tiled pattern [h0..h15, h0..h15] (torch.repeat),
        # NOT interleaved [h0,h0,h1,h1,...] (repeat_interleave).
        # The FLA kernel's built-in GQA (i_h // (H//Hg)) uses interleaved,
        # but the model weights expect tiled pairing.
        if layer.num_k_heads != layer.num_v_heads:
            gqa_ratio = layer.num_v_heads // layer.num_k_heads
            query = query.repeat(1, 1, gqa_ratio, 1)
            key = key.repeat(1, 1, gqa_ratio, 1)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        if _trace_gdn:
            # g: [1, seq, num_heads, 1], beta: [1, seq, num_heads, 1]
            gf = g.float().squeeze()  # [seq, num_heads] or [seq, num_heads, 1]
            gf0 = gf[0].flatten()  # token 0, all heads
            vals_g = ", ".join(f"{v:.6f}" for v in gf0[:10].tolist())
            print(f"TRACE GDN_g_t0-0 shape={list(gf.shape)} first10=[{vals_g}] mean={gf0.mean():.6f} std={gf0.std():.6f}", flush=True)
            # ik_llama g_in-0: [32, 6, 1, 1] first10=[-0.000077, -0.004989, -1.320975, ...]
            # That's [num_heads, seq, 1, 1] - head-major. Our g should have same values.
            bf = beta.float().squeeze()
            bf0 = bf[0].flatten()
            vals_b = ", ".join(f"{v:.6f}" for v in bf0[:10].tolist())
            print(f"TRACE GDN_beta_t0-0 shape={list(bf.shape)} first10=[{vals_b}] mean={bf0.mean():.6f} std={bf0.std():.6f}", flush=True)
            # ik_llama beta_in-0: [32, 1, 6, 1] first10=[0.252020, 0.647146, ...] — sigmoid applied

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if _trace_gdn:
                co = core_attn_out.float()
                # core_attn_out: [1, seq, num_v_heads, head_v_dim]
                co0 = co[0, 0].flatten()  # token 0 flattened
                vals_co = ", ".join(f"{v:.6f}" for v in co0[:10].tolist())
                print(f"TRACE GDN_recurrence_out-0 shape={list(co.shape)} first10=[{vals_co}] mean={co0.mean():.6f} std={co0.std():.6f}", flush=True)
                # Per-head stats for token 0
                for hi in range(min(16, co.shape[2])):
                    hslice = co[0, 0, hi].flatten()
                    print(f"TRACE GDN_recurrence_h{hi}_t0-0 mean={hslice.mean():.6f} std={hslice.std():.6f}", flush=True)

                # === Sequential recurrence comparison ===
                # Run ik_llama-style sequential loop and compare with FLA chunk output
                from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
                import math
                q_f = query.float()[0]   # [seq, Hq, Kd]  = [6, 16, 128]
                k_f = key.float()[0]     # [seq, Hk, Kd]  = [6, 16, 128]
                v_f = value.float()[0]   # [seq, Hv, Vd]  = [6, 32, 128]
                g_f = g.float()          # [1, seq, Hv]
                beta_f = beta.float()    # [1, seq, Hv]
                if g_f.dim() == 3:
                    g_f = g_f[0]         # [seq, Hv]
                    beta_f = beta_f[0]   # [seq, Hv]

                Hv = v_f.shape[1]        # 32
                Hk = k_f.shape[1]        # 16
                Kd = k_f.shape[2]        # 128
                Vd = v_f.shape[2]        # 128
                T = q_f.shape[0]         # 6
                gqa_ratio = Hv // Hk     # 2
                scale = 1.0 / math.sqrt(Kd)

                # L2-normalize Q, K (same as FLA kernel does)
                q_norm = torch.nn.functional.normalize(q_f, p=2, dim=-1)  # [T, Hq, Kd]
                k_norm = torch.nn.functional.normalize(k_f, p=2, dim=-1)  # [T, Hk, Kd]

                # Expand Q,K from Hk to Hv heads (GQA repeat)
                q_exp = q_norm.repeat_interleave(gqa_ratio, dim=1)  # [T, Hv, Kd]
                k_exp = k_norm.repeat_interleave(gqa_ratio, dim=1)  # [T, Hv, Kd]

                # Sequential recurrence (ik_llama style)
                # State convention: state[hv] is [K, V] (FLA convention)
                # h·k = state.T @ k → [V,K] @ [K] = [V]
                # h·q = state.T @ q → [V,K] @ [K] = [V]
                # h_new = decay * h + k outer vNew = decay*[K,V] + [K,1]*[1,V]
                state = torch.zeros(Hv, Kd, Vd, dtype=torch.float32, device=q_f.device)  # [Hv, K, V]
                seq_out = torch.zeros(T, Hv, Vd, dtype=torch.float32, device=q_f.device)

                for t in range(T):
                    for hv in range(Hv):
                        qt = q_exp[t, hv] * scale  # [Kd] - scaled query
                        kt = k_exp[t, hv]           # [Kd] - normalized key
                        vt = v_f[t, hv]             # [Vd]
                        bt = beta_f[t, hv].item()   # scalar (already sigmoid)
                        gt = g_f[t, hv].item()      # log-space gate
                        decay = math.exp(min(gt, 50.0))

                        # h·k = state.T @ k → [V] (current value retrieval)
                        hk = state[hv].T @ kt       # [Vd]

                        # Delta rule: vNew = beta * (v - decay * h·k)
                        vNew = bt * (vt - decay * hk)

                        # Output: o = decay * (state.T @ q_scaled) + vNew * (k · q_scaled)
                        hq = state[hv].T @ qt       # [Vd]
                        kq = (kt * qt).sum()         # scalar
                        seq_out[t, hv] = decay * hq + vNew * kq

                        # State update: h_new = decay*h + k⊗vNew  [K,V]
                        state[hv] = decay * state[hv] + kt.unsqueeze(1) * vNew.unsqueeze(0)
                        # Clamp like ik_llama
                        state[hv] = state[hv].clamp(-1e6, 1e6)

                # Compare sequential vs FLA
                fla_out = co[0]  # [T, Hv, Vd]
                for t in range(T):
                    fla_t = fla_out[t].flatten()
                    seq_t = seq_out[t].flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(fla_t.unsqueeze(0), seq_t.unsqueeze(0)).item()
                    print(f"TRACE SEQ_vs_FLA t={t} fla_std={fla_t.std():.6f} seq_std={seq_t.std():.6f} cos_sim={cos_sim:.6f} max_diff={((fla_t - seq_t).abs().max()):.6f}", flush=True)
                # Also print first token detail
                print(f"TRACE SEQ_out_t0 first10=[{', '.join(f'{v:.6f}' for v in seq_out[0].flatten()[:10].tolist())}]", flush=True)
                print(f"TRACE FLA_out_t0 first10=[{', '.join(f'{v:.6f}' for v in fla_out[0].flatten()[:10].tolist())}]", flush=True)

            if is_npu() or is_cpu():
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

        return core_attn_out
