from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
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
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashkda():
            from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
                FlashKDAKernel,
            )

            self.extend_kernel = FlashKDAKernel()
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            cutedsl_kernel = CuteDSLKDAKernel()
            if getattr(cutedsl_kernel, "supports_prefill", False):
                # SM100 chunk prefill pipeline.
                self.extend_kernel = cutedsl_kernel
            else:
                # CuTe DSL prefill kernels need SM100 (Blackwell); on older GPUs
                # fall back to the Triton chunk kernel.
                self.extend_kernel = triton_kernel
                rank0_log(
                    "KDA cutedsl prefill needs SM100; falling back to Triton extend."
                )
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton', 'flashkda', or 'cutedsl' "
                "(cutedsl prefill needs SM100)."
            )

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
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
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if the decode
        kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
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
    ):
        """Returns ``core_attn_out``, or ``(core_attn_out, h)`` when the caller
        passes ``output_intermediate_states=True`` (extra_buffer prefix-cache
        tracking; only the Triton prefill kernel supports it)."""
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


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)
        # KDA raw conv[0] is (K-1, channel) (KimiLinearStateShape swaps the axes vs
        # GDN's (channel, K-1)). The shared extra_buffer track code
        # (_init_track_conv_indices) reads conv_states_shape[-1] as the K-1 window
        # length, and forward_extend snapshots into the transposed (channel, K-1)
        # conv view. Store the TRANSPOSED shape so [-1] == K-1 for both backends.
        conv_shape = model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        self.conv_states_shape = (conv_shape[-1], conv_shape[-2])

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        # Physical track-dest slots for the conv-window snapshot (extra_buffer
        # prefix caching). Mirrors GDNAttnBackend.init_forward_metadata.
        if self.forward_metadata.has_mamba_track_mask:
            mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            self.forward_metadata.mamba_track_mask_indices = mask_indices
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mask_indices]
            )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
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

        # ReplaySSM ring: per-layer ring slices + the once-per-forward per-row
        # write cursor. All None unless --enable-linear-replayssm, so packed_decode
        # falls through to the byte-identical legacy KDA path. KDA ships WITHOUT
        # radix coordination for now, so force_flush is None/zeroed (the ring
        # flushes only at the natural write_pos == L-1 wrap; set in the shared
        # HybridLinearAttn metadata, which zeroes force_flush for KDA models).
        # NOTE: ReplaySSM decode is a GDN (scalar-gate) bandwidth win; on KDA the
        # per-K g_cache is K x larger and the reconstruction refolds the per-K
        # decay every step, so it is correct but SLOWER than packed (a measured
        # decode regression). Kept wired for correctness + the spec-decode path;
        # not recommended for KDA decode. Revisit on Blackwell (more tensor-core
        # throughput may flip the compute/bandwidth tradeoff).
        replayssm_write_pos = getattr(
            self.forward_metadata, "replayssm_write_pos", None
        )
        replayssm_force_flush = getattr(
            self.forward_metadata, "replayssm_force_flush", None
        )
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape by consuming the packed mixed_qkv directly in a
        # single fused Triton kernel (KDA per-K gate variant of GDN PR #20627).
        #
        # The packed kernel hard-assumes one token per sequence (T=1): it has no
        # query_start_loc / per-sequence loop. forward_decode is only entered in
        # decode mode (see HybridLinearAttnBackend.forward dispatch), where each
        # request contributes exactly one token, so #tokens == #requests. Multi-
        # token-per-seq speculative paths (target_verify / draft_extend) go
        # through forward_extend instead. Assert the invariant so a future
        # routing change fails loudly rather than silently corrupting state.
        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            return self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        return self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        mixed_qkv_t = mixed_qkv.transpose(0, 1)  # [seq, channel] -> [channel, seq]
        # extra_buffer prefix caching: snapshot the pre-conv window (the last
        # conv_state_len positions up to the tracked chunk boundary) into the conv
        # pool for the radix-tracked rows. `conv_states` is the transposed
        # [slots, channel, K-1] view (KDA's raw conv[0] is [slots, K-1, channel]),
        # so this matches GDN's [channel, num_tracked] -> [num_tracked, channel]
        # snapshot exactly.
        if self.forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv_t[
                :, self.forward_metadata.track_conv_indices
            ].transpose(0, 1)
            conv_states[self.forward_metadata.conv_states_mask_indices] = (
                mixed_qkv_to_track
            )
        q, k, v = mixed_qkv_t.split(splits, dim=0)
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        q = causal_conv1d_fn(
            q,
            q_conv_weight,
            q_bias,
            activation="silu",
            conv_states=q_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        k = causal_conv1d_fn(
            k,
            k_conv_weight,
            k_bias,
            activation="silu",
            conv_states=k_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        v = causal_conv1d_fn(
            v,
            v_conv_weight,
            v_bias,
            activation="silu",
            conv_states=v_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=getattr(layer, "lower_bound", None),
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            # target_verify / draft_extend_v2 also reach forward_extend; they must
            # stay rollback-able, so a kernel that commits state in place (e.g.
            # FlashKDA) must not run for them.
            is_spec_decode=(
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ),
            # extra_buffer prefix caching needs the per-chunk-boundary intermediate
            # SSM state `h`; only the Triton chunk_kda path exposes it (the fused
            # cutedsl / FlashKDA kernels assert against this flag).
            output_intermediate_states=self.forward_metadata.has_mamba_track_mask,
        )

        # When tracking, the dispatcher returns (core_attn_out, h); otherwise a
        # bare tensor. Snapshot h at the tracked chunk boundaries for radix reuse.
        if self.forward_metadata.has_mamba_track_mask:
            core_attn_out, h = core_attn_out
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, self.forward_metadata
            )

        return core_attn_out
