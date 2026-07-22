from typing import Optional, Tuple, Union

import torch

from sglang.jit_kernel import kda_fused_decode
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
    get_linear_attn_verify_backend,
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
        verify_backend: LinearAttnKernelBackend,
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
        elif decode_backend.is_flashinfer():
            # FlashInfer recurrent_kda: SM100 decode + MTP (target_verify).
            # Prefill stays on Triton / CuTe DSL (FlashInfer has no KDA chunk kernel).
            if not is_cuda():
                raise ValueError("KDA FlashInfer backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
                FlashInferKDAKernel,
            )

            self.decode_kernel = FlashInferKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA supports 'triton', 'cutedsl', or 'flashinfer'."
            )

        # target_verify kernel, selected via --linear-attn-verify-backend (defaults
        # to follow decode: flashinfer -> recurrent_kda, else triton).
        #   triton: fused chain + tree (retrieve_parent_token) verify; the reference
        #     the KDA correctness tests assert against.
        #   flashinfer: recurrent_kda (SM100, chain only); reuses the decode kernel
        #     when decode is also flashinfer.
        if verify_backend.is_triton():
            self.verify_kernel = triton_kernel
        elif verify_backend.is_flashinfer():
            if decode_backend.is_flashinfer():
                self.verify_kernel = self.decode_kernel
            else:
                if not is_cuda():
                    raise ValueError("KDA FlashInfer verify backend requires CUDA")
                from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
                    FlashInferKDAKernel,
                )

                self.verify_kernel = FlashInferKDAKernel()
        elif verify_backend.is_custom():
            # Future custom KDA verify kernel plugs in here.
            raise NotImplementedError(
                "--linear-attn-verify-backend custom: no custom KDA verify kernel "
                "is registered yet."
            )
        else:
            raise ValueError(
                f"Unsupported KDA verify backend: {verify_backend}. "
                "KDA verify supports 'triton' or 'flashinfer' "
                "(CuTe DSL has no verify kernel)."
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
        elif prefill_backend.is_nvidia_kda():
            if not is_cuda():
                raise ValueError("NVIDIA KDA prefill backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_nvidia import (
                NvidiaKDAKernel,
            )

            nvidia_kda_kernel = NvidiaKDAKernel()
            if nvidia_kda_kernel.supports_prefill:
                self.extend_kernel = nvidia_kda_kernel
            else:
                self.extend_kernel = triton_kernel
                rank0_log(
                    "NVIDIA KDA prefill needs SM100; falling back to Triton extend."
                )
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton', 'flashkda', 'cutedsl', or 'nvidia_kda' "
                "(cutedsl/nvidia_kda prefill need SM100)."
            )

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}, "
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
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MTP / speculative-decode verify, routed to ``self.verify_kernel``
        (FlashInfer decode -> recurrent_kda; Triton / CuTe DSL decode -> the Triton
        fused KDA verify)."""
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
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
    ) -> torch.Tensor:
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


def ragged_verify_dense_scatter_indices(
    *,
    query_start_loc: torch.Tensor,
    seq_len: int,
    draft_token_num: int,
) -> torch.Tensor:
    """Dense [bs, draft_token_num] slot index per packed ragged-verify token.

    Rows never exceed draft_token_num under either layout variant (cap for
    graph replay, planner construction for eager -- see
    RaggedVerifyLayout.padded_to_bucket), so in-row offsets stay in-row;
    tokens past the layout's coverage collapse into one ghost row at index
    bs * draft_token_num.
    """
    batch_size = query_start_loc.shape[0] - 1
    token_pos = torch.arange(seq_len, device=query_start_loc.device, dtype=torch.int32)
    token_slots = torch.searchsorted(query_start_loc[1:], token_pos, right=True)
    return (
        token_slots * draft_token_num
        + (token_pos - query_start_loc[token_slots]).to(torch.int64)
    ).clamp_(max=batch_size * draft_token_num)


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    # The verify kernel is varlen and the conv path scatters ragged tokens
    # to its dense layout, so ragged verify graphs are supported.
    supports_ragged_verify_graph: bool = True

    # Read by decide_needs_cpu_seq_lens. Decode/verify metadata is GPU-only
    # (graph replay already passes seq_lens_cpu=None), extend reads
    # extend_seq_lens_cpu from schedule, mamba track indices rebuild from req
    # objects, and the replayssm seq_lens_cpu force-flush is GDN-only.
    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # Needed by the extra_buffer track path: _init_track_conv_indices reads
        # conv_states_shape[-1] as the conv window length (kernel_size - 1).
        # The KDA pool stores conv states as [kernel-1, dim] — transposed vs
        # Mamba2/GDN's [dim, kernel-1] — so expose the transposed shape here.
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
            .transpose(-1, -2)
            .shape
        )
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        verify_backend = get_linear_attn_verify_backend()
        # KDA FlashInfer target_verify (recurrent_kda) is chain-only (no tree-ancestor
        # traversal). Reject EAGLE tree verify (topk > 1) early at setup, keyed on the
        # verify backend (not decode). The kernel keeps a per-call
        # retrieve_parent_token backstop that also covers ngram tree.
        speculative_topk = model_runner.server_args.speculative_eagle_topk or 1
        if verify_backend.is_flashinfer() and speculative_topk > 1:
            raise ValueError(
                "KDA FlashInfer speculative decoding only supports topk=1 "
                "(EAGLE tree verify / retrieve_parent_token is unsupported)."
            )
        self.kernel_dispatcher = KDAKernelDispatcher(
            decode_backend, prefill_backend, verify_backend
        )
        # Per-request row index into the speculative `intermediate_ssm` scratch,
        # used by the MTP / target_verify path (mirrors GDNAttnBackend).
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

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

        # ReplaySSM is mostly a GDN bandwidth optimization. It remains wired for
        # KDA correctness paths, but packed decode is faster for KDA today.
        replayssm_write_pos = getattr(
            self.forward_metadata, "replayssm_write_pos", None
        )
        replayssm_force_flush = getattr(
            self.forward_metadata, "replayssm_force_flush", None
        )
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        # Fully fused decode step: conv1d update + delta-rule recurrence +
        # gated RMSNorm in one kernel. Engages only when the model handed off
        # the output-norm gate for this forward (attempt-and-verify stash,
        # see kimi_k3.py) and the shapes are covered; the model applies the
        # norm itself whenever the stash is left unconsumed.
        if kda_fused_decode.KDA_FUSED_DECODE_ENABLED and replayssm_d is None:
            fused_static = getattr(layer, "_k3_fused_decode_args", None)
            onorm_gate = getattr(layer, "_k3_onorm_gate", None)
            if (
                fused_static is not None
                and onorm_gate is not None
                and mixed_qkv.shape[0] == cache_indices.shape[0]
                and b.ndim == 3
                and kda_fused_decode.covered(
                    mixed_qkv,
                    a,
                    b[0],
                    conv_states,
                    ssm_states,
                    cache_indices,
                    onorm_gate,
                )
            ):
                w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_w, onorm_eps = fused_static
                core_attn_out = kda_fused_decode.kda_fused_decode(
                    mixed_qkv,
                    a,
                    b[0],
                    conv_states,
                    w_q_t,
                    w_k_t,
                    w_v_t,
                    conv_bias,
                    a_log,
                    layer.dt_bias,
                    onorm_gate,
                    onorm_w,
                    ssm_states,
                    cache_indices,
                    scale=layer.head_k_dim**-0.5,
                    onorm_eps=onorm_eps,
                    lower_bound=layer.lower_bound,
                )
                layer._k3_onorm_consumed = True
                self._track_mamba_state_decode(
                    forward_batch,
                    conv_states,
                    ssm_states,
                    cache_indices,
                    layer.layer_id,
                )
                return core_attn_out
            elif fused_static is not None and onorm_gate is not None:
                # One-shot diagnostics: the model offered the handoff but the
                # runtime shapes were rejected (decode stays on the unfused
                # chain, which is correct but slower).
                if not getattr(KDAAttnBackend, "_fused_reject_logged", False):
                    KDAAttnBackend._fused_reject_logged = True
                    rank0_log(
                        "KDA fused decode rejected by covered(): "
                        f"mixed_qkv {tuple(mixed_qkv.shape)}/{mixed_qkv.dtype} "
                        f"stride {mixed_qkv.stride()}, "
                        f"conv_states {tuple(conv_states.shape)} "
                        f"stride {conv_states.stride()}, "
                        f"ssm_states {tuple(ssm_states.shape)}/{ssm_states.dtype}, "
                        f"b {tuple(b.shape)}, indices {cache_indices.dtype}"
                    )

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        # The packed kernel assumes one token per request. Assert the dispatch
        # invariant before taking the fused path.
        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            core_attn_out = self.kernel_dispatcher.packed_decode(
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
                lower_bound=layer.lower_bound,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return core_attn_out

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        core_attn_out = self.kernel_dispatcher.decode(
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
            lower_bound=layer.lower_bound,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
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
        # MTP / speculative-decode verify is a multi-token-per-seq path with
        # per-step state checkpointing + central rollback; handled separately.
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(layer, forward_batch, mixed_qkv, a, b)

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        if self.forward_metadata.has_mamba_track_mask:
            # Snapshot the conv sliding window at the last track-aligned chunk
            # boundary into the ping-pong track slots (the prefix-cache restore
            # source). The KDA pool stores conv states as [kernel-1, dim], so
            # rows of the raw [tokens, dim] pre-conv input index in directly
            # (GDN needs a transpose here; KDA does not).
            mamba_cache_params.conv[0][
                self.forward_metadata.conv_states_mask_indices
            ] = mixed_qkv[self.forward_metadata.track_conv_indices]

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
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

        track_ssm = self.forward_metadata.has_mamba_track_mask
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
            lower_bound=layer.lower_bound,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            # draft_extend_v2 must stay rollback-able, so kernels that commit state
            # in place (e.g. FlashKDA) must not run for it.
            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),
            return_intermediate_states=track_ssm,
            # Which global chunk rows of h the track snapshot will read; lets
            # kernels that cannot materialize per-chunk states (NVIDIA KDA) take the
            # fast path when the snapshot only needs the final state.
            track_ssm_h_src=(
                self.forward_metadata.track_ssm_h_src if track_ssm else None
            ),
        )
        if track_ssm:
            # Snapshot the SSM state at the last track-aligned chunk boundary
            # from the kernel's per-chunk states (h) / final states into the
            # ping-pong track slots (see _init_track_ssm_indices).
            core_attn_out, h = core_attn_out
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, self.forward_metadata
            )

        return core_attn_out

    def _forward_target_verify(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        """MTP / speculative-decode verify (topk=1), mirroring the GDN backend.

        Conv1d runs per draft token with intermediate-window checkpointing; the
        SSM verify kernel writes each draft token's post-state into the
        speculative `intermediate_ssm` scratch so the central post-verify rollback
        (update_mamba_state_after_mtp_verify) can commit the accepted-length state.
        """
        fm = self.forward_metadata
        seq_len = mixed_qkv.shape[0]
        query_start_loc = fm.query_start_loc
        cache_indices = fm.mamba_cache_indices
        retrieve_next_token = fm.retrieve_next_token
        retrieve_next_sibling = fm.retrieve_next_sibling
        retrieve_parent_token = fm.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        intermediate_state_cache = getattr(mamba_cache_params, "intermediate_ssm", None)
        # ReplaySSM: intermediate_ssm is intentionally None (the ring + commit-time
        # fold replace the per-step snapshots). Pass None to the verify kernel so it
        # skips the write (CACHE_INTERMEDIATE_STATES=False); the output is unaffected.
        replayssm_on = getattr(mamba_cache_params, "replayssm_rawv", None) is not None
        if intermediate_state_cache is None and not replayssm_on:
            raise RuntimeError(
                "KDA target_verify requires a speculative mamba cache "
                "(MambaPool.SpeculativeState); none found."
            )
        intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
        intermediate_state_indices = self.verify_intermediate_state_indices

        draft_token_num = forward_batch.spec_info.draft_token_num
        ragged_layout = forward_batch.spec_info.ragged_verify_layout
        if ragged_layout is None:
            batch_size = seq_len // draft_token_num
            dense_token_indices = None
            mixed_qkv_dense = mixed_qkv.view(batch_size, draft_token_num, -1)
        else:
            # Conv update and its per-step scratch want the dense
            # [bs, draft_token_num] layout: scatter ragged tokens to their
            # step slots, gather back after (pad steps are never committed).
            # Uncovered tier-leftover tokens land in the ghost row (see
            # ragged_verify_dense_scatter_indices); their values are pad
            # garbage, discarded downstream (ghost collisions are
            # value-irrelevant).
            batch_size = query_start_loc.shape[0] - 1
            num_dense_tokens = batch_size * draft_token_num
            dense_token_indices = ragged_verify_dense_scatter_indices(
                query_start_loc=query_start_loc,
                seq_len=seq_len,
                draft_token_num=draft_token_num,
            )
            dense = mixed_qkv.new_zeros(num_dense_tokens + 1, mixed_qkv.shape[-1])
            dense.index_copy_(0, dense_token_indices, mixed_qkv)
            mixed_qkv_dense = dense[:num_dense_tokens].view(
                batch_size, draft_token_num, -1
            )

        # causal_conv1d_update expects [.., dim, width]. KDA keeps dense conv-window
        # scratch because the deduplicated overlapping layout cannot be transposed.
        mixed_qkv_reshaped = mixed_qkv_dense.transpose(1, 2)
        mixed_qkv_processed = causal_conv1d_update(
            mixed_qkv_reshaped,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices[:batch_size],
            intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )
        mixed_qkv_flat = mixed_qkv_processed.transpose(1, 2).reshape(
            batch_size * draft_token_num, -1
        )
        if dense_token_indices is None:
            mixed_qkv = mixed_qkv_flat
        else:
            # Ghost row (zeros) so uncovered tail tokens gather finite values.
            padded_flat = mixed_qkv_flat.new_zeros(
                batch_size * draft_token_num + 1, mixed_qkv_flat.shape[-1]
            )
            padded_flat[: batch_size * draft_token_num] = mixed_qkv_flat
            mixed_qkv = padded_flat[dense_token_indices]

        q, k, v = mixed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # ReplaySSM (fold-every-commit): store the draft window's raw inputs into
        # the per-slot ring so commit replays the accepted prefix into the fp32
        # checkpoint, replacing the per-step intermediate_ssm. Dense verify only
        # (there is no intermediate_ssm fallback under replayssm); the compact/
        # ragged tier is refused at startup (see server_args).
        replayssm_rawv = getattr(mamba_cache_params, "replayssm_rawv", None)
        if replayssm_rawv is not None and ragged_layout is None:
            HVh, Kd, Vd = v.shape[2], k.shape[-1], v.shape[-1]
            T = draft_token_num
            # Under cuda graph the captured batch includes padding rows whose slot
            # is PAD_SLOT_ID (-1); a raw index_put_ with -1 device-asserts (CUDA
            # bounds-checks without wrapping). Clamp to slot 0 -- the reserved null
            # slot (allocator hands out 1..size), never read by any real commit --
            # so padding rows write harmlessly. Real slots (>=1) are unchanged.
            slots = cache_indices[:batch_size].clamp(min=0).to(torch.long)
            kk = k[0].reshape(batch_size, T, HVh, Kd)  # pre-norm k
            vv = v[0].reshape(batch_size, T, HVh, Vd)
            aa = a[0].reshape(batch_size, T, HVh, Kd)
            bb = b[0].reshape(batch_size, T, HVh)
            a_log = layer.A_log.reshape(HVh)
            dtb = layer.dt_bias.reshape(HVh, Kd)
            # gate MUST match the verify kernel's two branches exactly
            # (fused_sigmoid_gating_recurrent): safe gate when lower_bound is set
            # (K3), plain softplus otherwise. A mismatch silently commits the
            # wrong state.
            x = aa.float() + dtb.reshape(1, 1, HVh, Kd)
            exp_a_log = torch.exp(a_log).reshape(1, 1, HVh, 1)
            if layer.lower_bound is not None:
                gk = layer.lower_bound * torch.sigmoid(exp_a_log * x)
            else:
                gk = (-exp_a_log) * torch.nn.functional.softplus(x)
            beta = torch.sigmoid(bb.float())
            replayssm_rawv[slots, :, :T] = vv.transpose(1, 2).to(replayssm_rawv.dtype)
            mamba_cache_params.replayssm_rawk[slots, :, :T] = kk.transpose(1, 2).to(
                mamba_cache_params.replayssm_rawk.dtype
            )
            mamba_cache_params.replayssm_g[slots, :, :T] = gk.transpose(1, 2)
            mamba_cache_params.replayssm_beta[slots, :, :T] = beta.transpose(1, 2)

        core_attn_out = self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=draft_token_num,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=layer.lower_bound,
        )
        if dense_token_indices is not None:
            # Kernel output is empty-allocated and the capped qsl skips the
            # uncovered tail rows; zero them so discarded pad hidden states
            # stay finite. Uncovered == clamped-to-ghost.
            covered = dense_token_indices < (batch_size * draft_token_num)
            core_attn_out = torch.where(covered.view(1, -1, 1, 1), core_attn_out, 0.0)
        return core_attn_out
