from typing import Dict, Optional, Tuple, Union

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
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
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

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer only when the selected FlashInfer kernel
        # supports MTP verify. On SM100+ FlashInfer GDN decode is supported, but
        # its MTP verify path is not, so keep Triton as the verify fallback.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
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
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
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
        **kwargs,
    ) -> torch.Tensor:
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
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )
        # Per-layer persistent buffers used by gdn_mtp_cache_mode=none recovery.
        # Values are stable tensors or parameters; per-call sizes are derived
        # from runtime tensors because the dict spans multiple CUDA graph captures.
        self._no_cache_stash: Dict[int, Dict[str, torch.Tensor]] = {}
        # Option A (SGLANG_GDN_STASH_ELIM): per-layer persistent post-conv output
        # buffer, token-major [max_bs, draft, conv_dim]. The verify conv writes into
        # it (out=) so recovery reads k/v as strided views directly — no stash copy.
        self._conv_out_persist: Dict[int, torch.Tensor] = {}
        # Cached once on first forward; stable across calls.
        self._no_cache_draft_token_num: Optional[int] = None
        # Lazily-created side stream + fork/join events for SGLANG_GDN_STASH_OVERLAP:
        # run the recovery-stash copies concurrently with the WY verify kernel.
        # Created once (stable identity) so CUDA-graph replay re-uses them.
        self._stash_stream: Optional[torch.cuda.Stream] = None
        self._stash_fork_ev: Optional[torch.cuda.Event] = None
        self._stash_join_ev: Optional[torch.cuda.Event] = None

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

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

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

        # One-time bf16 pre-cast of the fp32 GDN decay/bias weights, done OUTSIDE CUDA
        # graph capture (during eager prefill warmup, which always precedes verify-graph
        # capture). The WY output-only verify kernel reads A_log/dt_bias as bf16; without
        # this, its internal fp32->bf16 cast misses the JIT cache at capture time and gets
        # baked into the captured decode graph, re-running ~1.3us serially before every WY
        # launch (1 per GDN layer per step). Pre-casting here is bit-identical to that
        # internal cast (same .to(bf16)) — only the timing moves to warmup, so accept
        # length / accuracy are unchanged. Stored on the layer (persistent, graph-stable).
        if (
            getattr(layer, "_gdn_A_log_bf16", None) is None
            and not torch.cuda.is_current_stream_capturing()
        ):
            layer._gdn_A_log_bf16 = layer.A_log.detach().to(torch.bfloat16).contiguous()
            layer._gdn_dt_bias_bf16 = (
                layer.dt_bias.detach().to(torch.bfloat16).contiguous()
            )

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
            # In cache_mode=none, target verify skips intermediate SSM writes;
            # accepted h_K is recovered after verification.
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            from sglang.srt.environ import envs

            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            # Option A: write the post-conv output into a persistent token-major buffer
            # [max_bs, draft, conv_dim] so recovery can read k/v as strided views (no
            # stash copy). The out= arg is persist[:bs].transpose(1,2), which carries the
            # same [batch, conv_dim, draft] strides empty_like(x) would have produced — so
            # the downstream transpose+view stays a no-copy view.
            _conv_out_arg = None
            if (
                intermediate_state_cache is None
                and envs.SGLANG_GDN_STASH_ELIM.get()
            ):
                conv_dim = mixed_qkv_reshaped.shape[1]
                max_bs = self.req_to_token_pool.size
                pbuf = self._conv_out_persist.get(layer.layer_id)
                if (
                    pbuf is None
                    or pbuf.shape[0] < batch_size
                    or pbuf.shape[1] != draft_token_num
                    or pbuf.shape[2] != conv_dim
                ):
                    with torch.inference_mode(False):
                        pbuf = torch.empty(
                            (max_bs, draft_token_num, conv_dim),
                            dtype=mixed_qkv_reshaped.dtype,
                            device=mixed_qkv_reshaped.device,
                        )
                    self._conv_out_persist[layer.layer_id] = pbuf
                _conv_out_arg = pbuf[:batch_size].transpose(1, 2)
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
                out=_conv_out_arg,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

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

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            from sglang.srt.environ import envs

            # Overlap the recovery stash onto a side stream (concurrent with the WY
            # verify kernel) when enabled. _stash_forked tracks whether we issued the
            # fork so the post-verify join only runs when it did.
            _stash_overlap = envs.SGLANG_GDN_STASH_OVERLAP.get()
            # Option A: skip the k/v stash copies entirely — recovery reads k/v as
            # strided views of the persistent conv-out buffer (written above via out=).
            # a/b still come from the projection (not the conv) and are stashed normally.
            _stash_elim = envs.SGLANG_GDN_STASH_ELIM.get()
            _stash_forked = False
            # In cache_mode=none, keep post-conv GDN inputs for accepted-state
            # recovery. Persistent buffers give CUDA graph replay stable addresses.
            if intermediate_state_cache is None:
                # Target layout: [1, max_tokens, heads, dim] where
                # max_tokens = pool.size * draft_token_num covers any
                # batch size SGLang may capture a CUDA graph for.
                draft_token_num = forward_batch.spec_info.draft_token_num
                max_tokens = self.req_to_token_pool.size * draft_token_num
                actual_seq_len = query.shape[1]

                stash_entry = self._no_cache_stash.get(layer.layer_id)
                if stash_entry is None or stash_entry["a"].shape[0] < max_tokens:
                    # Allocate outside inference_mode so buffers can be updated
                    # across forward invocations.
                    with torch.inference_mode(False):
                        stash_entry = {
                            "a": torch.empty(
                                (max_tokens, *a.shape[1:]),
                                dtype=a.dtype,
                                device=a.device,
                            ),
                            "b": torch.empty(
                                (max_tokens, *b.shape[1:]),
                                dtype=b.dtype,
                                device=b.device,
                            ),
                            "A_log": layer.A_log,
                            "dt_bias": layer.dt_bias,
                        }
                        if not _stash_elim:
                            stash_entry["k"] = torch.empty(
                                (key.shape[0], max_tokens, *key.shape[2:]),
                                dtype=key.dtype,
                                device=key.device,
                            )
                            stash_entry["v"] = torch.empty(
                                (value.shape[0], max_tokens, *value.shape[2:]),
                                dtype=value.dtype,
                                device=value.device,
                            )
                    self._no_cache_stash[layer.layer_id] = stash_entry
                if _stash_elim:
                    # Geometry for recovery to slice k/v out of the [B, T, conv_dim]
                    # persistent conv-out buffer. Stable per layer; set every call is fine.
                    stash_entry["conv_dims"] = (
                        layer.q_dim,
                        layer.k_dim,
                        layer.v_dim,
                        layer.num_k_heads,
                        layer.head_k_dim,
                        layer.num_v_heads,
                        layer.head_v_dim,
                    )
                # In-place slice copy with no new allocation; captured replays
                # refresh the stable buffer slice for that graph's batch size.
                if _stash_overlap:
                    # Fork the 4 stash copies onto a side stream so they run
                    # concurrently with the WY verify kernel below (both only READ the
                    # conv output; the stash writes a disjoint buffer -> no hazard).
                    # fork_ev marks the point where key/value/a/b are ready on the
                    # current stream; the side stream waits on it, then copies; the
                    # join (after target_verify) makes the current stream wait on the
                    # completion so the later eager recovery's wait_stream observes it.
                    # Capture-safe: the bracketing events record the cross-stream deps
                    # into the CUDA graph.
                    if self._stash_stream is None:
                        self._stash_stream = torch.cuda.Stream()
                        self._stash_fork_ev = torch.cuda.Event()
                        self._stash_join_ev = torch.cuda.Event()
                    self._stash_fork_ev.record()
                    self._stash_stream.wait_event(self._stash_fork_ev)
                    with torch.cuda.stream(self._stash_stream):
                        if not _stash_elim:
                            stash_entry["k"][:, :actual_seq_len].copy_(key)
                            stash_entry["v"][:, :actual_seq_len].copy_(value)
                        stash_entry["a"][:actual_seq_len].copy_(a)
                        stash_entry["b"][:actual_seq_len].copy_(b)
                        self._stash_join_ev.record(self._stash_stream)
                    _stash_forked = True
                else:
                    if not _stash_elim:
                        stash_entry["k"][:, :actual_seq_len].copy_(key)
                        stash_entry["v"][:, :actual_seq_len].copy_(value)
                    stash_entry["a"][:actual_seq_len].copy_(a)
                    stash_entry["b"][:actual_seq_len].copy_(b)
                # Do not store per-call sizes or tensor aliases in the stash;
                # eager recovery derives them from current runtime tensors.
                if self._no_cache_draft_token_num is None:
                    self._no_cache_draft_token_num = (
                        forward_batch.spec_info.draft_token_num
                    )
            # Feed the pre-cast bf16 weights to the WY (cache_mode=none) verify so its
            # internal cast hits the early-return (no per-step cast in the graph). Only
            # for WY: the full-mode state kernel reads A_log at full fp32 precision, so it
            # must keep the original fp32 weight (passing bf16 would change full-mode math).
            from sglang.srt.environ import envs
            from sglang.srt.server_args import get_global_server_args

            _use_wy_verify = (
                get_global_server_args().gdn_mtp_cache_mode == "none"
                and envs.SGLANG_GDN_WY_VERIFY.get()
                and getattr(layer, "_gdn_A_log_bf16", None) is not None
            )
            _verify_A_log = layer._gdn_A_log_bf16 if _use_wy_verify else layer.A_log
            _verify_dt_bias = layer._gdn_dt_bias_bf16 if _use_wy_verify else layer.dt_bias
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=_verify_A_log,
                dt_bias=_verify_dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
            if _stash_forked:
                # Join: the WY verify kernel has been issued on this stream, so it
                # overlapped the side-stream stash. Now make this stream wait on the
                # stash completion so the captured graph's tail — and the later eager
                # recovery's wait_stream(current) — observes the finished stash.
                torch.cuda.current_stream().wait_event(self._stash_join_ev)
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
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

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
