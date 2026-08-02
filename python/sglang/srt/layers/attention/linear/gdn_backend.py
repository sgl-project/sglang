from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_tree_verify import (
    advance_ssm_states_along_accept_paths,
    build_tree_ancestor_masks,
    chunk_gated_delta_rule_tree_verify,
    derive_tree_parent_tokens,
    tree_mask_capacity,
    tree_verify_conv1d_update,
)
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd, l2norm_fwd_strided
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    gdn_chunk_tree_verify_enabled,
    gdn_dflash_tfm_tree_verify_enabled,
    gdn_fused_tree_verify_enabled,
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
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip():
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill

MAX_FUSED_QKV_SPLIT_DIM = 8192

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

        # Verify kernel: use FlashInfer when the selected FlashInfer kernel
        # supports MTP verify. SM90 uses the fp32-state path; SM100 uses the
        # bf16-state adapter in FlashInferGDNKernel.
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

    needs_cpu_seq_lens: bool = False

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
        self.chunk_tree_verify_active = (
            gdn_chunk_tree_verify_enabled(model_runner.server_args)
            and not model_runner.is_draft_worker
        )
        self.fused_tree_verify_active = (
            self.chunk_tree_verify_active
            and gdn_fused_tree_verify_enabled(model_runner.server_args)
        )
        self.tree_verify_max_depth = None
        if self.fused_tree_verify_active:
            block_size = model_runner.server_args.speculative_dflash_block_size
            assert block_size is not None
            self.tree_verify_max_depth = int(block_size) - 1
        if (
            model_runner.server_args.speculative_gdn_verify_kernel == "chunk"
            and not model_runner.is_draft_worker
            and not self.chunk_tree_verify_active
            and (model_runner.server_args.speculative_num_draft_tokens or 0) > 0
        ):
            msg = (
                "speculative_gdn_verify_kernel=chunk requested but unsupported "
                f"(uses_tree_spec_inputs={self.uses_tree_spec_inputs}, "
                f"num_draft_tokens={model_runner.server_args.speculative_num_draft_tokens})."
            )
            if gdn_dflash_tfm_tree_verify_enabled(model_runner.server_args):
                raise ValueError(msg + " Refusing recurrent fallback for DFLASH_TFM.")
            rank0_log(msg + " Falling back to the recurrent verify kernel.")
        self.tree_verify_stash = None
        self.last_target_verify_used_tree = False

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
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

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
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
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
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            tree_verify_request = (
                self.chunk_tree_verify_active and retrieve_parent_token is not None
            )
            self.last_target_verify_used_tree = tree_verify_request
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            if tree_verify_request:
                self._ensure_tree_verify_step_metadata(
                    layer=layer,
                    batch_size=batch_size,
                    draft_token_num=draft_token_num,
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
                mixed_qkv_processed = tree_verify_conv1d_update(
                    mixed_qkv_reshaped,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=cache_indices[:batch_size],
                    parent_tokens=retrieve_parent_token,
                    intermediate_conv_window=intermediate_conv_window_cache,
                    intermediate_state_indices=intermediate_state_indices[:batch_size],
                )
            else:
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

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            if tree_verify_request and self.fused_tree_verify_active:
                core_attn_out = self._fused_tree_verify_forward(
                    layer=layer,
                    batch_size=batch_size,
                    draft_token_num=draft_token_num,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    retrieve_parent_token=retrieve_parent_token,
                )
            elif tree_verify_request:
                core_attn_out = self._chunk_tree_verify_forward(
                    layer=layer,
                    batch_size=batch_size,
                    draft_token_num=draft_token_num,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    retrieve_parent_token=retrieve_parent_token,
                )
            else:
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
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

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def _ensure_tree_verify_step_metadata(
        self,
        layer: RadixLinearAttention,
        batch_size: int,
        draft_token_num: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        retrieve_parent_token: torch.Tensor,
    ):
        """Once per verify step, derive tree parents and ancestor bitsets
        shared by every GDN layer's tree kernels."""
        if self.req_to_token_pool.mamba_map[layer.layer_id] != 0:
            return
        if self.tree_verify_stash is None:
            self.tree_verify_stash = TreeVerifyStash(
                num_layers=len(self.req_to_token_pool.mamba_map),
                num_slots=self.req_to_token_pool.mamba_pool.mamba_cache.intermediate_ssm.shape[
                    1
                ],
                T=draft_token_num,
                Hg=layer.num_k_heads,
                K=layer.head_k_dim,
                H=layer.num_v_heads,
                V=layer.head_v_dim,
                dtype=self.req_to_token_pool.mamba_pool.mamba_cache.conv[0].dtype,
                device=retrieve_parent_token.device,
            )
        derive_tree_parent_tokens(
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
            draft_token_num,
        )
        build_tree_ancestor_masks(
            retrieve_parent_token[:batch_size],
            draft_token_num,
            out=self.tree_verify_stash.ancestor_masks,
        )
        if self.fused_tree_verify_active:
            from sglang.srt.layers.attention.fla.gdn_tree_fused import (
                alloc_tree_structure_buffers,
                build_tree_structure_into_fast,
            )

            if getattr(self, "_fused_tree_struct", None) is None:
                assert self.tree_verify_max_depth is not None
                self._fused_tree_struct = alloc_tree_structure_buffers(
                    batch_size,
                    draft_token_num,
                    min(self.tree_verify_max_depth, draft_token_num - 1),
                    retrieve_parent_token.device,
                )
            build_tree_structure_into_fast(
                retrieve_parent_token[:batch_size].view(batch_size, draft_token_num),
                self._fused_tree_struct,
            )

    def _fused_tree_verify_forward(
        self,
        layer,
        batch_size: int,
        draft_token_num: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        retrieve_parent_token: torch.Tensor,
    ) -> torch.Tensor:
        """Verify draft trees with the fused K0-K3 lag-folding kernels.

        Same stash contract as _chunk_tree_verify_forward: writes no SSM
        state; per-layer (k, v, g, beta) are stashed so the commit replays
        the accept path. Output kernel differs only.
        """
        from sglang.srt.layers.attention.fla.gdn_tree_triton import (
            tree_gdn_triton_verify,
        )

        bs, T = batch_size, draft_token_num
        H, V = layer.num_v_heads, layer.head_v_dim
        Hg, K = layer.num_k_heads, layer.head_k_dim

        stash = self.tree_verify_stash
        assert stash is not None and T == stash.T and bs <= stash.k.shape[1]
        layer_ordinal = self.req_to_token_pool.mamba_map[layer.layer_id]

        # Gating, l2norm and the value copy land directly in the stash: the
        # verify kernel reads the stash-backed views and the commit replays
        # from the same memory, so the four per-layer stash copies vanish.
        # l2norm reads the strided projection slices in place, so q/k need no
        # contiguous() copies either.
        fused_gdn_gating(
            layer.A_log,
            a,
            b,
            layer.dt_bias,
            out_g=stash.g[layer_ordinal, :bs],
            out_beta=stash.beta[layer_ordinal, :bs],
        )
        query = l2norm_fwd_strided(query)
        key = l2norm_fwd_strided(
            key, out=stash.k[layer_ordinal, :bs].view(bs * T, Hg * K)
        ).view(key.shape)
        stash_v = stash.v[layer_ordinal, :bs].view(bs, T, H, V)
        stash_v.copy_(value.view(bs, T, H, V))
        value = stash_v

        tree = self._fused_tree_struct
        assert tree is not None, "fused tree structure metadata missing"
        o = tree_gdn_triton_verify(
            layer.A_log,
            a.view(bs, T, H),
            layer.dt_bias,
            1.0,
            20.0,
            query.view(bs, T, Hg, K),
            key.view(bs, T, Hg, K),
            value.view(bs, T, H, V),
            b.view(bs, T, H),
            ssm_states,
            cache_indices[:bs].long(),
            tree,
            use_qk_l2norm_in_kernel=False,
            precision="tf32",
        )
        return o.view(1, bs * T, H, V)

    def _chunk_tree_verify_forward(
        self,
        layer: RadixLinearAttention,
        batch_size: int,
        draft_token_num: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        retrieve_parent_token: torch.Tensor,
    ) -> torch.Tensor:
        """Verify draft trees with the block-quadratic kernel.

        Writes no SSM state: per-layer (k, v, g, beta) are stashed so that
        advance_ssm_states_after_verify can replay the accept path once the
        verified tokens are known.
        """
        bs, T = batch_size, draft_token_num
        H, V = layer.num_v_heads, layer.head_v_dim
        Hg, K = layer.num_k_heads, layer.head_k_dim

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        # q/k/v are views of the fused qkv projection.
        query = l2norm_fwd(query.contiguous())
        key = l2norm_fwd(key.contiguous())
        value = value.contiguous()

        stash = self.tree_verify_stash
        assert stash is not None and T == stash.T and bs <= stash.k.shape[1]
        layer_ordinal = self.req_to_token_pool.mamba_map[layer.layer_id]

        stash.k[layer_ordinal, :bs].copy_(key.view(bs, T, Hg * K))
        stash.v[layer_ordinal, :bs].copy_(value.view(bs, T, H * V))
        stash.g[layer_ordinal, :bs].copy_(g.view(bs, T, H))
        stash.beta[layer_ordinal, :bs].copy_(beta.view(bs, T, H))

        o = chunk_gated_delta_rule_tree_verify(
            q=query.view(bs, T, Hg, K),
            k=key.view(bs, T, Hg, K),
            v=value.view(bs, T, H, V),
            g=g.view(bs, T, H),
            beta=beta.view(bs, T, H),
            ancestor_masks=stash.ancestor_masks[:bs],
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices[:bs],
            scale=layer.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=False,
        )
        return o.view(1, bs * T, H, V)

    def advance_ssm_states_after_verify(
        self,
        last_correct_steps: torch.Tensor,
        track_slots: Optional[torch.Tensor],
        track_steps: Optional[torch.Tensor],
    ):
        """Commit SSM states along each request's accepted tree path."""
        stash = self.tree_verify_stash
        assert stash is not None, "no tree verify forward preceded the state commit"
        bs = last_correct_steps.shape[0]
        advance_ssm_states_along_accept_paths(
            k_stash=stash.k,
            v_stash=stash.v,
            g_stash=stash.g,
            beta_stash=stash.beta,
            ssm_states=self.req_to_token_pool.mamba_pool.mamba_cache.temporal,
            cache_indices=self.forward_metadata.mamba_cache_indices[:bs],
            ancestor_masks=stash.ancestor_masks,
            last_correct_steps=last_correct_steps,
            T=stash.T,
            Hg=stash.Hg,
            K=stash.K,
            V=stash.V,
            track_steps=track_steps,
            track_slots=track_slots,
        )


class TreeVerifyStash:
    """Per-layer verify activations kept alive for post-accept state advance."""

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        T: int,
        Hg: int,
        K: int,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.T, self.Hg, self.K, self.V = T, Hg, K, V
        self.k = torch.zeros(
            num_layers, num_slots, T, Hg * K, dtype=dtype, device=device
        )
        self.v = torch.zeros(num_layers, num_slots, T, H * V, dtype=dtype, device=device)
        self.g = torch.zeros(
            num_layers, num_slots, T, H, dtype=torch.float32, device=device
        )
        self.beta = torch.zeros(
            num_layers, num_slots, T, H, dtype=torch.float32, device=device
        )
        BT = tree_mask_capacity(T)
        self.ancestor_masks = torch.zeros(
            num_slots, BT, BT // 64, dtype=torch.int64, device=device
        )
