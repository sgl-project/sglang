"""Clover GDN preview metadata and causal verify specialization."""

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.clover_verify_conv import (
    clover_parallel_causal_conv,
    supports_clover_parallel_causal_conv,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    pp_spec_stable_rows_enabled,
    select_verify_intermediate_state_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInputType


class CloverGDNAttnBackend(GDNAttnBackend):
    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._dllm_graph_block_size = max_num_tokens // max_bs

    def _capture_metadata(self, bs, req_pool_indices, forward_mode, spec_info):
        preview = forward_mode.is_dllm_extend()
        metadata = super()._capture_metadata(
            bs,
            req_pool_indices,
            ForwardMode.DECODE if preview else forward_mode,
            spec_info,
        )
        if preview:
            metadata.query_start_loc.copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                * self._dllm_graph_block_size
            )
        return metadata

    def _replay_metadata(
        self,
        bs,
        req_pool_indices,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding=None,
        in_capture=False,
        mamba_track_indices=None,
    ):
        preview = forward_mode.is_dllm_extend()
        # Reuse slot translation/padding; preview offsets span a whole block,
        # including padded rows, whose state index is the -1 sentinel.
        metadata = super()._replay_metadata(
            bs,
            req_pool_indices,
            ForwardMode.DECODE if preview else forward_mode,
            spec_info,
            seq_lens_cpu,
            num_padding,
            in_capture,
            mamba_track_indices,
        )
        if preview:
            metadata.query_start_loc.copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                * self._dllm_graph_block_size
            )
        return metadata

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        spec = forward_batch.spec_info
        # This specialization owns only Clover's linear verify chain. Other
        # modes and tree verification retain their original path.
        if not (
            forward_batch.forward_mode.is_target_verify()
            and spec is not None
            and spec.spec_input_type == SpecInputType.CLOVER_VERIFY
            and isinstance(
                self.kernel_dispatcher.verify_kernel,
                (TritonGDNKernel, FlashInferGDNKernel),
            )
            and mixed_qkv.is_cuda
            and mixed_qkv.dtype == torch.bfloat16
            and spec.draft_token_num in (16, 32)
            and self.mis_metadata is None
            and self.forward_metadata.retrieve_parent_token is None
        ):
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        width = spec.draft_token_num
        batch_size = mixed_qkv.shape[0] // width
        packed = mixed_qkv.view(batch_size, width, -1)
        conv = cache.conv[0]
        windows = cache.intermediate_conv_window[0]
        state = cache.temporal
        snapshots = cache.intermediate_ssm
        if not (
            snapshots is not None
            and snapshots.shape[1] == width
            and supports_clover_parallel_causal_conv(
                packed.transpose(1, 2), conv, layer.conv_weights, windows
            )
        ):
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        slots = self.forward_metadata.mamba_cache_indices[:batch_size]
        snapshot_slots = self.verify_intermediate_state_indices
        if pp_spec_stable_rows_enabled():
            snapshot_slots = select_verify_intermediate_state_indices(
                snapshot_slots,
                forward_batch.req_pool_indices,
                slots >= 0,
                self.req_to_token_pool.size,
            )
        snapshot_slots = snapshot_slots[:batch_size]
        packed = clover_parallel_causal_conv(
            packed.transpose(1, 2),
            conv,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=slots,
            intermediate_conv_window=windows,
            intermediate_state_indices=snapshot_slots,
            update_state=False,
        ).transpose(1, 2)
        # Use the selected generic recurrence after parallel convolution.
        # Triton accepts strided Q/K/V views, avoiding a split kernel.
        if isinstance(self.kernel_dispatcher.verify_kernel, TritonGDNKernel):
            query, key, value = packed.split(
                [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            query = query.reshape(
                1, batch_size * width, layer.num_q_heads, layer.head_q_dim
            )
            key = key.reshape(
                1, batch_size * width, layer.num_k_heads, layer.head_k_dim
            )
            value = value.reshape(
                1, batch_size * width, layer.num_v_heads, layer.head_v_dim
            )
        else:
            query, key, value = fused_qkv_split_gdn_prefill(
                packed.reshape(batch_size * width, -1),
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        return self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            ssm_states=state,
            cache_indices=slots,
            query_start_loc=self.forward_metadata.query_start_loc,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=snapshot_slots,
            cache_steps=width,
            retrieve_parent_token=None,
            stable_rows=pp_spec_stable_rows_enabled(),
        )
