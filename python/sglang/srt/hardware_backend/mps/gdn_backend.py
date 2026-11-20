"""Gated DeltaNet linear-attention backend for eager Torch MPS."""

from typing import Union

import torch

from sglang.kernels.ops.attention.linear.gdn_mps import (
    causal_conv1d_update_mps,
    causal_conv1d_varlen_mps,
    gated_delta_rule_mps,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def _reject_unsupported_features(model_runner: ModelRunner) -> None:
    server_args = model_runner.server_args
    unsupported = {
        "speculative decoding": not model_runner.spec_algorithm.is_none(),
        "--enable-linear-replayssm": server_args.enable_linear_replayssm,
        "--enable-linear-replayssm-spec": server_args.enable_linear_replayssm_spec,
        "--enable-int8-mamba-checkpoint": server_args.enable_int8_mamba_checkpoint,
        "--enable-mis": server_args.enable_mis,
        "--mamba-ssm-dtype values other than float32": (
            server_args.mamba_ssm_dtype not in (None, "float32")
        ),
    }
    enabled = [name for name, on in unsupported.items() if on]
    if enabled:
        raise ValueError(f"The MPS GDN backend does not support {', '.join(enabled)}")


class MpsGDNAttnBackend(MambaAttnBackendBase):
    """GDN decode and varlen extend on MPS; no speculative verify or state tracking."""

    def __init__(self, model_runner: ModelRunner):
        _reject_unsupported_features(model_runner)
        super().__init__(model_runner)
        self.conv_states_shape = self.state_pool.mamba_cache.conv[0].shape
        self._has_initial_state = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        metadata = self.forward_metadata
        # The MPS kernels take int32 slots; padded rows are negative.
        metadata.mamba_cache_indices = metadata.mamba_cache_indices.to(torch.int32)
        if forward_batch.forward_mode.is_decode_or_idle():
            self._has_initial_state = torch.ones_like(
                metadata.mamba_cache_indices, dtype=torch.bool
            )
        else:
            self._has_initial_state = forward_batch.extend_prefix_lens > 0
        # Prefix-cache checkpoints need the extra_buffer strategy, which MPS rejects.
        assert not metadata.has_mamba_track_mask, "MPS GDN keeps no state checkpoints"

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, tuple],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.state_layer_cache(layer.layer_id)
        metadata = self.forward_metadata
        cache_indices = metadata.mamba_cache_indices
        mixed_qkv = causal_conv1d_update_mps(
            mixed_qkv,
            layer_cache.conv[0],
            layer.conv_weights,
            layer.bias,
            cache_indices,
        )
        return self._gated_delta_rule(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            ssm_states=layer_cache.temporal,
            cache_indices=cache_indices,
            query_start_loc=metadata.query_start_loc,
            has_initial_state=self._has_initial_state,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, tuple],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.state_layer_cache(layer.layer_id)
        metadata = self.forward_metadata
        cache_indices = metadata.mamba_cache_indices
        has_initial_state = self._has_initial_state
        mixed_qkv = causal_conv1d_varlen_mps(
            mixed_qkv,
            layer_cache.conv[0],
            layer.conv_weights,
            layer.bias,
            cache_indices,
            metadata.query_start_loc,
            has_initial_state,
        )
        return self._gated_delta_rule(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            ssm_states=layer_cache.temporal,
            cache_indices=cache_indices,
            query_start_loc=metadata.query_start_loc,
            has_initial_state=has_initial_state,
        )

    @staticmethod
    def _gated_delta_rule(
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = mixed_qkv.shape[0]
        query, key, value = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        return gated_delta_rule_mps(
            query.view(num_tokens, layer.num_q_heads, layer.head_q_dim),
            key.view(num_tokens, layer.num_k_heads, layer.head_k_dim),
            value.view(num_tokens, layer.num_v_heads, layer.head_v_dim),
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            ssm_states,
            cache_indices,
            query_start_loc,
            has_initial_state,
        )
