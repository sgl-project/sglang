from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

    SIMPLE_GLA_AVAILABLE = True
except ImportError:
    SIMPLE_GLA_AVAILABLE = False


def _build_slope_tensor(nheads: int) -> torch.Tensor:
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    return torch.tensor(get_slopes(nheads))


class SimpleGLAAttnBackend(MambaAttnBackendBase):
    """SimpleGLA state backend for MiniCPM hybrid models."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.model_config.hf_config
        self.conv_states_shape = None

        tp_size = get_parallel().attn_tp_size
        total_num_heads = config.lightning_nkv or 16
        assert total_num_heads % tp_size == 0, (
            f"lightning_nkv ({total_num_heads}) must be divisible by tp_size "
            f"({tp_size})"
        )
        self.num_heads = total_num_heads // tp_size
        self.g_gamma = -_build_slope_tensor(self.num_heads).to(
            dtype=torch.float32, device=self.device
        )

        head_dim = config.lightning_head_dim
        scale_config = config.lightning_scale
        if scale_config == "1/sqrt(d)":
            self.scale = head_dim**-0.5
        elif scale_config == "1/d":
            self.scale = head_dim**-1.0
        else:
            self.scale = 1.0

        if not SIMPLE_GLA_AVAILABLE:
            raise ImportError(
                "Simple GLA requires the 'flash-linear-attention' package."
            )

    def _get_mamba_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        if (
            self.forward_metadata is not None
            and self.forward_metadata.mamba_cache_indices is not None
        ):
            return self.forward_metadata.mamba_cache_indices
        return self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        return None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        seq_len = (
            1
            if forward_batch.forward_mode.is_decode()
            else torch.max(forward_batch.extend_seq_lens)
        )

        mamba_indices = self._get_mamba_indices(forward_batch)
        initial_state = None
        has_initial_state = (
            forward_batch.extend_prefix_lens is not None
            and forward_batch.extend_prefix_lens > 0
        )
        if forward_batch.forward_mode.is_decode() or has_initial_state.any():
            cache_idx = self.req_to_token_pool.mamba_map.get(layer_id)
            if cache_idx is None:
                raise RuntimeError(
                    f"SimpleGLA layer {layer_id} is missing from mamba_map"
                )
            layer_cache = self.req_to_token_pool.mamba_pool.mamba2_layer_cache(
                cache_idx
            )
            initial_state = layer_cache.temporal[mamba_indices, :].contiguous()

        mode = "fused_recurrent" if seq_len < 64 else "chunk"
        if forward_batch.forward_mode.is_decode() or mode == "fused_recurrent":
            o, final_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=self.g_gamma,
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=self.forward_metadata.query_start_loc,
            )
        else:
            o, final_state = chunk_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=self.g_gamma,
                initial_state=initial_state,
                output_final_state=True,
                scale=self.scale,
                cu_seqlens=self.forward_metadata.query_start_loc,
            )

        if final_state is not None:
            cache_idx = self.req_to_token_pool.mamba_map.get(layer_id)
            if cache_idx is None:
                raise RuntimeError(
                    f"SimpleGLA layer {layer_id} is missing from mamba_map"
                )
            layer_cache = self.req_to_token_pool.mamba_pool.mamba2_layer_cache(
                cache_idx
            )
            layer_cache.temporal[self._get_mamba_indices(forward_batch), :] = (
                final_state
            )

        return o.reshape(-1, num_heads * head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)
