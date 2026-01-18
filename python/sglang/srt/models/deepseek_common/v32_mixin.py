"""DeepSeek V3.2 NSA (Native Sparse Attention) mixin classes."""

import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

_NSA_AVAILABLE = False

try:
    from sglang.srt.configs.model_config import (
        get_nsa_index_head_dim,
        get_nsa_index_n_heads,
        get_nsa_index_topk,
        is_deepseek_nsa,
    )
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang.srt.layers.attention.nsa.utils import (
        can_cp_split,
        is_nsa_enable_prefill_cp,
        nsa_use_prefill_cp,
        prepare_input_dp_with_cp_dsa,
    )
    from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator

    _NSA_AVAILABLE = True
except ImportError:
    Indexer = None
    NSACPLayerCommunicator = None

    def is_deepseek_nsa(_config):
        return False

    def is_nsa_enable_prefill_cp():
        return False

    def nsa_use_prefill_cp(_forward_batch):
        return False

    def can_cp_split(_input_len, _cp_size, _use_nsa, _forward_batch):
        return False

    def prepare_input_dp_with_cp_dsa(_input_len, _cp_rank, _cp_size, _seq_lens):
        return None

    def get_nsa_index_n_heads(_config):
        return 0

    def get_nsa_index_head_dim(_config):
        return 0

    def get_nsa_index_topk(_config):
        return 0


if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer


def is_nsa_available() -> bool:
    return _NSA_AVAILABLE


class DeepseekV32AttentionMixin:
    """NSA-specific logic for the attention layer."""

    def init_nsa_attention(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        qk_rope_head_dim: int,
        q_lora_rank: Optional[int],
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[dict],
        quant_config: Optional[QuantizationConfig],
        layer_id: int,
        prefix: str,
        alt_stream: Optional[torch.cuda.Stream],
    ) -> tuple:
        """Initialize NSA components. Returns (use_nsa, nsa_enable_prefill_cp, attn_tp_rank, attn_tp_size, cp_size, indexer)."""
        use_nsa = is_deepseek_nsa(config)
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        if nsa_enable_prefill_cp:
            assert use_nsa, "CP currently only supports deepseek v3.2 model"

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        cp_size = None

        # CP reuses attn_tp comm group but duplicates weights
        if nsa_enable_prefill_cp and use_nsa:
            attn_tp_rank = 0
            attn_tp_size = 1
            cp_size = get_attention_tp_size()

        indexer = None
        if use_nsa and _NSA_AVAILABLE:
            indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_nsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )

        return use_nsa, nsa_enable_prefill_cp, attn_tp_rank, attn_tp_size, cp_size, indexer

    def forward_nsa_indexer_prefill(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        indexer: "Indexer",
    ) -> None:
        """Cache quantized keys during prefill."""
        if indexer is None:
            return
        indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            return_indices=False,
        )

    def forward_nsa_indexer_decode(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        indexer: "Indexer",
    ) -> Optional[torch.Tensor]:
        """Get top-k indices during decode."""
        if indexer is None:
            return None
        return indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
        )


class DeepseekV32DecoderLayerMixin:
    """NSA-specific logic for the decoder layer."""

    def init_nsa_layer_communicator(
        self,
        config: PretrainedConfig,
        layer_id: int,
        is_nextn: bool,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        qkv_latent_func: callable,
    ) -> tuple:
        """Initialize layer communicator. Returns (nsa_enable_prefill_cp, layer_communicator)."""
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        is_last_layer = is_nextn or (layer_id == config.num_hidden_layers - 1)

        if nsa_enable_prefill_cp and _NSA_AVAILABLE:
            layer_communicator = NSACPLayerCommunicator(
                layer_scatter_modes=layer_scatter_modes,
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=is_last_layer,
                qkv_latent_func=qkv_latent_func,
            )
        else:
            layer_communicator = LayerCommunicator(
                layer_scatter_modes=layer_scatter_modes,
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=is_last_layer,
                qkv_latent_func=qkv_latent_func,
            )

        return nsa_enable_prefill_cp, layer_communicator


class DeepseekV32ModelMixin:
    """NSA-specific logic at the model level."""

    def init_nsa_model(self, config: PretrainedConfig) -> tuple:
        """Initialize NSA model components. Returns (use_nsa, nsa_enable_prefill_cp, cp_rank, cp_size)."""
        use_nsa = is_deepseek_nsa(config)
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        if nsa_enable_prefill_cp:
            cp_rank = get_attention_tp_rank()
            cp_size = get_attention_tp_size()
        else:
            cp_rank = None
            cp_size = None

        return use_nsa, nsa_enable_prefill_cp, cp_rank, cp_size

    def prepare_nsa_forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        use_nsa: bool,
        nsa_enable_prefill_cp: bool,
        cp_rank: Optional[int],
        cp_size: Optional[int],
    ) -> None:
        """Prepare NSA metadata before forward pass."""
        if nsa_enable_prefill_cp and _NSA_AVAILABLE:
            if can_cp_split(len(input_ids), cp_size, use_nsa, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    cp_rank,
                    cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

    @staticmethod
    def should_use_nsa_prefill_cp(forward_batch: ForwardBatch) -> bool:
        return nsa_use_prefill_cp(forward_batch)
