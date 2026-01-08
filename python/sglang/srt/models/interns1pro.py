import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer
from sglang.srt.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class InternS1ProTextAttention(Qwen3MoeAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )
        # for fope
        fope_keys = {"fope_init_factor", "fope_sep_head", "num_inv_freq"}
        use_fope = any(rope_scaling.get(key) is not None for key in fope_keys)
        if use_fope:
            rope_scaling["use_fope"] = True
            rope_scaling["num_kv_heads"] = self.num_kv_heads

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.compatible_with_fused_kv_buffer = False
        self.use_fused_qk_norm_rope = False
        self._used_fused_qk_norm_rope_last_call = False

    def forward_prepare_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        raise NotImplementedError()


class InternS1ProTextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias

        self.self_attn = InternS1ProTextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )


class InternS1ProTextModel(Qwen3MoeLLMModel):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        decoder_layer_type=InternS1ProTextDecoderLayer,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )


class InternS1ProForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=InternS1ProTextModel,
    ) -> None:
        # deal with no deepstack
        if not hasattr(config.vision_config, "deepstack_visual_indexes"):
            config.vision_config.deepstack_visual_indexes = []

        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=language_model_cls,
        )

        # disable deepstack
        if len(config.vision_config.deepstack_visual_indexes) == 0:
            self.use_deepstack = {}

    def _load_fope_weights(self, name: str, loaded_weight: torch.Tensor, params_dict):
        """load fope weights"""
        attn_tp_size = get_attention_tp_size()
        attn_tp_rank = get_attention_tp_rank()

        num_key_value_heads = loaded_weight.size(0)
        # replicate head if necessary
        if num_key_value_heads < attn_tp_size:
            n_replicate = attn_tp_size // num_key_value_heads
            attn_tp_size = num_key_value_heads
            attn_tp_rank = attn_tp_rank // n_replicate
        loaded_weight = loaded_weight.chunk(attn_tp_size, dim=0)[attn_tp_rank]

        # copy rotary_emb weights to decode layers
        for layer_idx in range(self.config.num_hidden_layers):
            param_name = name.replace(
                ".rotary_emb.", f".layers.{layer_idx}.self_attn.rotary_emb."
            )
            if param_name not in params_dict:
                continue
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights"""
        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        other_weights = dict()
        for name, loaded_weight in weights:
            if "sin_coef" in name or "cos_coef" in name:
                name = name.replace(r"model.language_model.", r"model.")
                self._load_fope_weights(name, loaded_weight, params_dict)
            else:
                other_weights[name] = loaded_weight

        super().load_weights(other_weights.items())


EntryClass = InternS1ProForConditionalGeneration
