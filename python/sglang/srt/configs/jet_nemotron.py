from dataclasses import dataclass
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    extra_groups_for_head_shards,
)
from sglang.srt.distributed.utils import divide


@dataclass
class JetBlockConfig:
    mode: str
    expand_v: float
    num_heads: int
    head_dim: int
    norm_eps: str
    conv_size: int
    dconv_generator_reduction: int
    dconv_implementation: str


class JetNemotronConfig(PretrainedConfig):
    model_type: str = "jet_nemotron"

    efficient_attention_config: dict[str, dict[str, Any]]
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_types: list[str]
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: None
    rope_theta: float

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type in ("attn", "swa")
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "jet"
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        jet_block_config = JetBlockConfig(**self.efficient_attention_config["jet"])

        num_heads = jet_block_config.num_heads
        head_k_dim = jet_block_config.head_dim
        head_v_dim = int(head_k_dim * jet_block_config.expand_v)
        total_v_dim = num_heads * head_v_dim

        shape = JetNemotronStateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=total_v_dim,
            n_groups=num_heads,
            num_heads=num_heads,
            head_dim=head_k_dim,
            state_size=head_v_dim,
            conv_kernel=jet_block_config.conv_size,
        )

        return Mamba2CacheParams(shape=shape, layers=self.linear_layer_ids)


@dataclass(kw_only=True, frozen=True)
class JetNemotronStateShape(Mamba2StateShape):
    @staticmethod
    def create(
        *,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> "JetNemotronStateShape":
        if n_groups % tp_world_size != 0:
            extra_groups = extra_groups_for_head_shards(n_groups, tp_world_size)
            n_groups += extra_groups
        conv_dim = intermediate_size

        conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel

        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return JetNemotronStateShape(
            conv=conv_state_shape,
            temporal=temporal_state_shape,
            intermediate_size=intermediate_size,
            conv_dim=conv_dim,
            ssm_state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )
