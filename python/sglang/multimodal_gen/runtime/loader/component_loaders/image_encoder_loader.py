import types

import torch

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.models.encoders.base import finalize_encoder_folding
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _dreamzero_non_causal_clip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    qkv_states, _ = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
    query_states = query_states.reshape(
        query_states.shape[0],
        query_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )
    key_states = key_states.reshape(
        key_states.shape[0],
        key_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )
    value_states = value_states.reshape(
        value_states.shape[0],
        value_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )

    if self.attn.backend == AttentionBackendEnum.TORCH_SDPA:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(
                    dtype=query_states.dtype
                )
                attn_mask = (1.0 - attn_mask) * torch.finfo(query_states.dtype).min
            else:
                attn_mask = attention_mask
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
        )
        attn_output = attn_output.transpose(1, 2)
    else:
        attn_output = self.attn(query_states, key_states, value_states)

    attn_output = attn_output.reshape(
        attn_output.shape[0],
        attn_output.shape[1],
        self.num_heads_per_partition * self.head_dim,
    )
    attn_output, _ = self.out_proj(attn_output)
    return attn_output, None


def _patch_dreamzero_clip_vision_attention(model: torch.nn.Module) -> None:
    for layer in model.vision_model.encoder.layers:
        attention = layer.self_attn
        attention.attn.attn_impl.causal = False
        attention.forward = types.MethodType(
            _dreamzero_non_causal_clip_attention_forward,
            attention,
        )


def load_dreamzero_image_encoder(
    server_args: ServerArgs,
    component_model_path: str,
) -> torch.nn.Module:
    image_encoder = ImageEncoderLoader().load_customized(
        component_model_path,
        server_args,
        "image_encoder",
    )
    _patch_dreamzero_clip_vision_attention(image_encoder)
    return image_encoder


class ImageEncoderLoader(TextEncoderLoader):
    component_names = ["image_encoder"]
    expected_library = "transformers"

    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        should_offload = server_args.image_encoder_cpu_offload
        if not should_offload:
            return False
        # _fsdp_shard_conditions is in arch_config, not directly on model_config
        arch_config = (
            getattr(model_config, "arch_config", model_config) if model_config else None
        )
        fsdp_shard_conditions = (
            getattr(arch_config, "_fsdp_shard_conditions", []) if arch_config else []
        )
        use_cpu_offload = should_offload and len(fsdp_shard_conditions) > 0
        return use_cpu_offload

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str = "image_encoder",
        cpu_offload_flag: bool | None = None,
    ):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        model_config = get_diffusers_component_config(
            component_path=component_model_path
        )

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)
        # Keep the proposed fold group only if the encoder is wide enough
        # (image encoders are small, so this normally reverts to replicated).
        finalize_encoder_folding(encoder_config)

        # Always start with local device; load_model will adjust for offload if needed
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_model_path,
            encoder_config,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
            cpu_offload_flag=(
                cpu_offload_flag
                if cpu_offload_flag is not None
                else server_args.image_encoder_cpu_offload
            ),
        )
