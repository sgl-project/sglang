import json as json_lib
import logging
import math
import os
import re
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import Llama4Config, Llama4VisionConfig
from transformers.models.llama4.modeling_llama4 import (
    Llama4MultiModalProjector,
    vision_apply_rotary_emb,
)

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Llama4VisionMLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool,
        output_activation: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        cls_fc1 = ReplicatedLinear if use_data_parallel else ColumnParallelLinear
        self.fc1 = cls_fc1(
            input_size=input_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        cls_fc2 = ReplicatedLinear if use_data_parallel else RowParallelLinear
        self.fc2 = cls_fc2(
            input_size=intermediate_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.activation_fn = nn.GELU()
        self.output_activation = output_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        if self.output_activation:
            return self.activation_fn(hidden_states)
        return hidden_states


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.size()

    reshaped_tensor = input_tensor.view(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.mlp = Llama4VisionMLP(
            input_size=config.intermediate_size,
            intermediate_size=config.projector_input_dim,
            output_size=config.projector_output_dim,
            bias=config.multi_modal_projector_bias,
            output_activation=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


def apply_position_embedding(q, k, freqs_ci, shape):
    # [batch_size_times_num_tiles, num_channels]
    input_shape = shape[:2]
    # [batch_size_times_num_tiles, num_channels, num_heads, head_dim]
    hidden_shape = (*input_shape, *q.shape[-2:])
    q = q.view(hidden_shape)
    k = k.view(hidden_shape)
    q, k = vision_apply_rotary_emb(q, k, freqs_ci)
    return q, k


class Llama4VisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.self_attn = VisionAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size,
            use_qkv_parallel=True,
            # vision_model is explicitly ignored in Maverick-17B-128E-Instruct-FP8
            quant_config=None,
            dropout=0.0,
            qkv_backend="sdpa",
            softmax_in_single_precision=False,
            flatten_batch=False,
            prefix=add_prefix("self_attn", prefix),
            qkv_bias=True,
            customized_position_embedding_applier=apply_position_embedding,
        )
        self.mlp = Llama4VisionMLP(
            input_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            output_activation=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_state: torch.Tensor,
        freqs_ci: torch.Tensor,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, position_embeddings=freqs_ci)
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        outputs = hidden_state
        return outputs


class Llama4VisionEncoder(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,  # TODO: move this to an attribute instead of keeping it around
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape
                    `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation. This is useful if you
                want more control over how to convert `input_ids` indices into
                associated vectors than the model's internal embedding
                lookup matrix.
        """

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, freqs_ci=freqs_ci)
            hidden_states = layer_outputs

        return hidden_states


class Llama4UnfoldConvolution(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        params = {
            "input_size": config.num_channels * kernel_size[0] * kernel_size[1],
            "output_size": config.hidden_size,
            "bias": False,
            "quant_config": quant_config,
            "prefix": f"{prefix}.linear",
        }
        if use_data_parallel:
            cls = ReplicatedLinear
        else:
            cls = ColumnParallelLinear
            params["gather_output"] = True
        self.linear = cls(**params)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states, _ = self.linear(hidden_states)
        return hidden_states


class Llama4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
        frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim)
        )
        freqs_x = (
            (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        freq_cis = torch.view_as_complex(
            torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        )
        self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

    def forward(self, hidden_states):
        return self.freqs_ci.to(hidden_states.device)


class Llama4VisionModel(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.patch_embedding",
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.hidden_size)
        )

        self.rotary_embedding = Llama4VisionRotaryEmbedding(config)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # encoders
        self.model = Llama4VisionEncoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.model",
        )
        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            config,
            quant_config,
            prefix=f"{prefix}.vision_adapter",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # Patch embedding
        hidden_state = self.patch_embedding(pixel_values)
        num_tiles, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        class_embedding = self.class_embedding.expand(
            hidden_state.shape[0], 1, hidden_state.shape[-1]
        )
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            num_tiles,
            1,
            num_patches,
            hidden_dim,
        )
        positional_embedding = self.positional_embedding_vlm.to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )
        hidden_state = hidden_state + positional_embedding
        hidden_state = self.layernorm_pre(hidden_state)
        hidden_state = hidden_state.view(num_tiles, -1, hidden_dim)
        freqs_ci = self.rotary_embedding(pixel_values)
        # Apply encoder
        hidden_state = self.model(hidden_state, freqs_ci=freqs_ci)
        hidden_state = self.layernorm_post(hidden_state)

        # Remove CLS token output
        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        return hidden_state


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Pattern to match language model layers only (skip vision_model and multi_modal_projector)
    lora_pattern = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"
    )

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Check if this is a text-only model (modelopt fp8 llama4 has no vision components)
        self.has_vision_weights = self._has_vision_weights(config)
        if not self.has_vision_weights:
            logger.warning(
                "No vision weights found in checkpoint. Model will run in text-only mode. "
                "Multimodal capabilities (vision understanding) will be unavailable. "
                "Please not that this warning might be inaccurate if the weights haven't been fully downloaded"
            )

        self.has_vision = (
            self.has_vision_weights and get_global_server_args().enable_multimodal
        )

        if self.has_vision:
            # TODO: make this more general
            ignore_quant_layers = getattr(config, "quantization_config", {}).get(
                "ignore", {}
            )
            if (
                "model.layers.vision_model*" in ignore_quant_layers
                and "model.layers.multi_modal_projector*" in ignore_quant_layers
            ):
                vision_quant_config = None
            else:
                vision_quant_config = quant_config
            self.vision_model = Llama4VisionModel(
                config.vision_config,
                quant_config=vision_quant_config,
                prefix=add_prefix("vision_model", prefix),
            )

            self.multi_modal_projector = Llama4MultiModalProjector(config)
        else:
            self.vision_model = None
            self.multi_modal_projector = None

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config if hasattr(config, "text_config") else config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(
            config.text_config if hasattr(config, "text_config") else config
        )
        self.padding_pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def _has_vision_weights(self, config) -> bool:
        """Check if the model has vision components by examining the checkpoint."""
        model_path = getattr(config, "_name_or_path", None)
        if not model_path:
            return False

        # Check if this is a local path first
        if os.path.isdir(model_path):
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                return self._check_vision_weights_in_index(index_file)

        # For HuggingFace models, we need to check the actual checkpoint
        # The config might say it's multimodal, but the checkpoint might be text-only
        try:
            # Try to access the HuggingFace cache directory
            from huggingface_hub import try_to_load_from_cache

            # Check if index file exists in cache
            index_file_path = try_to_load_from_cache(
                repo_id=model_path,
                filename="model.safetensors.index.json",
                cache_dir=None,
            )
            if index_file_path and os.path.exists(index_file_path):
                return self._check_vision_weights_in_index(index_file_path)

        except Exception:
            # If we can't access the cache, fall back to config-based detection
            pass

        # Fallback, assume text-only
        return False

    def _check_vision_weights_in_index(self, index_file: str) -> bool:
        """Check if the model.safetensors.index.json contains vision weights."""
        try:
            with open(index_file, "r") as f:
                index_data = json_lib.load(f)

            vision_patterns = ["vision_model", "vision_tower", "multi_modal_projector"]
            weight_names = index_data.get("weight_map", {}).keys()
            return any(
                pattern in weight_name
                for weight_name in weight_names
                for pattern in vision_patterns
            )
        except (OSError, json_lib.JSONDecodeError, KeyError):
            return False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.padding_pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(
        self,
        items: List[MultimodalDataItem],
    ) -> torch.Tensor:
        # For text-only models, return None or raise an error
        if not self.has_vision or self.vision_model is None:
            raise ValueError("Vision model not available for text-only checkpoint")
        pixel_values = (
            torch.concat([item.feature for item in items])
            .to(next(self.vision_model.parameters()).device)
            .type(next(self.vision_model.parameters()).dtype)
        )
        image_features = self.vision_model(pixel_values)

        vision_flat = image_features.view(-1, image_features.size(-1))

        projected_vision_flat = self.multi_modal_projector(vision_flat)

        return projected_vision_flat

    def should_apply_lora(self, module_name: str) -> bool:
        """Skip vision model and multi_modal_projector for LoRA."""
        return bool(self.lora_pattern.match(module_name))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:

        # For text-only models, pass None for image_data_embedding_func
        image_embedding_func = self.get_image_feature if self.has_vision else None

        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: image_embedding_func,
            },
            positions=positions,
        )

        return hs

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.language_model.config.head_dim * n_heads
            attn_out = self.language_model.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            if _is_cpu:
                dim = self.language_model.config.original_total_num_kv_heads
            else:
                dim = self.language_model.config.num_key_value_heads
            loaded_weight = permute(loaded_weight, dim)
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            if _is_cpu:
                dim = self.language_model.config.original_num_attention_heads
            else:
                dim = self.language_model.config.num_attention_heads
            loaded_weight = permute(loaded_weight, dim)

        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        num_experts = (
            self.config.text_config.num_local_experts
            if hasattr(self.config, "text_config")
            else self.config.num_local_experts
        )

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        loaded_params = set()

        for name, loaded_weight in weights:
            if self._should_skip_weight(name):
                continue

            name = self._transform_weight_name(name)

            if "vision" in name:
                name = name.replace(".self_attn.o_proj", ".self_attn.proj")
            else:
                name, loaded_weight = self.permute_qk_weight_for_rotary(
                    name, loaded_weight
                )

            if self._handle_scale_remapping(name, params_dict):
                loaded_params.add(name)
                continue

            if self._handle_stacked_params(
                name, loaded_weight, stacked_params_mapping, params_dict, loaded_params
            ):
                continue

            if self._handle_expert_weights(
                name,
                loaded_weight,
                expert_params_mapping,
                params_dict,
                num_experts,
                loaded_params,
            ):
                continue

            loaded_params.add(name)
            self._handle_default_weight(name, loaded_weight, params_dict)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints {unloaded_params}"
            )

    def _should_skip_weight(self, name: str) -> bool:
        """Check if we should skip loading this weight."""
        return not self.has_vision and (
            "vision" in name or "multi_modal_projector" in name
        )

    def _transform_weight_name(self, name: str) -> str:
        """Transform weight name by adding language_model prefix if needed."""
        if (
            not name.startswith("language_model.")
            and "vision" not in name
            and "multi_modal_projector" not in name
        ):
            return f"language_model.{name}"
        return name

    def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            return remapped_name != name
        return False

    def _handle_stacked_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        stacked_params_mapping: list,
        params_dict: dict,
        loaded_params: set,
    ) -> bool:
        """Handle stacked parameter loading. Returns True if handled."""
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                transformed_name = name.replace(weight_name, param_name)
                loaded_params.add(transformed_name)
                param = params_dict[transformed_name]
                param.weight_loader(param, loaded_weight, shard_id)
                return True
        return False

    def _handle_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        expert_params_mapping: list,
        params_dict: dict,
        num_experts: int,
        loaded_params: set,
    ) -> bool:
        """Handle expert weight loading for MoE (Mixture of Experts) layers.

        Args:
            name: Parameter name from the checkpoint
            loaded_weight: The weight tensor to be loaded
            expert_params_mapping: Mapping of parameter names to expert configurations
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts in the MoE layer

        Returns:
            bool: True if the parameter was handled (is an expert parameter), False otherwise
        """
        if ".experts" not in name:
            return False

        if "experts.gate_up_proj" not in name and "experts.down_proj" not in name:
            return self._handle_other_expert_params(
                name, loaded_weight, expert_params_mapping, params_dict, loaded_params
            )

        if "scale" in name:
            return self._handle_expert_scale_params(
                name, loaded_weight, params_dict, num_experts, loaded_params
            )
        else:
            return self._handle_expert_weight_params(
                name, loaded_weight, params_dict, num_experts, loaded_params
            )

    def _handle_other_expert_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        expert_params_mapping: list,
        params_dict: dict,
        loaded_params: set,
    ) -> bool:
        """Handle expert parameters that are not gate_up_proj or down_proj weights.

        Args:
            name: Parameter name from the checkpoint
            loaded_weight: The weight tensor to be loaded
            expert_params_mapping: List of tuples mapping checkpoint names to model parameters
            params_dict: Dictionary of model parameters
            loaded_params: Set of loaded parameter names

        Returns:
            bool: True if parameter was found and handled, False otherwise
        """
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            if weight_name in name:
                transformed_name = name.replace(weight_name, param_name)
                param = params_dict[transformed_name]
                param.weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                loaded_params.add(transformed_name)
                return True
        return False

    def _transform_expert_name(
        self, name: str, is_weight: bool = False
    ) -> Tuple[str, str, List[str]]:
        """Transform expert parameter name and get shard information.

        Args:
            name: The original parameter name
            is_weight: Whether this is a weight parameter (adds _weight suffix)

        Returns:
            Tuple of (transformed_name, shard_id, shard_id_list)
        """
        suffix = "_weight" if is_weight else ""

        if ".gate_up_proj" in name:
            transformed_name = name.replace(
                ".experts.gate_up_proj", f".experts.w13{suffix}"
            )
            shard_id = "w13"
            shard_id_list = ["w1", "w3"]
        else:  # down_proj
            transformed_name = name.replace(
                ".experts.down_proj", f".experts.w2{suffix}"
            )
            shard_id = "w2"
            shard_id_list = ["w2"]

        return transformed_name, shard_id, shard_id_list

    def _handle_expert_scale_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        num_experts: int,
        loaded_params: set,
    ) -> bool:
        """Handle quantization scale parameters for expert weights.

        Args:
            name: Parameter name containing scale information
            loaded_weight: Scale tensor to be loaded
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts for broadcast operations
            loaded_params: Set of loaded parameter names

        Returns:
            bool: True (always handles scale parameters)
        """
        import re

        # Check if this matches the expert parameter pattern: experts.{expert_id}.{param_name}
        expert_match = re.search(r"experts\.(\d+)\.", name)

        # Transform name
        transformed_name, _, _ = self._transform_expert_name(name)

        if transformed_name not in params_dict:
            return True

        param = params_dict[transformed_name]

        # Handle scale parameters
        if expert_match:
            # If we have a specific expert ID, only load for that expert
            expert_id = int(expert_match.group(1))
            # For scale parameters, we can directly set the value
            param.data[expert_id] = loaded_weight
        else:
            # No expert ID found - this is a single scale for all experts
            # Load the same scale for all experts
            for expert_id in range(num_experts):
                param.data[expert_id] = loaded_weight
        loaded_params.add(transformed_name)

        return True

    def _handle_expert_weight_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        num_experts: int,
        loaded_params: set,
    ) -> bool:
        """Handle actual weight tensors for expert layers (gate_up_proj and down_proj).

        Args:
            name: Parameter name (should contain gate_up_proj or down_proj)
            loaded_weight: Weight tensor(s) to be loaded
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts for tensor distribution
            loaded_params: Set of loaded parameter names

        Returns:
            bool: True (always handles weight parameters)
        """
        # Transform name and get shard info
        transformed_name, _, shard_id_list = self._transform_expert_name(
            name, is_weight=True
        )

        if ".gate_up_proj" in name:
            loaded_weight_list = loaded_weight.chunk(2, dim=-1)
        else:  # down_proj
            loaded_weight_list = [loaded_weight]

        for param_name, weight_chunk, shard_id in zip(
            [transformed_name] * len(shard_id_list), loaded_weight_list, shard_id_list
        ):
            if param_name not in params_dict:
                continue

            param = params_dict[param_name]
            weight_loader = param.weight_loader
            loaded_params.add(param_name)

            # Handle the case where loaded_weight might be a single tensor for all experts
            if weight_chunk.dim() == 2:
                # Single tensor case - load for all experts
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        weight_chunk.T,
                        param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
            else:
                # Multiple experts case - load each expert's weights
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        weight_chunk[expert_id].T,
                        param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )

        return True

    def _handle_default_weight(
        self, name: str, loaded_weight: torch.Tensor, params_dict: dict
    ):
        """Handle default weight loading."""
        # Skip loading extra bias for GPTQ models
        if name.endswith(".bias") and name not in params_dict:
            return

        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):
            self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def get_embed_and_head(self):
        # For EAGLE3, we delegate to the language model which should have this method
        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
        embed = self.language_model.get_embed()
        if hasattr(self.language_model, "get_embed_and_head"):
            return self.language_model.get_embed_and_head()
        elif hasattr(self.language_model, "lm_head"):
            return embed, self.language_model.lm_head.weight
        else:
            # For EAGLE3, head might not be needed
            return embed, None

    def set_embed_and_head(self, embed, head):
        if hasattr(self.language_model, "set_embed_and_head"):
            return self.language_model.set_embed_and_head(embed, head)
        else:
            # For EAGLE3, only set embed
            return self.language_model.set_embed(embed)

    def get_embed(self):
        return self.language_model.get_embed()

    def set_embed(self, embed):
        return self.language_model.set_embed(embed)

    def get_hidden_dim(self, module_name, layer_idx):
        # return input_dim, output_dim
        if module_name == "qkv_proj":
            return (
                self.config.hidden_size,
                self.config.head_dim
                * (
                    self.config.num_attention_heads
                    + self.config.num_key_value_heads * 2
                ),
            )
        elif module_name == "o_proj":
            return (
                self.config.head_dim * self.config.num_attention_heads,
                self.config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            return self.config.hidden_size, self.config.intermediate_size * 2
        elif module_name == "down_proj":
            decoder_layer = self.language_model.get_layers()[layer_idx]
            intermediate_size = decoder_layer.get_intermediate_size()
            return intermediate_size, self.config.hidden_size
        else:
            raise NotImplementedError()


EntryClass = Llama4ForConditionalGeneration
