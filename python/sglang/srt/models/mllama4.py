# TODO: add Aapted from vllm/mllama4.py
import math
from collections.abc import Iterable, Mapping
from itertools import tee
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import BatchFeature, Llama4Config, Llama4VisionConfig
from transformers.image_utils import SizeDict
from transformers.modeling_outputs import BaseModelOutput

from python.sglang.srt.layers.activation import get_act_fn
from python.sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils import add_prefix


class Llama4ImagePatchInputs(TypedDict):
    type: Literal["pixel_values"]
    flat_data: torch.Tensor
    """
    Shape:
    `(batch_size * num_chunks, num_channels, image size, image size)`
    """
    patches_per_image: torch.Tensor
    """
    The number of total patches for each image in the batch.

    This is used to split the embeddings which has the first two dimensions
    flattened just like `flat_data`.
    """
    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.
    """
    aspect_ratios: Union[torch.Tensor, list[torch.Tensor]]
    """
    A list of aspect ratios corresponding to the number of tiles
    in each dimension that each image in the batch corresponds to.

    Shape:
    `(batch_size, ratio)` where ratio is a pair `(ratio_h, ratio_w)`
    """


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
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=input_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.activation_fn = get_act_fn("gelu")
        self.output_activation = output_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        if self.output_activation:
            return self.activation_fn(hidden_states)
        return hidden_states


class Llama4MultiModalProjector(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            input_size=config.vision_config.vision_output_dim,
            output_size=config.text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
        )

    def forward(self, image_features):
        hidden_states, _ = self.linear_1(image_features)
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
    ):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(
            config.projector_input_dim // (self.pixel_shuffle_ratio**2)
        )
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP(
            input_size=config.intermediate_size,
            intermediate_size=config.projector_input_dim,
            output_size=config.projector_output_dim,
            bias=config.multi_modal_projector_bias,
            output_activation=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


class Llama4VisionAttention(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        assert self.num_heads % self.tp_size == 0
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        # Use SGLang's RadixAttention
        self.attn = RadixAttention(
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_heads,
            head_dim=self.head_dim,
            dropout=self.attention_dropout,
        )
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            self.num_heads,  # num_kv_heads = num_heads for vision model
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Use SGLang's rotary embedding implementation
        self.rotary_emb = get_rope(
            dim=self.head_dim,
            # number of image patches
            max_position=(config.image_size // config.patch_size) ** 2,
            base=config.rope_theta,
            rope_scaling={"rope_type": "mllama4"},
            is_neox_style=False,
            dtype=torch.complex64,  # important
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Generate position IDs for rotary embeddings
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Project inputs to queries, keys, and values
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(position_ids, seq_len)
        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply scaling
        q = q * self.scaling

        # Use RadixAttention for efficient attention computation
        attn_output = self.attn(q, k, v)

        # Reshape and project back to hidden size
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_output, _ = self.o_proj(attn_output)

        return attn_output


class Llama4VisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.self_attn = Llama4VisionAttention(
            config, quant_config=quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Llama4VisionMLP(
            input_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            output_activation=False,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_state: torch.Tensor,
    ):
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(hidden_state)
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)
        return outputs


class Llama4VisionEncoder(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape
                    `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation. This is useful if you
                want more control over how to convert `input_ids` indices into
                associated vectors than the model's internal embedding
                lookup matrix.
        """

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class Llama4UnfoldConvolution(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = ColumnParallelLinear(
            config.num_channels * kernel_size[0] * kernel_size[1],
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            gather_output=True,
            prefix=add_prefix("linear", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states, _ = self.linear(hidden_states)
        return hidden_states


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
            config, quant_config=quant_config, prefix=f"{prefix}.patch_embedding"
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.hidden_size)
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # encoders
        self.model = Llama4VisionEncoder(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            config, quant_config, prefix=add_prefix("vision_adapter", prefix)
        )

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate
        grads on inputs.
        """
        return self.patch_embedding

    def forward(
        self,
        images_flattened: torch.Tensor,
    ) -> BaseModelOutput:
        # Patch embedding
        hidden_state = self.patch_embedding(images_flattened)
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

        # Apply encoder
        output = self.model(hidden_state)
        hidden_state = output.last_hidden_state

        hidden_state = self.layernorm_post(hidden_state)

        # Remove CLS token output
        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            attentions=None,
        )


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vision_model = Llama4VisionModel(
            config.vision_config,
            quant_config,
            prefix=add_prefix("vision_model", prefix),
        )
        self.multi_modal_projector = Llama4MultiModalProjector(
            self.config,
            quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
        )

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[Llama4ImagePatchInputs]:
        # num_images, 1, num_chunks, channel, image_size, image_size
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        # num_images x num_chunks, channel, image_size, image_size
        # TODO: confirm handling for variable lengths
        flat_pixel_values = self.flatten_bn(pixel_values, concat=True)
        patches_per_image = self.flatten_bn(kwargs.pop("patches_per_image"))

        embed_is_patch = kwargs.pop("embed_is_patch", None)
        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of embed_is_patch. " f"Got type: {type(embed_is_patch)}"
            )

        aspect_ratios = kwargs.pop("aspect_ratios", None)
        if not isinstance(aspect_ratios, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of aspect_ratios. " f"Got type: {type(aspect_ratios)}"
            )

        return Llama4ImagePatchInputs(
            type="pixel_values",
            flat_data=flat_pixel_values,
            patches_per_image=patches_per_image,
            embed_is_patch=embed_is_patch,
            aspect_ratios=aspect_ratios,
        )

    def flatten_bn(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
        *,
        concat: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

        The input tensor should have shape ``(B, N, ...)```.
        """
        if isinstance(x, torch.Tensor):
            return x.flatten(0, 1)

        if concat:
            return torch.cat(x)

        return [x_n for x_b in x for x_n in x_b]

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        pass

    def separate_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        prefix: str,
    ) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str, torch.Tensor]]]:
        weights1, weights2 = tee(weights, 2)

        def get_prefix_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights1:
                if name.startswith(prefix):
                    yield (name, data)

        def get_other_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights2:
                if not name.startswith(prefix):
                    yield (name, data)

        return get_prefix_weights(), get_other_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: Set[str] = set()

        # language_model is an Llama4ForCausalLM instance. We load it's
        # using llama4's load_weights routine.
        language_model_prefix = "language_model.model."
        language_model_weights, other_weights = self.separate_weights(
            weights, prefix=language_model_prefix
        )

        # TODO: finish loading weights


EntryClass = Llama4ForConditionalGeneration
