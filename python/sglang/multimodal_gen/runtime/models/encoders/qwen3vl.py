# SPDX-License-Identifier: Apache-2.0
"""Inference-only Qwen3-VL model compatible with HuggingFace weights.

The language model is srt's (``qwen3vl_text``); the vision tower is native here.
"""

from typing import Iterable, Optional, Tuple, Union

try:
    from typing import Unpack  # type: ignore[attr-defined]
except ImportError:
    # Python 3.10 and below
    from typing_extensions import Unpack

import torch
import torch.nn as nn
from transformers import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_text import (
    Qwen3VLTextModel,
    load_srt_text_weight,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionTransformer,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen_vl import QwenVLModelBase
from sglang.multimodal_gen.runtime.utils.common import add_prefix


class Qwen3VLModel(QwenVLModelBase):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3DecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        vision_quant_config = (
            quant_config
            if quant_config is not None and quant_config.supports_srt_linear_layers
            else None
        )
        self.visual = Qwen3VLVisionTransformer(
            config.vision_config,
            quant_config=vision_quant_config,
            prefix=add_prefix("visual", prefix),
        )
        self.language_model = Qwen3VLTextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
            deepstack_visual_indexes=config.vision_config.deepstack_visual_indexes,
        )
        self.rope_deltas = None  # cache rope_deltas here
        self.config = config

        # Initialize weights and apply final processing

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            return self._get_text_rope_index(input_ids, attention_mask)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    # not cached across requests: hashing pixel_values to look them up costs more than the vision tower
    def _get_flat_visual_features(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: Optional[torch.LongTensor],
    ):
        visual_out = self.visual(pixel_values, grid_thw=grid_thw)
        return visual_out.pooler_output, visual_out.deepstack_features

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
            pixel_values, image_grid_thw
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds_owned = inputs_embeds is None
        if inputs_embeds_owned:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(image_mask, image_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    img_embed.shape[0] + vid_embed.shape[0], img_embed.shape[-1]
                )
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

        return output if return_dict else output.to_tuple()


class Qwen3VLForConditionalGeneration(TextEncoder):
    layer_names = [
        *TextEncoder.layer_names,
        "model.visual.blocks",
        "model.visual.deepstack_merger_list",
    ]
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        quant_config = config.quant_config
        config = config.arch_config
        self.model = Qwen3VLModel(config, quant_config=quant_config, prefix="model")
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self._lm_head_from_embedding = False
        if getattr(config, "tie_word_embeddings", False) or getattr(
            config.text_config, "tie_word_embeddings", False
        ):
            # Tied checkpoints may omit lm_head.weight, as Hugging Face does.
            # Keep one registered parameter so strict native loading stays safe.
            embedding = self.model.get_input_embeddings().weight
            if embedding.shape == self.lm_head.weight.shape:
                self.lm_head.weight = embedding
            else:
                # srt's table is padded and split over the encoder's TP ranks;
                # the head then loads the table's checkpoint tensor itself
                self._lm_head_from_embedding = True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        output_attentions = False
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_params: set[str] = set()

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "visual." in name:
                name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
            if name.startswith("model.language_model."):
                loaded_params.add(
                    load_srt_text_weight(params_dict, name, loaded_weight)
                )
                if (
                    self._lm_head_from_embedding
                    and name == "model.language_model.embed_tokens.weight"
                ):
                    self.lm_head.weight.data.copy_(loaded_weight)
                    loaded_params.add("lm_head.weight")
                continue

            try:
                param = params_dict[name]
            except KeyError:
                raise KeyError(
                    f"Unexpected weight name while loading Qwen3VL: {name}"
                ) from None

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = loaded_weight.to(param.dtype)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = Qwen3VLForConditionalGeneration
