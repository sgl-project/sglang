"""Inference-only LLaVa model compatible with HuggingFace weights."""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.managers.router.model_runner import InputMetadata
from sglang.srt.models.llama2 import LlamaForCausalLM
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModel, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)


class LlavaLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlavaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config, linear_method)

    def pad_input_ids(self, input_ids, pad_value):
        pad_ids = pad_value * (
            (self.image_feature_len + len(pad_value)) // len(pad_value)
        )
        offset = input_ids.index(self.config.image_token_index)
        # old_len + pad_len - 1, because we need to remove image_token_id
        new_input_ids = (
            input_ids[:offset]
            + pad_ids[: self.image_feature_len]
            + input_ids[offset + 1 :]
        )
        return new_input_ids, offset

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        pixel_values: Optional[List[Optional[np.array]]] = None,
        image_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if input_metadata.forward_mode == ForwardMode.EXTEND:
            bs = input_metadata.batch_size

            # Embed text input
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Embed vision input
            need_vision = (
                (positions[input_metadata.extend_start_loc] < self.image_feature_len)
                .cpu()
                .numpy()
            )
            # FIXME: We need to substract the length of the system prompt
            has_pixel = np.array([pixel_values[i] is not None for i in range(bs)])
            need_vision = need_vision & has_pixel

            if need_vision.any():
                pixel_values = torch.tensor(
                    np.array([pixel_values[i] for i in range(bs) if need_vision[i]]),
                    device=self.vision_tower.device,
                )

                image_outputs = self.vision_tower(
                    pixel_values, output_hidden_states=True
                )
                # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.

                selected_image_feature = image_outputs.hidden_states[
                    self.vision_feature_layer
                ]
                if self.vision_feature_select_strategy in ["default", "patch"]:
                    selected_image_feature = selected_image_feature[:, 1:]
                elif self.vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )
                image_features = self.multi_modal_projector(selected_image_feature)

                extend_start_loc_cpu = input_metadata.extend_start_loc.cpu().numpy()
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    pad_len, pad_dim = image_features[pt].shape
                    dim = input_embeds.shape[1]
                    assert (
                        pad_dim == dim
                    ), "invalid pad_dim={}, input_embed_dim={}!".format(pad_dim, dim)
                    # Fill in the placeholder for the image
                    try:
                        input_embeds[
                            start_idx
                            + image_offsets[i] : start_idx
                            + image_offsets[i]
                            + pad_len
                        ] = image_features[pt]
                    except RuntimeError as e:
                        print(f"RuntimeError in llava image encoding: {e}")
                        print(input_embeds.shape)
                        print(start_idx, image_offsets[i])
                    pt += 1

            return self.language_model(
                input_embeds, positions, input_metadata, skip_embed=True
            )
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.language_model(
                input_ids, positions, input_metadata, skip_embed=False
            )

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = self.config.mm_vision_tower
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_path, torch_dtype=torch.float16
        ).cuda()
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size
        self.image_feature_len = int((self.image_size / self.patch_size) ** 2)
        if self.vision_feature_select_strategy == "patch":
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        # TODO: support TP?
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # FIXME: why projector weights read two times?
            if "projector" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # load language model
        self.language_model.load_weights(
            model_name_or_path, cache_dir, load_format, revision
        )

        monkey_path_clip_vision_embed_forward()


first_call = True


def clip_vision_embed_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]

    # Move this conv layer to CPU to avoid a bug in torch >= 2.1 on A10G.
    global first_call
    if first_call:
        self.patch_embedding.cpu().float()
        first_call = False
    pixel_values = pixel_values.to(dtype=torch.float32, device="cpu")
    patch_embeds = self.patch_embedding(pixel_values).cuda().half()

    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


def monkey_path_clip_vision_embed_forward():
    import transformers

    setattr(
        transformers.models.clip.modeling_clip.CLIPVisionEmbeddings,
        "forward",
        clip_vision_embed_forward,
    )
