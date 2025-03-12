# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================582====================================================

from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.internlm2 import InternLM2ForCausalLM
from sglang.srt.models.internvl_vit import InternVisionModel
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.utils import logger


class InternVLChatModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        config.llm_config.attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        self.vision_model = InternVisionModel(
            config=config.vision_config, quant_config=quant_config
        )
        if config.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self.language_model = InternLM2ForCausalLM(config.llm_config)
            self.llm_architectures = "InternLM2ForCausalLM"
        elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(config.llm_config)
            self.llm_architectures = "Qwen2ForCausalLM"
        else:
            raise NotImplementedError(
                f"{config.llm_config.architectures[0]} is not implemented."
            )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            logger.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if (
            forward_batch.image_inputs is not None
            and forward_batch.image_inputs[0] is not None
        ):
            image_input = forward_batch.image_inputs[0]

            image_token_indices = torch.isin(
                input_ids,
                torch.tensor(image_input.pad_values).to(device=input_ids.device),
            )
            if image_token_indices.sum() == 0:
                pass
            else:
                # [B * S] -> [B, S]
                input_ids = input_ids.unsqueeze(0)
                input_ids.clamp_(min=0, max=self.config.vocab_size - 1)
                if self.llm_architectures == "Qwen2ForCausalLM":
                    input_embeds = self.language_model.model.embed_tokens(input_ids)
                else:
                    input_embeds = self.language_model.model.tok_embeddings(input_ids)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                pixel_values = image_input.pixel_values
                vit_embeds = self.extract_feature(pixel_values)

                num_image_tokens = image_token_indices.sum()
                input_embeds[image_token_indices] = vit_embeds.reshape(-1, C)[
                    -num_image_tokens:
                ].to(input_embeds.device)
                input_embeds = input_embeds.reshape(N, C)
                input_ids = None

        if input_ids is not None:
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        media_token_pairs = [(im_start_id, im_end_id)]
        pad_values = image_inputs.pad_values
        media_token_pairs = media_token_pairs
        start_tokens = [s for s, _e in media_token_pairs]
        end_tokens = [e for _s, e in media_token_pairs]
        # First start token marks new media
        media_start_token = start_tokens[0]

        padded_ids = []
        last_idx = 0
        media_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_tokens]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] == media_start_token:
                media_idx += 1

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[media_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids)
        return padded_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if self.llm_architectures == "Qwen2ForCausalLM":
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "up_proj", 1),
                ("gate_up_proj", "gate_proj", 0),
            ]
        else:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "w1", 0),
                ("gate_up_proj", "w3", 1),
            ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                if "wqkv" in name:
                    config = self.config
                    kv_groups = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(
                        -1, 2 + kv_groups, head_dim, loaded_weight.shape[-1]
                    )
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1], dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, "q")
                    weight_loader(param, wk, "k")
                    weight_loader(param, wv, "v")
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = InternVLChatModel
