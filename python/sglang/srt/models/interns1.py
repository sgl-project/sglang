from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.internvl import InternVisionModel
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.utils import logger


class InternS1ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self._update_hf_config()
        image_size = (
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        patch_size = config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.select_layer = config.vision_feature_layer
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = getattr(config, "ps_version", "v1")
        # self.template = getattr(config, 'template', 'internvl2_5')

        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.text_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")

        self.vision_model = InternVisionModel(config.vision_config)
        if config.text_config.architectures[0] == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(
                config=config.text_config, quant_config=quant_config
            )
        elif config.text_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.language_model = Qwen3MoeForCausalLM(
                config=config.text_config, quant_config=quant_config
            )
        else:
            raise NotImplementedError(
                f"{config.text_config.architectures[0]} is not implemented."
            )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def _update_hf_config(self):
        """update hf config to support tp"""
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        num_heads = self.config.vision_config.num_attention_heads
        head_dim = self.config.vision_config.hidden_size // num_heads
        num_dummy_heads = 0

        if num_heads % world_size != 0:
            num_dummy_heads = (
                (num_heads + world_size) // world_size
            ) * world_size - num_heads

        setattr(self.config.vision_config, "head_dim", head_dim)
        setattr(self.config.vision_config, "num_dummy_heads", num_dummy_heads)

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

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        image_features = self.extract_feature(pixel_values)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hs

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):
        """pad attn qkv weights for dummy heads"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads
        if num_dummy_heads == 0:
            return loaded_weight
        head_dim = self.config.vision_config.head_dim

        if any([_ in name for _ in ["attn.q_proj", "attn.k_proj", "attn.v_proj"]]):
            if name.endswith(".weight"):
                dummy_shape = [num_dummy_heads, head_dim, loaded_weight.shape[-1]]
            elif name.endswith(".bias"):
                dummy_shape = [num_dummy_heads, head_dim]
            else:
                raise RuntimeError(f"Unsupported weight with name={name}")
            padded_weight = loaded_weight.new_zeros(dummy_shape)
            loaded_weight = torch.cat(
                [loaded_weight.unflatten(0, (-1, head_dim)), padded_weight], dim=0
            ).flatten(0, 1)
        if "attn.proj.weight" in name:
            padded_weight = loaded_weight.new_zeros(
                loaded_weight.shape[0], head_dim * num_dummy_heads
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
        if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
        return loaded_weight

    def _mapping_interns1_name(self, name):
        names_map = {
            "lm_head.weight": "language_model.lm_head.weight",
            "model.multi_modal_projector.layer_norm.bias": "mlp1.0.bias",
            "model.multi_modal_projector.layer_norm.weight": "mlp1.0.weight",
            "model.multi_modal_projector.linear_1.bias": "mlp1.1.bias",
            "model.multi_modal_projector.linear_1.weight": "mlp1.1.weight",
            "model.multi_modal_projector.linear_2.bias": "mlp1.3.bias",
            "model.multi_modal_projector.linear_2.weight": "mlp1.3.weight",
            "model.vision_tower.embeddings.cls_token": "vision_model.embeddings.class_embedding",
            "model.vision_tower.embeddings.patch_embeddings.projection.bias": "vision_model.embeddings.patch_embedding.bias",
            "model.vision_tower.embeddings.patch_embeddings.projection.weight": "vision_model.embeddings.patch_embedding.weight",
            "model.vision_tower.embeddings.position_embeddings": "vision_model.embeddings.position_embedding",
        }
        if name in names_map:
            name = names_map[name]
        elif name.startswith("model.language_model."):
            name = "language_model.model." + name[len("model.language_model.") :]
        elif name.startswith("model.vision_tower."):
            name = "vision_model." + name[len("model.vision_tower.") :]

        if name.startswith("vision_model.encoder.layer"):

            name = name.replace(r".layer.", r".layers.")
            name = name.replace(r".attention.", r".attn.attn.")
            name = name.replace(r".projection_layer.", r".proj.")
            name = name.replace(r".lambda_1", r".ls1")
            name = name.replace(r".lambda_2", r".ls2")
            name = name.replace(r".layernorm_before.", r".norm1.")
            name = name.replace(r".layernorm_after.", r".norm2.")
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = []
        if "Qwen3MoeForCausalLM" in self.config.text_config.architectures:
            expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = self._mapping_interns1_name(name)
            if "vision_model" in name:
                loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )
        return loaded_params


EntryClass = [InternS1ForConditionalGeneration]
