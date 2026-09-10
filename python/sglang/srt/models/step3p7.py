from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.configs.step3p7 import Step3p7Config
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.step3_vl_10b import PerceptionEncoder
from sglang.srt.models.step3p5 import Step3p5ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix


class Step3p7ForConditionalGeneration(nn.Module):
    # NVFP4 checkpoints (e.g. huangyu-nv/step3p7-nvfp4-moe-only-kvfp8) use
    # "model.language_model." prefix, while sglang parameters are named
    # "language_model.model.". This mapper remaps the quantization ignore
    # patterns so that is_layer_skipped works correctly.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_model": "vision_model",
            "model.vit_large_projector": "vit_large_projector",
        }
    )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return Step3p5ForCausalLM.get_model_config_for_expert_location(
            config.text_config
        )

    def __init__(
        self,
        config: Step3p7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.vision_model = PerceptionEncoder(
            config.vision_config,
            ACT2FN[config.vision_config.hidden_act],
            quant_config=None,  # Vision weights are not quantized
            prefix=add_prefix("vision_model", prefix),
        )
        self.vit_large_projector = ColumnParallelLinear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            gather_output=True,
            quant_config=None,  # Projector weights are bf16
            prefix=add_prefix("vit_large_projector", prefix),
        )
        self.language_model = Step3p5ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)

    @property
    def device(self) -> torch.device:
        return self.vit_large_projector.weight.device

    def _flatten_embeddings(self, embeddings) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.flatten(0, -2)
        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.vit_large_projector(image_features)
        return image_features

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            raise ValueError("Step3.7 get_image_feature requires at least one item.")

        # Thumbnails and local patches have different semantics, so batch each
        # group independently and restore the original item/image order below.
        all_thumbnails = []
        all_patches = []
        # Per-item metadata: (thumbnail count, patch counts per image, patch count)
        item_metadata = []

        for item in items:
            pixel_values = item.feature.type(self.vision_model.dtype)
            num_patches = item.model_specific_data.get("num_patches")
            if num_patches is None:
                raise ValueError("Step3.7 image item is missing num_patches.")
            if isinstance(num_patches, torch.Tensor):
                num_patches = [int(x) for x in num_patches.flatten().cpu().tolist()]
            elif isinstance(num_patches, (list, tuple)):
                num_patches = [
                    int(x.item()) if isinstance(x, torch.Tensor) else int(x)
                    for x in num_patches
                ]
            else:
                num_patches = [int(num_patches)]
            if any(count < 0 for count in num_patches):
                raise ValueError("Step3.7 image item has a negative num_patches value.")

            thumbnail_count = pixel_values.shape[0]
            if len(num_patches) != thumbnail_count:
                raise ValueError(
                    "Step3.7 image item has mismatched thumbnail and num_patches "
                    f"counts: {thumbnail_count} != {len(num_patches)}."
                )

            patch_pixel_values = item.model_specific_data.get(
                "patch_pixel_values", None
            )
            if patch_pixel_values is not None and patch_pixel_values.shape[0] == 0:
                patch_pixel_values = None
            if patch_pixel_values is not None:
                patch_pixel_values = patch_pixel_values.type(
                    self.vision_model.dtype
                ).to(self.device)

            patch_count = (
                patch_pixel_values.shape[0] if patch_pixel_values is not None else 0
            )
            expected_patch_count = sum(num_patches)
            if patch_count != expected_patch_count:
                raise ValueError(
                    "Step3.7 image item has mismatched patch_pixel_values and "
                    f"num_patches counts: {patch_count} != {expected_patch_count}."
                )

            all_thumbnails.append(pixel_values)
            if patch_pixel_values is not None:
                all_patches.append(patch_pixel_values)
            item_metadata.append((thumbnail_count, num_patches, patch_count))

        # Use one ViT/projector pass for all thumbnails and one for all local
        # patches. Avoid a redundant allocation on the single-item path.
        thumbnail_batch = (
            all_thumbnails[0]
            if len(all_thumbnails) == 1
            else torch.cat(all_thumbnails, dim=0)
        )
        all_thumbnail_features = self._process_image_features(
            self._get_vision_model_output(thumbnail_batch)
        )

        all_patch_features = None
        if all_patches:
            patch_batch = (
                all_patches[0]
                if len(all_patches) == 1
                else torch.cat(all_patches, dim=0)
            )
            all_patch_features = self._process_image_features(
                self._get_vision_model_output(patch_batch)
            )

        # Split the batched output back into item/image order. The flattened
        # result must align with the multimodal placeholders in input_ids.
        merged_image_features = []
        thumbnail_offset = 0
        patch_offset = 0
        for thumbnail_count, num_patches, patch_count in item_metadata:
            item_thumbnail_features = all_thumbnail_features[
                thumbnail_offset : thumbnail_offset + thumbnail_count
            ]
            thumbnail_offset += thumbnail_count
            item_patch_features = (
                all_patch_features[patch_offset : patch_offset + patch_count]
                if patch_count > 0
                else None
            )
            patch_offset += patch_count

            current_patch_index = 0
            for image_index, num_patch in enumerate(num_patches):
                current_features = []
                if num_patch > 0:
                    patch_slice = item_patch_features[
                        current_patch_index : current_patch_index + num_patch
                    ]
                    current_features.append(patch_slice.view(-1, patch_slice.shape[-1]))
                current_features.append(
                    item_thumbnail_features[image_index].view(
                        -1, item_thumbnail_features.shape[-1]
                    )
                )
                current_patch_index += num_patch
                merged_image_features.append(
                    torch.cat(current_features)
                    if len(current_features) > 1
                    else current_features[0]
                )
        return self._flatten_embeddings(merged_image_features)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )
        return hidden_states

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        self.language_model.set_embed_and_head(embed, head)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)

        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            # NVFP4 checkpoints use "model.language_model." prefix for
            # language weights and "model.vision_model." for vision weights,
            # while FP8 checkpoints use "model." and "vision_model." directly.
            name = name.replace("language_model.", "", 1)

            if "vision_model" in name or "vit_large_projector" in name:
                # Strip leading "model." for vision weights (NVFP4 format)
                if name.startswith("model."):
                    name = name[len("model.") :]
                name = name.replace(r".attn.in_proj_weight", r".attn.qkv_proj.weight")
                name = name.replace(r".attn.in_proj_bias", r".attn.qkv_proj.bias")
                name = name.replace(r".attn.out_proj.bias", r".attn.proj.bias")
                name = name.replace(r".attn.out_proj.weight", r".attn.proj.weight")
                name = name.replace(".mlp.c_fc", ".mlp.fc1")
                name = name.replace(".mlp.c_proj", ".mlp.fc2")
                vision_weights.append((name, loaded_weight))
            else:
                language_weights.append((name, loaded_weight))

        # Load vision tower weights
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in vision_weights:
            if name not in params_dict:
                raise ValueError(f"Weight {name} not found in params_dict")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)


EntryClass = Step3p7ForConditionalGeneration
