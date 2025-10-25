import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import timm
import timm.data
import tokenizers
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
import transformers
from PIL import Image
from timm.models.vision_transformer import LayerScale
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType

from sglang.srt.configs.openvla import OpenVLAConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert (
            len(timm_model_ids) <= 2
        ), "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(
                self.featurizer.get_intermediate_layers,
                n={len(self.featurizer.blocks) - 2},
            )
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(
                    self.fused_featurizer.get_intermediate_layers,
                    n={len(self.fused_featurizer.blocks) - 2},
                )
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


class PrismaticProjector(nn.Module):
    def __init__(
        self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


class OpenVLAForActionPrediction(PreTrainedModel):
    config_class: PretrainedConfig = OpenVLAConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def __init__(
        self,
        config: OpenVLAConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config)
        self.embeddings_layer = None
        self.past_key_values = None
        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config=quant_config
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = (
            self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        assert len(mm_inputs.mm_items) == 1, "OpenVLA only supports single image inputs"
        pad_value = mm_inputs.mm_items[0].pad_value
        input_ids = input_ids[:1] + [pad_value] * 256 + input_ids[1:]
        if input_ids[-1] != 29871:
            input_ids.append(29871)  # OpenVLA Specific
        return input_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)
        new_weights = []
        params_dict = dict(self.named_parameters())
        for name, weight in weights:
            if not "language_model" in name:
                param = params_dict[name]
                default_weight_loader(param, weight)
                continue

            new_name = None
            _KEYS_TO_MODIFY_MAPPING = {
                "language_model.model": "model",
                "language_model.lm_head": "lm_head",
            }
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    new_name = name.replace(key_to_modify, new_key)

            if new_name is not None:
                new_weights.append((new_name, weight))
            else:
                new_weights.append((name, weight))

        weights = new_weights

        self.language_model.load_weights(weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        need_vision = (
            forward_batch.mm_inputs is not None
            and forward_batch.mm_inputs[0] is not None
        )

        # === Handle Unimodal Forward, this is for warmup only ===
        if not need_vision or len(positions) == 1:
            assert (
                input_ids is not None
            ), "Missing `input_ids` in language-only forward!"
            return self.language_model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=None,
            )

        # === Handle Multimodal Forward ===

        # No need to patch image embeddings if decode
        if forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

        embedding_layer = self.language_model.model.embed_tokens
        input_ids.clamp_(min=0, max=32064 - 1)  # Clamp image pad_value token ids
        bs = forward_batch.batch_size
        assert bs == len(forward_batch.mm_inputs), (
            "Batch size doesn't match the number of image inputs, "
            "each request can have only one image."
        )

        extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().tolist()
        extend_seq_lens_cpu = forward_batch.extend_seq_lens.cpu().tolist()
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

        pt = 0
        for i, image_input in enumerate(forward_batch.mm_inputs):
            assert image_input is not None, "Missing mm_inputs entry"
            assert (
                len(image_input.mm_items) == 1
            ), "OpenVLA only supports single image inputs"

            mm_item = image_input.mm_items[0]

            if not getattr(mm_item, "offsets", None):
                mm_item.offsets = [(1, 256)]

            relative_id_image_start = 1 - prefix_lens_cpu[i]
            relative_id_image_end = relative_id_image_start + 256
            id_start = max(pt + relative_id_image_start, extend_start_loc_cpu[i])
            id_end = min(
                pt + relative_id_image_end,
                extend_start_loc_cpu[i] + extend_seq_lens_cpu[i],
            )
            if id_end < 0:
                id_end = 0

            if id_end > id_start:
                pad_val = mm_item.pad_value
                input_ids[id_start:id_end] = pad_val

            pt += extend_seq_lens_cpu[i]

        def _image_embedder(items: List[MultimodalDataItem]) -> torch.Tensor:
            outs = []
            for it in items:
                pixel_value = it.feature.to(torch.bfloat16).to(input_ids.device)
                patch_features = self.vision_backbone(pixel_value)
                projected = self.projector(patch_features)[0]
                outs.append(projected)
            return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=None,
            data_embedding_funcs={Modality.IMAGE: _image_embedder},
            placeholder_tokens=None,
            use_deepstack=False,
            positions=positions,
        )


EntryClass = OpenVLAForActionPrediction
