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
# ==============================================================================
"""Inference-only VILA 1.5 model (NVIDIA Visual-language Architecture).

Supports the Efficient-Large-Model/VILA1.5-* checkpoint family including
quantized variants (e.g. VILA1.5-3B-AWQ).

VILA 1.5 stores all parameters — both LLaMA backbone fields and multimodal
fields — in a single flat config with model_type='llava_llama'.  Its
safetensors weight names follow the same LLaVA-Llama naming convention used
by the original LLaVA-1.5 checkpoints.

Reference: https://github.com/NVlabs/VILA
"""

from collections.abc import Iterable
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPVisionModel, SiglipVisionModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
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
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import logger


class VILAConfig(PretrainedConfig):
    """Configuration for VILA 1.5 models.

    VILA1.5 uses a flat config layout: LLaMA backbone parameters coexist
    with multimodal parameters at the top level under model_type='llava_llama'.
    This differs from the structured sub-config layout used by later NVILA
    checkpoints.
    """

    model_type = "llava_llama"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        mm_vision_tower: str = "google/siglip-so400m-patch14-384",
        mm_hidden_size: int = 1152,
        mm_projector_type: str = "mlp2x_gelu",
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = "patch",
        mm_patch_merge_type: str = "flat",
        image_aspect_ratio: str = "resize",
        image_token_index: int = 32000,
        **kwargs,
    ) -> None:
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.image_aspect_ratio = image_aspect_ratio
        self.image_token_index = image_token_index
        super().__init__(**kwargs)


class VILAMultiModalProjector(nn.Module):
    """Two-layer GELU MLP projector (mlp2x_gelu) used by VILA 1.5.

    Maps SigLIP/CLIP patch features from mm_hidden_size to the LLM's
    hidden_size so they can be concatenated with text token embeddings.
    """

    def __init__(self, mm_hidden_size: int, lm_hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(mm_hidden_size, lm_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(lm_hidden_size, lm_hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class VILAForConditionalGeneration(nn.Module):
    """VILA 1.5 multimodal model.

    Architecture: SigLIP (or CLIP) vision encoder → two-layer MLP projector
    → LLaMA language model.

    Also exposed under the name ``LlavaLlamaModel`` to match the
    ``architectures`` field written in VILA1.5 config.json files.
    """

    def __init__(
        self,
        config: VILAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # Vision tower is populated lazily inside load_weights so we can
        # respect the hub path stored in config.mm_vision_tower.
        self.vision_tower: Optional[nn.Module] = None
        self.vision_feature_layer: int = config.mm_vision_select_layer
        self.vision_feature_select_strategy: str = config.mm_vision_select_feature

        self.multi_modal_projector = VILAMultiModalProjector(
            mm_hidden_size=config.mm_hidden_size,
            lm_hidden_size=config.hidden_size,
        )
        # VILA's flat config contains all LLaMA fields directly, so we can
        # pass it straight to LlamaForCausalLM.
        self.language_model = LlamaForCausalLM(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_image_feature(self, mm_input: List[MultimodalDataItem]) -> Tensor:
        if self.vision_tower is None:
            raise RuntimeError(
                "VILA vision tower is not initialised; load_weights must run before forward."
            )
        pixel_values = torch.cat(
            [torch.as_tensor(x.feature) for x in mm_input], dim=0
        )
        # Cast to the dtype/device of the vision tower.
        vt_param = next(self.vision_tower.parameters())
        pixel_values = pixel_values.to(device=vt_param.device, dtype=vt_param.dtype)

        output: BaseModelOutputWithPooling = self.vision_tower(
            pixel_values, output_hidden_states=True
        )
        assert output.hidden_states is not None
        features: Tensor = output.hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "patch":
            # Remove the CLS token present in CLIP-style encoders.
            features = features[:, 1:]

        features = self.multi_modal_projector(features)
        # Flatten (n_images, n_patches, D) → (n_images * n_patches, D)
        return features.flatten(0, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        output = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            get_embedding=get_embedding,
            positions=positions,
        )
        assert isinstance(output, LogitsProcessorOutput)
        return output

    # ------------------------------------------------------------------
    # Padding helper
    # ------------------------------------------------------------------

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        vision_path = self.config.mm_vision_tower
        if "siglip" in vision_path.lower():
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
            # SigLIP encodes all patch tokens without a leading CLS token,
            # so we keep the full sequence rather than stripping index 0.
            self.vision_feature_select_strategy = "full"
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
        self.vision_tower.eval()

        # Map VILA1.5 checkpoint weight names to SGLang parameter names.
        # VILA follows the LLaVA-Llama naming convention:
        #   model.mm_projector.0.*  → multi_modal_projector.linear_1.*
        #   model.mm_projector.2.*  → multi_modal_projector.linear_2.*
        #   model.vision_tower.vision_tower.*  → vision_tower.*
        # (transformers >= 5.6 flattened the vision_model intermediate
        #  module, so we handle both name forms)
        _PROJECTOR_MAP: dict[str, str] = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower",
            "vision_tower.vision_model.": "vision_tower.",
        }

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            is_mm = any(k in name for k in ("mm_projector", "vision_tower"))
            if is_mm:
                remapped = name
                for src, dst in _PROJECTOR_MAP.items():
                    if remapped.startswith(src):
                        remapped = dst + remapped[len(src):]
                        break
                    elif src in remapped:
                        remapped = remapped.replace(src, dst, 1)
                        break
                param = params_dict.get(remapped)
                if param is None:
                    logger.debug("vila: skipping unrecognised weight %s", name)
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            else:
                # All remaining keys (model.embed_tokens, model.layers, lm_head, …)
                # are LLaMA weights and are forwarded directly.
                self.language_model.load_weights([(name, loaded_weight)])


class LlavaLlamaModel(VILAForConditionalGeneration):
    """Architecture alias for VILAForConditionalGeneration.

    VILA1.5 config.json files set ``architectures: ["LlavaLlamaModel"]``.
    Registering this subclass allows SGLang to resolve that name to the
    correct implementation without modifying the checkpoint.
    """


EntryClass = [VILAForConditionalGeneration, LlavaLlamaModel]
