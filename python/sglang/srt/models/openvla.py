import logging
from dataclasses import dataclass
from PIL import Image
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import timm
import timm.data
import tokenizers
import torch
import torch.nn as nn
import transformers
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING
from python.sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.openvla import PrismaticProjector, PrismaticVisionBackbone, PrismaticProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM

# logger = logging.getLogger(__name__)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
# IGNORE_INDEX = -100


class OpenVLAConfig(PretrainedConfig):
    model_type: str = "openvla"
    is_composition: bool = False

    def __init__(
        self,
        norm_stats: Optional[
            Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
        ] = None,
        n_action_bins: int = 256,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(
                self.vision_backbone_id.startswith(v)
                for v in ["dinoclip", "dinosiglip"]
            )
        )

        self.timm_model_ids = [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
            "clip-vit-l": ["quick_gelu"],
            "clip-vit-l-336px": ["quick_gelu"],
            "dinov2-vit-l": [None],
            "in1k-vit-l": [None],
            "siglip-vit-so400m": [None],
            "siglip-vit-so400m-384px": [None],
            "dinoclip-vit-l-336px": [None, "quick_gelu"],
            "dinosiglip-vit-so-224px": [None, None],
            "dinosiglip-vit-so-384px": [None, None],
        }

        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = [224, 224]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = "meta-llama/Llama-2-7b-hf"
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        LLM_BACKBONE_TO_HF_METACLASS = {
            "llama2-7b-pure": "llama",
        }

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](
                **text_config
            )
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.vocab_size = 32064
        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


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
            input_ids.append(29871) # OpenVLA Specific
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
        self.processor = PrismaticProcessor.from_pretrained(
                "openvla/openvla-7b", trust_remote_code=True
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        need_vision = forward_batch.mm_inputs is not None and forward_batch.mm_inputs[0] is not None

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
        input_ids.clamp_(min=0, max=32064 - 1) # Clamp image pad_value token ids
        input_embeddings = embedding_layer(input_ids)
        
        pt = 0
        bs = forward_batch.batch_size
        extend_start_loc_cpu = forward_batch.extend_start_loc
        extend_seq_lens = forward_batch.extend_seq_lens
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
        # print("\nNew batch")
        # print(f"{extend_start_loc_cpu=}")
        # print(f"{extend_seq_lens=}")
        # print(f"{prefix_lens_cpu=}")

        assert bs == len(forward_batch.mm_inputs), "Batch size doesn't match the number of image inputs, each request can have only one image."
        for i, image_input in enumerate(forward_batch.mm_inputs):
            assert len(image_input.mm_items) == 1, "OpenVLA only supports single image inputs"
            if prefix_lens_cpu[i] > 256 + 1: # No need to patch if already computed
                continue
            mm_item = image_input.mm_items[0]
            image_data = Image.fromarray(mm_item.feature)
            pixel_value = self.processor.process_image(image_data).to(torch.bfloat16).to(0)
            patch_features = self.vision_backbone(pixel_value)
            projected_patch_embeddings = self.projector(patch_features)[0]

            relative_id_image_start = 1 - prefix_lens_cpu[i]
            relative_id_image_end = relative_id_image_start + 256
            
            # Supports chunked prefill
            id_start = max(pt + relative_id_image_start, extend_start_loc_cpu[i])
            id_end = min(pt + relative_id_image_end, extend_start_loc_cpu[i] + extend_seq_lens[i])
            length = id_end - id_start
            id_image_start = 0 if prefix_lens_cpu[i] == 0 else prefix_lens_cpu[i] - 1 
            # print(f"{id_start=} {id_end=} {length=} {id_image_start=} {id_image_start + length=}")
            input_embeddings[id_start : id_end] = (
                projected_patch_embeddings[id_image_start: id_image_start + length]
            )
            pt += forward_batch.extend_seq_lens_cpu[i]

        return self.language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeddings,
        )

EntryClass = OpenVLAForActionPrediction