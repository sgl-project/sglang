import logging
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING


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
