import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Eagle2_5_VLConfig(PretrainedConfig):
    model_type = "eagle_2_5_vl"
    is_composition = True
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": Qwen2Config}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-4,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        loss_version="v1",
        min_dynamic_tiles=1,
        max_dynamic_tiles=6,
        mlp_checkpoint=False,
        image_token_index=151667,
        use_pixel_shuffle=True,
        mlp_connector_layers=2,
        llm_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"model_type": "siglip_vision_model"}
            logger.info(
                "vision_config is None. Initializing the SiglipVisionConfig with default values."
            )

        if llm_config is not None:
            text_config = llm_config
            logger.info("Using llm_config as text_config for Eagle2.5-VL.")

        if text_config is None:
            text_config = {"architectures": ["Qwen2ForCausalLM"]}
            logger.info(
                "text_config is None. Initializing the Qwen2Config with default values."
            )

        if vision_config["model_type"] != "siglip_vision_model":
            raise ValueError(f"Unsupported model_type: {vision_config['model_type']}")
        self.vision_config = SiglipVisionConfig(**vision_config)

        arch = text_config["architectures"][0]
        if arch == "Qwen2ForCausalLM":
            self.text_config = Qwen2Config(**text_config)
        elif arch == "Qwen3ForCausalLM":
            self.text_config = Qwen3Config(**text_config)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.loss_version = loss_version
        self.min_dynamic_tiles = min_dynamic_tiles
        self.max_dynamic_tiles = max_dynamic_tiles
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index
        self.use_pixel_shuffle = use_pixel_shuffle
        self.mlp_connector_layers = mlp_connector_layers

        logger.info("min_dynamic_tiles: %s", self.min_dynamic_tiles)
        logger.info("max_dynamic_tiles: %s", self.max_dynamic_tiles)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["pad2square"] = self.pad2square
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["min_dynamic_tiles"] = self.min_dynamic_tiles
        output["max_dynamic_tiles"] = self.max_dynamic_tiles
        output["tie_word_embeddings"] = self.tie_word_embeddings
        output["image_token_index"] = self.image_token_index
        if hasattr(self, "_attn_implementation"):
            output["_attn_implementation"] = self._attn_implementation
        if hasattr(self, "_attn_implementation_autoset"):
            output["_attn_implementation_autoset"] = self._attn_implementation_autoset
        output["use_pixel_shuffle"] = self.use_pixel_shuffle
        output["mlp_connector_layers"] = self.mlp_connector_layers
        return output
