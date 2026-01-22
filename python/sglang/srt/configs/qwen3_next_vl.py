from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig

class Qwen3NextVLVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5"
    base_config_key = "vision_config"

class Qwen3NextVLTextConfig(Qwen3NextConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"


class Qwen3NextVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLMoeModel`]. It is used to instantiate a
    Qwen3-VL-MOE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-VL-30B-A3B-Instruct [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLMoeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLMoeVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeConfig

    >>> # Initializing a Qwen3-VL-MOE style configuration
    >>> configuration = Qwen3VLMoeConfig()

    >>> # Initializing a model from the Qwen3-VL-30B-A3B style configuration
    >>> model = Qwen3VLMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_5"
    sub_configs = {
        "vision_config": Qwen3NextVLVisionConfig,
        "text_config": Qwen3NextVLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)



class Qwen3NextVLMoEVisionConfig(Qwen3NextVLVisionConfig):
    model_type = "qwen3_5_moe"

class Qwen3NextVLMoETextConfig(Qwen3NextVLTextConfig):
    model_type = "qwen3_5_moe_text"

class Qwen3NextVLMoEConfig(Qwen3NextVLConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3NextVLMoEVisionConfig,
        "text_config": Qwen3NextVLMoETextConfig,
    }