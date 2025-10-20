from transformers import AutoProcessor, PretrainedConfig
from transformers.processing_utils import ProcessingKwargs

try:
    from transformers import Qwen2_5_VLProcessor
except ImportError:
    raise ImportError(
        "Qwen2_5_VLProcessor can not be found. Please upgrade your transformers version."
    )

from sglang.srt.configs.deepseekvl2 import DeepseekV2Config


class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,  # vision encoder embed size
        hidden_size: int = 1536,  # after merger hidden size
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation="flash_attention_2",  # "eager","sdpa","flash_attention_2"
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,  # ve causal forward
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


class DotsVLMConfig(PretrainedConfig):
    model_type = "dots_vlm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.im_span_id = kwargs.get("image_token_id", 128815)
        self.video_span_id = kwargs.get("video_token_id", 128836)
        self.vision_config = DotsVisionConfig(**vision_config)
        self.language_config = DeepseekV2Config(**kwargs)
        self.architectures = ["DotsVLMForCausalLM"]


class DotsVLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class DotsVLMProcessor(Qwen2_5_VLProcessor):
    r"""
    Constructs a DotsVLM processor which derives from Qwen2_5_VLProcessor, but overrides the image and video token ids.
    Besides, its tokenizer is a LlamaTokenizerFast instead of Qwen2TokenizerFast.
    [`DotsVLMProcessor`] offers all the functionalities of [`DotsVisionConfig`] and [`LlamaTokenizerFast`]. See the
    [`~DotsVLMProcessor.__call__`] and [`~DotsVLMProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]

    valid_kwargs = ["chat_template"]

    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = (
            "<|imgpad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.img_token = (
            "<|img|>" if not hasattr(tokenizer, "img_token") else tokenizer.img_token
        )
        self.endofimg_token = (
            "<|endofimg|>"
            if not hasattr(tokenizer, "endofimg_token")
            else tokenizer.endofimg_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.encode(self.image_token)[0]
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.encode(self.video_token)[0]
        )


AutoProcessor.register(DotsVLMConfig, DotsVLMProcessor)
