from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig


class Step3p7VisionEncoderConfig(PretrainedConfig):
    model_type = "perception_encoder"

    def __init__(
        self,
        width=1536,
        layers=47,
        heads=16,
        num_channels=3,
        image_size=728,
        patch_size=14,
        mlp_ratio=8960 / 1536,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        use_cls_token=False,
        use_ln_pre=True,
        use_ln_post=False,
        use_abs_posemb=True,
        use_rope2d=True,
        ls_init_value=0.1,
        output_dim=None,
        pool_type="none",
        **kwargs,
    ):
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.ls_init_value = ls_init_value
        self.output_dim = output_dim
        self.pool_type = pool_type
        super().__init__(**kwargs)


class Step3p7Config(PretrainedConfig):
    model_type = "step3p7"

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step3p7VisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, PretrainedConfig]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        image_token_len: int = 169,
        patch_token_len: int = 81,
        im_start_token: str = "<im_start>",
        im_end_token: str = "<im_end>",
        im_patch_token: str = "<im_patch>",
        use_im_start_end: bool = True,
        vision_select_layer: int = -1,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = Step3p7VisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = Step3p7VisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            from sglang.srt.configs.step3p5 import Step3p5Config

            text_config = Step3p5Config()
        elif isinstance(text_config, dict):
            from sglang.srt.configs.step3p5 import Step3p5Config

            text_config = Step3p5Config(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id
        self.image_token_len = image_token_len
        self.patch_token_len = patch_token_len
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer

        super().__init__(**kwargs)
