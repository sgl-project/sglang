# SPDX-License-Identifier: Apache-2.0
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/configuration_kimi_vl.py

from transformers import DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig


class K2VLConfig(PretrainedConfig):
    """K2VL model configuration.
    Args:
        text_config (dict | DeepseekV3Config): Configuration for the text model.

        Vision Tower Parameters (from MoonViT3dConfig):
            patch_size (int): Patch size for vision tower.
            init_pos_emb_height (int): Initial position embedding height.
            init_pos_emb_width (int): Initial position embedding width.
            init_pos_emb_time (int): Initial position embedding time dimension.
            pos_emb_type (str): Type of position embedding.
            vt_num_attention_heads (int): Number of attention heads in vision tower.
            vt_num_hidden_layers (int): Number of hidden layers in vision tower.
            vt_hidden_size (int): Hidden size of vision tower.
            vt_intermediate_size (int): Intermediate size in vision tower FFN.
            merge_kernel_size (tuple): Kernel size for patch merging.
            video_attn_type (str): Type of video attention.
            merge_type (str): Type of merge operation.
            _attn_implementation (str): Attention implementation type.

        MM Projector Parameters (from MultiModalProjectorConfig):
            mm_projector_type (str): Type of multimodal projector.
            mm_hidden_size (int): Hidden size from vision tower (should match vt_hidden_size).
            projector_hidden_act (str): Activation function for projector.
            projector_ln_eps (float): Layer norm epsilon for projector.

        Other Parameters:
            ignore_index (int): The ignore index for the loss function.
            media_placeholder_token_id (int): The token ID to use for media placeholders.
            pad_token_id (int): The token ID to use for padding.
    """

    model_type = "k2_vl"

    def __init__(
        self,
        text_config: dict | DeepseekV3Config = None,
        # Vision tower parameters (from MoonViT3dConfig)
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 16,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1152,
        vt_intermediate_size: int = 4304,
        merge_kernel_size: tuple = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        # MM Projector parameters (from MultiModalProjectorConfig)
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = True,
        video_placeholder="<|k2vl_video_placeholder|>",
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)

        self.text_config = text_config

        # Vision tower config
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps

        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)
