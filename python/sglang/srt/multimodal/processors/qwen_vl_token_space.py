from sglang.srt.multimodal.processors.token_space_processor import (
    MultimodalProcessorOutputBuilder,
)


class QwenVLMultimodalProcessorOutputBuilder(MultimodalProcessorOutputBuilder):
    """Qwen VL serving output: padded ids, vision boundaries and M-RoPE positions."""

    def __init__(self, hf_config):
        model_type = hf_config.model_type
        if model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config
        super().__init__(hf_config)
        self.model_type = model_type
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)
        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self._spatial_merge_size = hf_config.vision_config.spatial_merge_size
        self._tokens_per_second = getattr(
            hf_config.vision_config, "tokens_per_second", None
        )
