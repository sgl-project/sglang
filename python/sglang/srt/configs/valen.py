"""Native configuration for Valen's Qwen3.5 decision checkpoints."""

from sglang.srt.configs.qwen3_5 import Qwen3_5Config


class ValenQwenConfig(Qwen3_5Config):
    model_type = "valen_qwen"

    def __init__(
        self,
        valen_head=None,
        qwen_execution="shared_state",
        valen_max_length=32768,
        valen_media_kwargs=None,
        valen_backbone_autocast_dtype=None,
        **kwargs,
    ):
        if qwen_execution != "shared_state":
            raise ValueError("SGLang supports Valen qwen_execution='shared_state' only")
        self.valen_head = valen_head or {"head_type": "mixer"}
        if self.valen_head.get("head_type", "mixer") != "mixer":
            raise ValueError("SGLang supports Valen mixer heads only")
        self.qwen_execution = qwen_execution
        self.valen_max_length = valen_max_length
        self.valen_media_kwargs = valen_media_kwargs or {}
        self.valen_backbone_autocast_dtype = valen_backbone_autocast_dtype
        super().__init__(**kwargs)
