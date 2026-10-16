# SPDX-License-Identifier: Apache-2.0
"""Native configuration for the released Qwen3.5 + Mixer Valen checkpoints."""

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
        super().__init__(**kwargs)
        self.valen_head = valen_head or {"head_type": "mixer"}
        self.qwen_execution = qwen_execution
        self.valen_max_length = valen_max_length
        self.valen_media_kwargs = valen_media_kwargs or {}
        self.valen_backbone_autocast_dtype = valen_backbone_autocast_dtype
        if qwen_execution != "shared_state":
            raise ValueError(
                "SGLang Valen serving requires qwen_execution=shared_state"
            )
        if self.valen_head.get("head_type") != "mixer":
            raise ValueError("SGLang Valen serving supports the Mixer decision head")
        if type(valen_max_length) is not int or valen_max_length < 1:
            raise ValueError("valen_max_length must be a positive integer")


def is_valen_config(config) -> bool:
    return getattr(config, "architectures", None) == ["ValenQwenForDecisionMaking"]
