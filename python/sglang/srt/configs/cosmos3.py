# SPDX-License-Identifier: Apache-2.0
"""Configuration for the Cosmos3 Reasoner (understanding tower).

The Cosmos3 unified checkpoint stores a Qwen3-VL understanding tower alongside
a generation (diffusion) tower. The Reasoner only serves the understanding
tower, so it reuses the Qwen3-VL config schema and just declares its own
``model_type`` so ``AutoConfig`` can resolve the checkpoint.
"""

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig


class Cosmos3Config(Qwen3VLConfig):
    model_type = "cosmos3_omni"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Qwen3-VL inference stack accesses ``config.vision_config`` and
        # ``config.text_config`` as objects (e.g. ``config.vision_config.hidden_size``,
        # ``config.vision_config.deepstack_visual_indexes``). Some transformers
        # versions leave these sub-configs
        # as raw dicts after construction, which would raise
        # ``'dict' object has no attribute 'hidden_size'`` at model init. Coerce
        # any dict-valued sub-config into its proper config object so the model
        # loads regardless of the installed transformers version.
        for attr, sub_cls in self.sub_configs.items():
            sub = getattr(self, attr, None)
            if isinstance(sub, dict):
                setattr(self, attr, sub_cls(**sub))
