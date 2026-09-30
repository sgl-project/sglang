# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sglang-side ``PretrainedConfig`` class for MiniCPM-V 4.7.

4.7 reuses the 4.6 stack unchanged — SigLip NaViT tower with a mid-ViT 2x2
window-attention merger, an MLP connector and a Qwen3.5 hybrid backbone — so
only the version it announces differs.
"""

from typing import Any

from sglang.srt.configs.minicpmv4_6 import MiniCPMV4_6Config


class MiniCPMV4_7Config(MiniCPMV4_6Config):
    model_type = "minicpmv4_7"

    # Explicit ``__init__``: transformers 5 wraps ``PretrainedConfig``
    # subclasses with ``@dataclass(kw_only=True)``, whose generated ``__init__``
    # would bypass ``MiniCPMV4_6Config.__init__`` and leave the sub-configs as
    # raw dicts.
    def __init__(
        self,
        version: float = 4.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.version = version


__all__ = ["MiniCPMV4_7Config"]
