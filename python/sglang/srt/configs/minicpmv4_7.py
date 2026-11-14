# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from typing import Any

from sglang.srt.configs.minicpmv4_6 import MiniCPMV4_6Config


class MiniCPMV4_7Config(MiniCPMV4_6Config):
    model_type = "minicpmv4_7"

    # transformers 5 wraps config subclasses in ``@dataclass(kw_only=True)``;
    # its generated ``__init__`` would skip ``MiniCPMV4_6Config.__init__``.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


__all__ = ["MiniCPMV4_7Config"]
