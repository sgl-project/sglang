from __future__ import annotations

from dataclasses import dataclass


from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobResultProcessor:
    server_args: ServerArgs
    model_config: ModelConfig
