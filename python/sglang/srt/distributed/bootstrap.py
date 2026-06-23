from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchDistributedResult:
    tp_group: object
    pp_group: object
    attention_tp_group: object
    pre_model_load_memory: float
