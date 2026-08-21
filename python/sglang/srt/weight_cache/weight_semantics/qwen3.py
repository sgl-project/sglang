from __future__ import annotations

from .qwen3_5 import (
    Qwen35WeightSemanticsAdapter,
)


class Qwen3WeightSemanticsAdapter(Qwen35WeightSemanticsAdapter):
    def _has_attention_output_gate(self) -> bool:
        return bool(getattr(self._config, "attn_output_gate", False))
