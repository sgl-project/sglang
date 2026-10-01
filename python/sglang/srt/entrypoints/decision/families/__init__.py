from typing import Any, Optional, Tuple, Type

from sglang.srt.entrypoints.decision.families.base import DecisionModelFamily
from sglang.srt.entrypoints.decision.families.intern import InternDecisionFamily

FAMILIES: Tuple[Type[DecisionModelFamily], ...] = (InternDecisionFamily,)


def detect_family(tokenizer: Any) -> Optional[DecisionModelFamily]:
    for family in FAMILIES:
        if family.detect(tokenizer):
            return family(tokenizer)
    return None
