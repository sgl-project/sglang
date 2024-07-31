from dataclasses import dataclass
from enum import IntEnum, auto


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()


@dataclass
class ForwardBatch:
    """Model forward information for a batch."""

    pass
