from __future__ import annotations

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES

# Default fixture sizes — small enough for fast tests, large enough that ring overflow / multi-req cases
# stay realistic without bloating the assertion surface.
DEFAULT_RING_CAPACITY: int = 64
DEFAULT_NUM_SLOTS: int = 32
DEFAULT_SLOT_STRIDE_BYTES: int = CANARY_SLOT_BYTES

_U64_MASK: int = (1 << 64) - 1
_I64_SIGN_BIT: int = 1 << 63
