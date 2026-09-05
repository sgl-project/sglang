"""OpAuto — optional runtime policy over KernelSpec / BaseFusedOp backends.

Disabled by default (``SGLANG_OPAUTO=0``). When enabled, applies cold-start
JIT avoidance, sticky demotion on probe/launch failure, and optional
persisted state under the JIT cache directory.
"""

from sglang.kernels.opauto.policy import (
    OpAutoPolicy,
    can_use_or_demote,
    enable_opauto,
    get_policy,
    is_enabled,
    set_cold_skip_jit,
    should_prefer_native_aot_fallback,
    should_skip_cold_jit,
    status_payload,
)
from sglang.kernels.opauto.state import get_state

__all__ = [
    "OpAutoPolicy",
    "can_use_or_demote",
    "enable_opauto",
    "get_policy",
    "get_state",
    "is_enabled",
    "set_cold_skip_jit",
    "should_prefer_native_aot_fallback",
    "should_skip_cold_jit",
    "status_payload",
]
