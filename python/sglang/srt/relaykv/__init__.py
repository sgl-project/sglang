"""RelayKV experimental integration for SGLang.

MVP-0 is shadow-only: it computes resident/cold KV plans and logs metrics,
but it must not move KV tensors or alter attention behavior.
"""

from .config import RelayKVConfig
from .planner import RelayKVPlan, build_shadow_plan, make_shadow_plan

__all__ = ["RelayKVConfig", "RelayKVPlan", "build_shadow_plan", "make_shadow_plan"]
