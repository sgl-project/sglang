import importlib
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class RankParallelismConfig:
    """
    Complete parallelism configuration for a single inference rank.

    This configuration captures all the parallelism settings needed to recreate
    a model shard outside of sglang. It supports:
    - TP/PP/EP for model parallelism
    - MoE-TP/Attn-TP/Attn-DP for MoE and DP attention.
    """

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    attn_tp_size: int = 1
    attn_tp_rank: int = 0
    attn_dp_size: int = 1
    attn_dp_rank: int = 0

    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0

    @property
    def has_dp_attention(self) -> bool:
        """Check if DP attention is enabled."""
        return self.attn_dp_size > 1

    @property
    def has_expert_parallelism(self) -> bool:
        """Check if expert parallelism is enabled."""
        return self.ep_size > 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankParallelismConfig":
        """Create from dictionary, filtering unknown fields."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_parallel_state(cls, local_rank: int = 0) -> "RankParallelismConfig":
        """Extract current parallelism settings from distributed.parallel_state."""

        def get_val(module_name: str, func_name: str, default=0):
            try:
                module = importlib.import_module(module_name)
                return getattr(module, func_name)()
            except (ImportError, AttributeError, AssertionError):
                return default

        p_state = "sglang.srt.distributed.parallel_state"
        dp_attn = "sglang.srt.layers.dp_attention"

        tp_size = get_val(p_state, "get_tensor_model_parallel_world_size", 1)
        tp_rank = get_val(p_state, "get_tensor_model_parallel_rank", 0)

        return cls(
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=get_val(p_state, "get_pipeline_model_parallel_world_size", 1),
            pp_rank=get_val(p_state, "get_pipeline_model_parallel_rank", 0),
            ep_size=get_val(p_state, "get_moe_expert_parallel_world_size", 1),
            ep_rank=get_val(p_state, "get_moe_expert_parallel_rank", 0),
            moe_tp_size=get_val(p_state, "get_moe_tensor_parallel_world_size", tp_size),
            moe_tp_rank=get_val(p_state, "get_moe_tensor_parallel_rank", tp_rank),
            attn_tp_size=get_val(dp_attn, "get_attention_tp_size", tp_size),
            attn_tp_rank=get_val(dp_attn, "get_attention_tp_rank", tp_rank),
            attn_dp_size=get_val(dp_attn, "get_attention_dp_size", 1),
            attn_dp_rank=get_val(dp_attn, "get_attention_dp_rank", 0),
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            global_rank=dist.get_rank() if dist.is_initialized() else 0,
            local_rank=local_rank,
        )

    def __repr__(self) -> str:
        parts = [
            f"TP={self.tp_rank}/{self.tp_size}",
            f"PP={self.pp_rank}/{self.pp_size}",
        ]
        if self.has_expert_parallelism:
            parts.append(f"EP={self.ep_rank}/{self.ep_size}")
            parts.append(f"MoE-TP={self.moe_tp_rank}/{self.moe_tp_size}")
        if self.has_dp_attention:
            parts.append(f"AttnTP={self.attn_tp_rank}/{self.attn_tp_size}")
            parts.append(f"AttnDP={self.attn_dp_rank}/{self.attn_dp_size}")
        parts.append(f"Global={self.global_rank}/{self.world_size}")
        return f"RankParallelismConfig({', '.join(parts)})"
