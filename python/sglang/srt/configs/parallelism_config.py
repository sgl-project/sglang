import importlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class RankParallelismConfig:
    """
    Complete parallelism configuration for a single inference rank.
    """

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    attn_dp_size: Optional[int] = None
    attn_dp_rank: Optional[int] = None

    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0

    def __post_init__(self):
        expected_world_size = self.tp_size * self.pp_size
        if self.world_size > 1:
            assert (
                self.world_size >= expected_world_size
            ), f"world_size {self.world_size} must be >= tp_size * pp_size ({expected_world_size})"

        # Set attention defaults if not provided
        if self.attn_tp_size is None:
            self.attn_tp_size = self.tp_size
        if self.attn_tp_rank is None:
            self.attn_tp_rank = self.tp_rank
        if self.attn_dp_size is None:
            self.attn_dp_size = 1
        if self.attn_dp_rank is None:
            self.attn_dp_rank = 0

    @property
    def has_dp_attention(self) -> bool:
        return self.attn_tp_size < self.tp_size or self.attn_dp_size > 1

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "RankParallelismConfig":
        return cls(**data)

    @classmethod
    def from_parallel_state(cls, local_rank: int = 0) -> "RankParallelismConfig":
        """
        Extract current parallelism settings from distributed.parallel_state.
        """

        def get_val(module_name: str, func_name: str, default=0):
            try:
                module = importlib.import_module(module_name)
                return getattr(module, func_name)()
            except (ImportError, AttributeError):
                return default

        p_state = "sglang.srt.distributed.parallel_state"
        dp_attn = "sglang.srt.layers.dp_attention"

        return cls(
            tp_size=get_val(p_state, "get_tensor_model_parallel_world_size", 1),
            tp_rank=get_val(p_state, "get_tensor_model_parallel_rank", 0),
            pp_size=get_val(p_state, "get_pipeline_model_parallel_world_size", 1),
            pp_rank=get_val(p_state, "get_pipeline_model_parallel_rank", 0),
            ep_size=get_val(p_state, "get_expert_model_parallel_world_size", 1),
            ep_rank=get_val(p_state, "get_expert_model_parallel_rank", 0),
            attn_tp_size=get_val(dp_attn, "get_attention_tp_size", None),
            attn_tp_rank=get_val(dp_attn, "get_attention_tp_rank", None),
            attn_dp_size=get_val(dp_attn, "get_attention_dp_size", 1),
            attn_dp_rank=get_val(dp_attn, "get_attention_dp_rank", 0),
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            global_rank=dist.get_rank() if dist.is_initialized() else 0,
            local_rank=local_rank,
        )

    def __repr__(self) -> str:
        return (
            f"RankParallelismConfig (TP={self.tp_rank}/{self.tp_size}, "
            f"PP={self.pp_rank}/{self.pp_size}, "
            f"EP={self.ep_rank}/{self.ep_size}, "
            f"Global={self.global_rank}/{self.world_size})"
        )
