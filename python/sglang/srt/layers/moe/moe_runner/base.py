from dataclasses import dataclass
from typing import Optional


@dataclass
class MoeRunnerConfig:
    activation: str = "silu"
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None
