from abc import ABC, abstractmethod

import torch
from torch._higher_order_ops import auto_functionalized

from sglang.srt.compilation.sglang_config import get_current_sglang_config
from sglang.srt.layers.layernorm import RMSNorm


def _get_default_op(ns: str, name: str):
    if not hasattr(torch.ops, ns):
        return None
    mod = getattr(torch.ops, ns)
    if not hasattr(mod, name):
        return None
    pkt = getattr(mod, name)  # OpOverloadPacket
    if hasattr(pkt, "default"):
        return pkt.default  # OpOverload (auto_functionalized need)
    if hasattr(pkt, "overloads") and pkt.overloads():
        return getattr(pkt, pkt.overloads()[0])
    return None


RMS_OP = _get_default_op("sgl_kernel", "rmsnorm") or _get_default_op("_C", "rmsnorm")
RMS_ADD_OP = _get_default_op("sgl_kernel", "fused_add_rmsnorm") or _get_default_op(
    "_C", "fused_add_rmsnorm"
)
ENABLE_PDL = False

# RMS_OP schema:
# sgl_kernel::rmsnorm(
#   Tensor($0! -> ) output,
#   Tensor input,
#   Tensor weight,
#   float eps,
#   bool enable_pdl
# ) -> ()
#
# RMS_ADD_OP schema:
# sgl_kernel::fused_add_rmsnorm(
#   Tensor($0! -> ) input,
#   Tensor($1! -> ) residual,
#   Tensor weight, float eps,
#   bool enable_pdl
# ) -> ()


class MatcherCustomOp(ABC):
    def __init__(self, enabled: bool):
        config = get_current_sglang_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config else None

        self.enabled = enabled
        self.forward = self.forward_custom if enabled else self.forward_native

    @abstractmethod
    def forward_custom(self, *args, **kws):
        pass

    @abstractmethod
    def forward_native(self, *args, **kws):
        pass

    def __call__(self, *args, **kws):
        return self.forward(*args, **kws)

    def empty(self, *args, **kws):
        return torch.empty(*args, dtype=self.model_dtype, device=self.device, **kws)

    def empty_f32(self, *args, **kws):
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kws)

    def inputs(self) -> list[torch.Tensor]:
        """Utility for inputs to the pattern"""
        raise NotImplementedError


class MatcherRMSNorm(MatcherCustomOp):
    def __init__(self, epsilon: float, enabled: bool | None = None):
        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon

    def inputs(self):
        input = self.empty(5, 16) if self.enabled else self.empty_f32(5, 16)
        weight = self.empty(16)
        return [input, weight]

    def forward_custom(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(input)
        ret = auto_functionalized(
            RMS_OP,
            output=out,
            input=input,
            weight=weight,
            eps=self.epsilon,
            enable_pdl=ENABLE_PDL,
        )

        _, out = ret
        return out

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight
        )


class MatcherFusedAddRMSNorm(MatcherCustomOp):
    def __init__(self, epsilon: float, enabled: bool | None = None):
        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon

    def inputs(self):
        input = self.empty(5, 16) if self.enabled else self.empty_f32(5, 16)
        weight = self.empty(16)
        residual = self.empty(5, 16)
        return [input, weight, residual]

    def forward_custom(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ret = auto_functionalized(
            RMS_ADD_OP,
            input=input,
            residual=residual,
            weight=weight,
            eps=self.epsilon,
            enable_pdl=ENABLE_PDL,
        )

        _, new_input, new_residual = ret
        return new_input, new_residual

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight, residual
        )
