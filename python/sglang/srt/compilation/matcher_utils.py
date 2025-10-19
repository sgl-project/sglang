from abc import ABC, abstractmethod

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from torch._higher_order_ops import auto_functionalized

from sglang.srt.configs.sglang_config import get_current_sglang_config
from sglang.srt.layers.layernorm import RMSNorm

RMS_OP = rmsnorm
RMS_ADD_OP = fused_add_rmsnorm


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
        result = torch.empty_like(input)
        _, result = auto_functionalized(
            RMS_OP,
            result=result,
            input=input,
            weight=weight,
            epsilon=self.epsilon,
        )

        return result

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
        _, result, residual = auto_functionalized(
            RMS_ADD_OP,
            input=input,
            residual=residual,
            weight=weight,
            epsilon=self.epsilon,
        )

        return result, residual

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight, residual
        )
