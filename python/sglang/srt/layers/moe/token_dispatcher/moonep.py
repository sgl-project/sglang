from __future__ import annotations

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode


_MOONEP_UNSUPPORTED_MESSAGE = (
    "MoonEP MoE A2A is recognized by SGLang, but the runtime dispatcher is not "
    "implemented yet. MoonEP is not a drop-in DeepEP-compatible backend: it "
    "returns a MoonEPCommPlan/cu_seqlens and requires MoonEP-compatible "
    "contiguous symmetric-memory expert weights plus a VM-group expert GEMM."
)


class MoonEPDispatcher(BaseDispatcher):
    """Placeholder dispatcher for MoonEP.

    This keeps backend selection explicit while preventing accidental execution
    through the DeepEP/Mooncake/NIXL dispatcher contracts, which use different
    dispatch output formats from MoonEP.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.async_finish = async_finish
        self.return_recv_hook = return_recv_hook
        self.expert_mask_gpu = None

    @staticmethod
    def _raise_unimplemented():
        raise NotImplementedError(_MOONEP_UNSUPPORTED_MESSAGE)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self._raise_unimplemented()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._raise_unimplemented()

    def dispatch_b(self):
        self._raise_unimplemented()

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        self._raise_unimplemented()

    def combine_b(self):
        self._raise_unimplemented()

    def register_deepep_dispatch_hook(self, hook):
        self._raise_unimplemented()
