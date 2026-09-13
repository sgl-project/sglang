"""Construct a small real DeepSeek MoE forward around controlled routing.

Model initialization and router logits are fixtures. Shared MLP, dispatcher,
MoeRunner, expert GEMMs, weighting and model composition execute production code.
"""

import torch

from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


class RoutedExperts(torch.nn.Module):
    def __init__(self, dispatcher, quant, config, fused_scaling):
        super().__init__()
        self.dispatcher, self.quant = dispatcher, quant
        self.should_fuse_routed_scaling_factor_in_topk = fused_scaling
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, config)

    def forward(self, hidden_states, topk_output):
        dispatched = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        return self.dispatcher.combine(combine_input=self.run_moe_core(dispatched))

    def run_moe_core(self, dispatch_output):
        return self.runner.run(dispatch_output, self.quant)

    def set_overlap_args(self, **kwargs):
        self.runner.set_overlap_args(**kwargs)

    def clear_overlap_args(self):
        self.runner.clear_overlap_args()


def make_moe(
    dispatcher,
    quant,
    config,
    shared,
    router,
    *,
    scale=2.5,
    fused_scaling=False,
    sbo=False
):
    model = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(model)
    model.shared_experts = shared
    model._nccl_ep_shared_experts_on_current_stream = True
    model.alt_stream = torch.cuda.Stream()
    model._fuse_shared_experts_inside_sbo = sbo
    model.is_nextn = False
    model.num_fused_shared_experts = 0
    model.layer_id = config.layer_id
    model.ep_size = (
        dispatcher._inners[0].world_size
        if hasattr(dispatcher, "_inners")
        else dispatcher.world_size
    )
    model.routed_scaling_factor = scale
    model.gate = lambda x, **kwargs: torch.zeros(
        len(x), config.num_experts, device=x.device
    )
    model.experts = RoutedExperts(dispatcher, quant, config, fused_scaling)
    model.topk = router
    return model
