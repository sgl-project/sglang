"""Construct a small real DeepSeek MoE forward around controlled routing.

Model initialization and router logits are fixtures. Shared MLP, dispatcher,
MoeRunner, expert GEMMs, weighting and model composition execute production code.
"""

from types import SimpleNamespace

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
    model.ep_size = dispatcher.world_size
    model.routed_scaling_factor = scale
    model.gate = lambda x, **kwargs: torch.zeros(
        len(x), config.num_experts, device=x.device
    )
    model.experts = RoutedExperts(dispatcher, quant, config, fused_scaling)
    model.topk = router
    return model


def metadata(mapping, *, ep_size=1, rank=0, device="cuda", num_logical=None):
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    physical = torch.tensor(mapping, dtype=torch.int64, device=device)
    logical = num_logical or (max(max(row) for row in mapping) + 1)
    inverse = torch.full(
        (len(mapping), logical, len(mapping[0])), -1, dtype=torch.int64
    )
    for layer, row in enumerate(mapping):
        for expert in range(logical):
            slots = [slot for slot, value in enumerate(row) if value == expert]
            inverse[layer, expert, : len(slots)] = torch.tensor(slots)
    return ExpertLocationMetadata._init_raw(
        server_args=SimpleNamespace(
            ep_dispatch_algorithm="static", ep_join_mode=None, nnodes=1, ep_size=ep_size
        ),
        ep_size=ep_size,
        physical_to_logical_map=physical,
        logical_to_all_physical_map=inverse.to(device),
        moe_ep_rank=rank,
    )
