"""A DeepseekV2MoE with fake experts for the DeepSeek-V4 reduction tests: the routed
output is x * (rank + 1) and the shared output x * 0.5, so a reduction is visible
in the result and a skipped one is too."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.topk import TopKOutputFormat
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def make_dsv4_moe_stub(
    rank: int, *, dual: bool, shared_tp1: bool, tp_size: int = 4
) -> DeepseekV2MoE:
    class Experts:
        quant_method = None
        moe_runner_config = SimpleNamespace(inplace=False)

        def __call__(self, x, *args, **kwargs):
            return x * (rank + 1)

    class Moe(DeepseekV2MoE):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.tp_size = tp_size
            self.is_deepseek_v4 = True
            self._shared_expert_tp1 = shared_tp1
            self.layer_id = 0
            self.is_nextn = False
            self.is_hash = False
            self._fuse_shared_experts_inside_sbo = False
            self._fuse_finalize_all_reduce = False
            self.num_fused_shared_experts = 0
            self.routed_scaling_factor = 1.0
            self.experts = Experts()
            self.alt_stream = torch.cuda.Stream()
            self.topk = lambda *a, **kw: SimpleNamespace(
                format=TopKOutputFormat.STANDARD
            )

        def _maybe_quant_moe_input_once(self, x):
            return None

        def _should_quant_routed_input_mxfp8(self, x):
            return False

        def _forward_gate(self, x, *args, **kwargs):
            return x, None

        def _forward_shared_experts(self, x, *args, **kwargs):
            return x * 0.5

        def forward(self, x, *args, **kwargs):
            return (self.forward_normal_dual_stream if dual else self.forward_normal)(x)

    return Moe()
