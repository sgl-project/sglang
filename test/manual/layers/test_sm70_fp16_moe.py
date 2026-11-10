"""SM70 FP16 MoE must honor the routed scaling factor and the SwiGLU clamp.

Run on a V100: python -m pytest test/manual/layers/test_sm70_fp16_moe.py
"""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_context

EXPERTS, HIDDEN, INTER, TOPK, TOKENS = 16, 512, 256, 4, 3


def _sm70_ops_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
        return False
    from sglang.srt.layers.quantization.sm70_fp16_moe import can_use_sm70_fp16_moe

    return can_use_sm70_fp16_moe(torch.float16)


def _reference(x, w13, w2, topk_weights, topk_ids, routed_scale, swiglu_limit):
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            gate_up = x[t].double() @ w13[e].double().T
            gate, up = gate_up[:INTER], gate_up[INTER:]
            if swiglu_limit is not None:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            act = F.silu(gate) * up
            out[t] += float(topk_weights[t, k]) * (act @ w2[e].double().T)
    return out * (routed_scale if routed_scale is not None else 1.0)


@unittest.skipUnless(_sm70_ops_available(), "needs a V100 with the TurboMind ops")
class TestSM70FP16MoE(unittest.TestCase):
    def _run(self, *, routed_scale, swiglu_limit, expect_turbomind):
        from sglang.srt.layers.quantization.sm70_fp16_moe import SM70FP16MoEMethod

        torch.manual_seed(0)
        dev = "cuda"
        w13 = (torch.randn(EXPERTS, 2 * INTER, HIDDEN, device=dev) * 0.05).half()
        w2 = (torch.randn(EXPERTS, HIDDEN, INTER, device=dev) * 0.05).half()
        # Large inputs push some gate/up values past the clamp.
        x = (torch.randn(TOKENS, HIDDEN, device=dev) * 6.0).half()
        topk_ids = torch.stack(
            [torch.randperm(EXPERTS, device=dev)[:TOPK] for _ in range(TOKENS)]
        ).int()
        topk_weights = torch.softmax(torch.randn(TOKENS, TOPK, device=dev), -1)
        ref = _reference(
            x, w13, w2, topk_weights, topk_ids, routed_scale, swiglu_limit
        )
        if swiglu_limit is not None:
            unclamped = _reference(
                x, w13, w2, topk_weights, topk_ids, routed_scale, None
            )
            self.assertGreater(float((unclamped - ref).norm() / ref.norm()), 0.05)

        layer = nn.Module()
        layer.w13_weight = nn.Parameter(w13.clone(), requires_grad=False)
        layer.w2_weight = nn.Parameter(w2.clone(), requires_grad=False)
        layer.should_fuse_routed_scaling_factor_in_topk = False
        config = MoeRunnerConfig(
            num_experts=EXPERTS,
            num_local_experts=EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTER,
            top_k=TOPK,
            params_dtype=torch.float16,
            routed_scaling_factor=routed_scale,
            swiglu_limit=swiglu_limit,
        )
        method = SM70FP16MoEMethod()
        method.create_moe_runner(layer, config)
        method.process_weights_after_loading(layer)

        dispatch = StandardDispatchOutput(
            hidden_states=x.clone(),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
            ),
        )
        # The Triton fallback reads the published execution config; it runs
        # in place, so the input is a copy.
        with get_context().override_server_args():
            out = method.apply(layer, dispatch).hidden_states.double()
        rel = float((out - ref).norm() / ref.norm())
        self.assertLess(rel, 5e-3)
        self.assertEqual(method.use_turbomind, expect_turbomind)

    def test_plain_silu_uses_turbomind(self):
        self._run(routed_scale=None, swiglu_limit=None, expect_turbomind=True)

    def test_routed_scaling_factor_is_applied(self):
        self._run(routed_scale=2.5, swiglu_limit=None, expect_turbomind=True)

    def test_clamped_swiglu_falls_back(self):
        self._run(routed_scale=2.5, swiglu_limit=10.0, expect_turbomind=False)


if __name__ == "__main__":
    unittest.main()
