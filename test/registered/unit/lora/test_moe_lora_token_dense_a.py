"""The token_dense LoRA-A family: one batched GEMM over the token domain for
every resident adapter slot, written as one bridge plane per slot."""

import unittest

import torch

from sglang.srt.lora.moe.execution_plan import (
    BridgeLayout,
    LoraAFamily,
    LoraASpec,
    Site,
)
from sglang.srt.lora.moe.kernels.lora_a import run_lora_a
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class TestTokenDenseLoraA(unittest.TestCase):
    def test_matches_a_per_slot_matmul(self):
        torch.manual_seed(0)
        slots, tokens, hidden, rank2 = 3, 5, 16, 8
        hidden_states = torch.randn(tokens, hidden)
        weight = torch.randn(slots, rank2, hidden)  # shared-outer: one factor per slot
        output = torch.empty(slots * tokens, rank2)
        route = RouteView(
            view=RouteViewKind.RAW,
            block_size=16,
            topk_ids=torch.zeros(tokens, 2, dtype=torch.int32),
            token_lora_mapping=torch.zeros(tokens, dtype=torch.int32),
            num_local_experts=4,
            is_shared_outer=True,
            max_loras=slots,
        )
        spec = LoraASpec(
            Site.GATE_UP,
            LoraAFamily.TOKEN_DENSE,
            is_shared_outer=True,
            output_layout=BridgeLayout.TOKEN_MAJOR,
        )
        out = run_lora_a(
            spec,
            input=hidden_states,
            weight=weight,
            output=output,
            routing=route,
            config={},
        )
        expected = torch.stack([hidden_states @ weight[s].T for s in range(slots)])
        torch.testing.assert_close(out.view(slots, tokens, rank2), expected)


if __name__ == "__main__":
    unittest.main()
