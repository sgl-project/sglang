import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_mod
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS = 128
NUM_ROUTED_TOPK = 4
NUM_FUSED_SHARED = 1
TOP_K = NUM_ROUTED_TOPK + NUM_FUSED_SHARED


def _fake_gate(**kwargs):
    """Stand-in for moe_fused_gate, honouring its width contract.

    The kernel (kernels/jit/csrc/moe/moe_fused_gate.cuh) derives
    `topk_routed = topk - num_fused_shared_experts` and fills the remaining
    columns with id `num_experts + (slot - topk_routed)`, so its `topk`
    argument is the TOTAL width, not the routed width.
    """
    gating_output = kwargs["gating_output"]
    topk = kwargs["topk"]
    num_shared = kwargs["num_fused_shared_experts"]
    # The kernel enforces the same thing via RuntimeCheck before launching.
    assert topk > num_shared, "moe_fused_gate requires topk > num_fused_shared_experts"
    m, n = gating_output.shape
    k_routed = topk - num_shared
    ids = torch.arange(k_routed, dtype=torch.int32).repeat(m, 1)
    if num_shared:
        shared = (n + torch.arange(num_shared, dtype=torch.int32)).repeat(m, 1)
        ids = torch.cat([ids, shared], dim=-1)
    return torch.ones(m, topk, dtype=torch.float32), ids


def _fake_append(topk_ids, topk_weights, num_shared, scale_factor, base_id):
    m = topk_ids.shape[0]
    return (
        torch.cat(
            [topk_ids, torch.full((m, num_shared), base_id, dtype=topk_ids.dtype)],
            dim=-1,
        ),
        torch.cat(
            [
                topk_weights,
                torch.full((m, num_shared), scale_factor, dtype=topk_weights.dtype),
            ],
            dim=-1,
        ),
    )


class TestAiterFusedSharedExpertsAppend(CustomTestCase):
    """The aiter router must emit K_routed routed experts, not K_routed - 1.

    select_experts asks the ungrouped gate for `top_k - num_fused_shared_experts`
    columns on the aiter path because _post_process_topk_ids appends the shared
    slot afterwards. Naming the shared expert to that gate as well makes it
    reserve one of those columns for the shared marker, so the layer routes one
    real expert fewer and weights the shared expert twice. The final width is
    top_k either way, so nothing raises -- only the routing is wrong.
    """

    def _route(self):
        config = TopKConfig(
            top_k=TOP_K,
            renormalize=True,
            scoring_func="sigmoid",
            num_fused_shared_experts=NUM_FUSED_SHARED,
            correction_bias=torch.zeros(NUM_EXPERTS, dtype=torch.float32),
            routed_scaling_factor=2.0,
        )
        # The append is reached on any host: _aiter_append reads only
        # num_fused_shared_experts and _use_aiter, not _is_hip. _is_cpu is forced
        # off because it is the only gate on the JIT gate dispatch.
        fake_kernels = types.ModuleType(
            "sglang.kernels.ops.moe.fused_moe_triton_kernels"
        )
        fake_kernels.fused_append_shared_experts = _fake_append
        with (
            patch.object(topk_mod, "_use_aiter", True),
            patch.object(topk_mod, "_is_cpu", False),
            patch.object(topk_mod, "biased_topk_jit_kernel_impl", _fake_gate),
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.moe.fused_moe_triton_kernels": fake_kernels},
            ),
        ):
            return select_experts(
                hidden_states=torch.zeros(2, 8, dtype=torch.float32),
                router_logits=torch.zeros(2, NUM_EXPERTS, dtype=torch.float32),
                topk_config=config,
            ).topk_ids

    def test_routed_and_shared_slot_counts(self):
        topk_ids = self._route()
        self.assertEqual(topk_ids.shape[-1], TOP_K)
        for row in topk_ids.tolist():
            shared = [e for e in row if e >= NUM_EXPERTS]
            routed = [e for e in row if e < NUM_EXPERTS]
            self.assertEqual(len(shared), NUM_FUSED_SHARED, row)
            self.assertEqual(len(routed), NUM_ROUTED_TOPK, row)
            self.assertEqual(len(set(routed)), NUM_ROUTED_TOPK, row)


if __name__ == "__main__":
    unittest.main()
