"""Check local EP routing against a dense reference using real NPU kernels.

One NPU runs each logical EP rank sequentially; summing its partial outputs
models the all-reduce performed by the model after the ``none`` dispatcher.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
    AscendTPCombineInput,
    AscendTPDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires Ascend NPU kernels")
class TestNpuNoneDispatcher(CustomTestCase):
    def _check_routing(self, ids, num_routed=4, ep_size=2, num_shared=0, int8=False):
        ids = torch.tensor(ids, dtype=torch.int32).reshape(-1, 2)
        num_tokens, top_k = ids.shape
        num_experts = num_routed + num_shared
        local_routed = num_routed // ep_size
        num_local = local_routed + num_shared
        # Binary fractions make BF16 matmul/combine exact; distinct expert
        # transforms detect a wrong rank offset even when count shapes match.
        x = ((torch.arange(num_tokens * 32) % 13 - 6) / 8).reshape(num_tokens, 32)
        weights = torch.stack(
            [torch.eye(32).roll(expert % 32, dims=1) for expert in range(num_experts)]
        )
        scores = torch.tensor([0.25, 0.75]).repeat(num_tokens, 1)
        expected = torch.zeros_like(x)
        for row in range(num_tokens):
            for slot in range(top_k):
                expert = int(ids[row, slot])
                if expert >= 0:
                    expected[row] += scores[row, slot] * (x[row] @ weights[expert])

        total = torch.zeros_like(x)
        for rank in range(ep_size):
            with self.subTest(rank=rank):
                config = MoeRunnerConfig(
                    num_experts=num_experts,
                    num_local_experts=num_local,
                    num_fused_shared_experts=num_shared,
                    top_k=top_k,
                )
                with patch(
                    "sglang.srt.layers.moe.token_dispatcher.ascend_tp.get_parallel",
                    return_value=SimpleNamespace(moe_ep_size=ep_size, moe_ep_rank=rank),
                ):
                    dispatcher = AscendTPDispatcher(config)
                if int8:
                    dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
                local_ids = list(range(rank * local_routed, (rank + 1) * local_routed))
                local_ids += list(range(num_routed, num_experts))
                local_weights = weights[local_ids].clone()
                # Shared experts are replicated across EP ranks. Their caller
                # supplies the 1/EP scaling; the dispatcher must keep them local.
                if num_shared:
                    local_weights[-num_shared:] /= ep_size
                local_weights = local_weights.to(device="npu", dtype=torch.bfloat16)
                topk = StandardTopKOutput(
                    topk_weights=scores.to(device="npu", dtype=torch.bfloat16),
                    topk_ids=ids.npu(),
                    router_logits=None,
                )
                hidden = x.to(device="npu", dtype=torch.bfloat16)
                # Repeat to cover cached expert mappings and cleared combine state.
                for _ in range(2):
                    dispatched = dispatcher.dispatch(hidden, topk)
                    counts = dispatched.expert_tokens.cpu()
                    self.assertEqual(counts.numel(), local_weights.shape[0])
                    torch.testing.assert_close(
                        counts,
                        torch.tensor([(ids == expert).sum() for expert in local_ids]),
                    )
                    permuted = dispatched.hidden_states
                    if int8:
                        self.assertEqual(permuted.dtype, torch.int8)
                        permuted = (
                            permuted.float()
                            * dispatched.hidden_states_scale.float().reshape(-1, 1)
                        ).to(torch.bfloat16)
                    expert_output = torch.ops.npu.npu_grouped_matmul(
                        [permuted],
                        [local_weights],
                        group_list=dispatched.expert_tokens,
                        group_type=0,
                        split_item=3,
                        group_list_type=dispatched.group_list_type,
                    )[0]
                    # Unused GMM rows are undefined. Poison them so a finalizer
                    # that accidentally reads dropped routes cannot pass by luck.
                    expert_output[int(counts.sum()) :].fill_(float("nan"))
                    actual = dispatcher.combine(AscendTPCombineInput(expert_output))
                    partial = torch.zeros_like(x)
                    for row in range(num_tokens):
                        for slot in range(top_k):
                            expert = int(ids[row, slot])
                            if expert in local_ids:
                                scale = 1 / ep_size if expert >= num_routed else 1
                                partial[row] += (
                                    scores[row, slot]
                                    * scale
                                    * (x[row] @ weights[expert])
                                )
                    torch.testing.assert_close(
                        actual.float().cpu(), partial, rtol=0, atol=0.008 if int8 else 0
                    )
                total += actual.float().cpu()
        torch.testing.assert_close(total, expected, rtol=0, atol=0.008 if int8 else 0)

    def test_ep1(self):
        self._check_routing([[0, 2], [1, 3]], ep_size=1)

    def test_ep2(self):
        self._check_routing([[0, 2], [1, 3], [2, 3], [0, 1]])

    def test_ep2_qwen36_expert_count(self):
        self._check_routing(
            [[0, 127], [128, 255], [255, 1], [129, 126]], num_routed=256
        )

    def test_ep2_no_local_tokens(self):
        self._check_routing([[2, 3]])

    def test_ep2_shared_experts_and_padding(self):
        self._check_routing([[0, 4], [2, 4], [-1, 4]], num_shared=1)

    def test_ep2_int8_routing(self):
        self._check_routing([[0, 2], [1, 3], [2, 3], [0, 1]], int8=True)

    def test_ep2_empty_batch(self):
        self._check_routing([])


if __name__ == "__main__":
    unittest.main()
