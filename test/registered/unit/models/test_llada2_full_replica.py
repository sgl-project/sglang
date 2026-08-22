# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import llada2  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLLaDA2FullReplica(CustomTestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            num_experts_per_tok=1,
            norm_topk_prob=True,
            hidden_size=4,
            num_shared_experts=1,
            hidden_act="silu",
            num_experts=2,
            moe_intermediate_size=8,
        )

    def _shared_expert_kwargs(self, fully_replicated):
        backend = SimpleNamespace(is_deepep=lambda: False)
        with (
            patch.object(
                llada2,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=2),
            ),
            patch.object(
                llada2,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=0)
                ),
            ),
            patch.object(
                llada2,
                "enable_moe_fully_dp",
                return_value=fully_replicated,
            ),
            patch.object(
                llada2,
                "get_moe_a2a_backend",
                return_value=backend,
            ),
            patch.object(
                llada2,
                "LLaDA2MoeGate",
                return_value=SimpleNamespace(expert_bias=None),
            ),
            patch.object(llada2, "TopK", return_value=object()),
            patch.object(
                llada2,
                "get_moe_impl_class",
                return_value=lambda **_kwargs: object(),
            ),
            patch.object(
                llada2,
                "LLaDA2MoeMLP",
                return_value=object(),
            ) as shared_mlp,
        ):
            llada2.LLaDA2MoeSparseMoeBlock(
                layer_id=0,
                config=self._config(),
            )

        return shared_mlp.call_args.kwargs

    def test_full_replica_shared_expert_is_tp1(self):
        replicated_kwargs = self._shared_expert_kwargs(fully_replicated=True)
        self.assertEqual(replicated_kwargs["tp_rank"], 0)
        self.assertEqual(replicated_kwargs["tp_size"], 1)

        standard_kwargs = self._shared_expert_kwargs(fully_replicated=False)
        self.assertNotIn("tp_rank", standard_kwargs)
        self.assertNotIn("tp_size", standard_kwargs)

    def test_full_replica_skips_global_post_expert_allreduce(self):
        hidden_states = torch.zeros((2, 3))

        for fully_replicated, expected_calls in ((True, 0), (False, 1)):
            with self.subTest(fully_replicated=fully_replicated):
                block = SimpleNamespace(
                    alt_stream=None,
                    num_shared_experts=1,
                    tp_size=2,
                    moe_fully_replicated=fully_replicated,
                    _forward_shared_experts=lambda tensor: torch.full_like(tensor, 2),
                    _forward_router_experts=lambda tensor: torch.full_like(tensor, 3),
                )
                with (
                    patch.object(
                        llada2,
                        "should_skip_post_experts_all_reduce",
                        return_value=False,
                    ),
                    patch.object(
                        llada2,
                        "tensor_model_parallel_all_reduce",
                        side_effect=lambda tensor: tensor,
                    ) as all_reduce,
                ):
                    output = llada2.LLaDA2MoeSparseMoeBlock.forward_normal(
                        block,
                        hidden_states,
                    )

                self.assertTrue(torch.equal(output, torch.full_like(output, 5)))
                self.assertEqual(all_reduce.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
