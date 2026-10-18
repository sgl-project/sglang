"""Unit tests for routed_scaling_factor counting exactly once around a MoE block.

routed_scaling_factor is applied either in top-k, folded into the routed weights
(aiter, or a quant method that asks for it), or to the MoE output after the
experts. Every step that applies it or compensates for it must agree on which:

  * a fused shared expert's topk weight, on the per-rank-slot path and under
    Waterfill: 1.0 when the factor is folded in top-k, else 1/rsf so that the
    later multiply nets it out to 1.0x.
  * the two-batch-overlap MoE epilogue: scales the routed output only when top-k
    did not.
  * standard EP path (no per-rank slots): every rank computes the fused shared
    expert and the outputs are all-reduced, so the model-supplied 1/ep_size
    factor must be applied to the shared weight.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.operations import _StateDict
from sglang.srt.layers.moe import hash_topk as hash_topk_module
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK, TopKConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.model_runner_components import moe_ep_setup
from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.test.test_utils import CustomTestCase


class TestFusedSharedExpertScaling(CustomTestCase):
    # 256 physical routed experts over ep_size=8 -> 32 local routed per rank.
    NUM_PHYSICAL_ROUTED = 256
    EP_SIZE = 8
    EP_RANK = 0
    ROUTED_SCALING_FACTOR = 2.5
    # DeepSeek-V3-style grouped routing with a correction bias, the route that
    # aiter folds rsf into whatever the flag says.
    ROUTING = {"use_grouped_topk": True, "correction_bias": torch.zeros(256)}
    UNBIASED_ROUTING = {"use_grouped_topk": True, "correction_bias": None}

    def _run_remap(self, *, use_aiter, folded_in_topk=False):
        # Layout: [routed, routed, routed, shared]; the trailing column is the
        # fused shared expert. The shared weight starts at a sentinel to prove
        # the function overwrites it.
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        routed_weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, -123.0]], dtype=torch.float32)

        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=folded_in_topk,
            **self.ROUTING,
        )

        with (
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            _out_ids, out_weights = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )

        # Routed weights must never be touched by the shared-weight override.
        self.assertTrue(torch.equal(out_weights[0, :-1], routed_weights))
        return out_weights[0, -1].item()

    def test_aiter_path_uses_unit_shared_weight(self):
        # routed_scaling_factor is folded into the routed weights and the
        # post-MoE multiply is skipped -> shared weight must be 1.0, NOT 1/rsf.
        shared_weight = self._run_remap(use_aiter=True)
        self.assertAlmostEqual(shared_weight, 1.0)

    def test_post_moe_scaling_path_compensates_with_inverse_rsf(self):
        # Default path: the whole MoE output is multiplied by rsf afterward, so
        # the shared weight must be 1/rsf to net out to 1.0.
        shared_weight = self._run_remap(use_aiter=False)
        self.assertAlmostEqual(shared_weight, 1.0 / self.ROUTED_SCALING_FACTOR)

    def _waterfill_shared_weight(self, *, use_aiter, folded_in_topk):
        # Waterfill takes the shared column out of TopK and appends it itself.
        topk = TopK.__new__(TopK)
        torch.nn.Module.__init__(topk)
        topk.layer_id = 0
        topk.enable_waterfill = True
        topk.waterfill_balancer = None
        topk.topk_config = TopKConfig(
            top_k=8,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=folded_in_topk,
            **self.ROUTING,
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(n_routed_experts=self.NUM_PHYSICAL_ROUTED)
        )
        with (
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                moe_ep_setup,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=0)
                ),
            ),
            get_parallel().override(moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK),
        ):
            moe_ep_setup.prepare_moe_topk(
                model=torch.nn.ModuleList([topk]), model_config=model_config
            )
        return topk.waterfill_balancer.shared_weight

    def test_waterfill_shared_weight_matches_the_per_rank_slot(self):
        # Waterfill and the per-rank slot place the same fused shared expert, and
        # its weight is 1.0 only where the route really folded rsf: a folding
        # quant method's flag, or aiter's biased grouped router (not aiter as such).
        rsf = self.ROUTED_SCALING_FACTOR
        cases = (
            # (use_aiter, folded_in_topk, routing, shared weight for a net 1.0x)
            (False, False, self.ROUTING, 1.0 / rsf),
            (False, True, self.ROUTING, 1.0),
            (True, False, self.ROUTING, 1.0),
            (True, False, self.UNBIASED_ROUTING, 1.0 / rsf),
        )
        for use_aiter, folded_in_topk, routing, expected in cases:
            with (
                self.subTest(
                    use_aiter=use_aiter,
                    folded_in_topk=folded_in_topk,
                    bias=routing["correction_bias"] is not None,
                ),
                patch.object(self, "ROUTING", routing),
            ):
                waterfill = self._waterfill_shared_weight(
                    use_aiter=use_aiter, folded_in_topk=folded_in_topk
                )
                per_rank_slot = self._run_remap(
                    use_aiter=use_aiter, folded_in_topk=folded_in_topk
                )
                self.assertAlmostEqual(waterfill, expected)
                self.assertAlmostEqual(per_rank_slot, expected)

    def test_two_batch_overlap_epilogue_applies_factor_once(self):
        """The two-batch-overlap MoE epilogue must not scale the combined routed
        output again when top-k already folded routed_scaling_factor into it."""
        torch.manual_seed(0)
        routed = torch.randn(4, 6)
        shared = torch.randn(4, 6)
        for folded_in_topk in (False, True):
            for has_shared in (False, True):
                with self.subTest(folded_in_topk=folded_in_topk, has_shared=has_shared):
                    moe = SimpleNamespace(
                        routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
                        routed_scaling_factor_fused_in_topk=folded_in_topk,
                    )
                    state = _StateDict()
                    state.hidden_states_after_combine = routed.clone()
                    state.shared_output = shared.clone() if has_shared else None
                    with get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP):
                        DeepseekV2MoE.op_output(moe, state)

                    expected = routed * (
                        1.0 if folded_in_topk else self.ROUTED_SCALING_FACTOR
                    )
                    if has_shared:
                        expected = expected + shared
                    torch.testing.assert_close(state.hidden_states_mlp_output, expected)

    def _build_deepseek_moe(
        self,
        *,
        folded_in_topk,
        use_aiter,
        correction_bias=True,
        deepseek_v4=False,
        hash_layer=False,
    ):
        config = SimpleNamespace(
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            n_shared_experts=None,
            n_routed_experts=16,
            num_experts_per_tok=2,
            hidden_act="silu",
            hidden_size=8,
            moe_intermediate_size=8,
            norm_topk_prob=True,
            n_group=4,
            topk_group=2,
            scoring_func="sqrtsoftplus" if deepseek_v4 else "sigmoid",
            num_hash_layers=1 if hash_layer else 0,
            vocab_size=4,
        )
        gate = SimpleNamespace(
            e_score_correction_bias=torch.zeros(16) if correction_bias else None
        )
        experts = SimpleNamespace(
            should_fuse_routed_scaling_factor_in_topk=folded_in_topk
        )
        exec_config = SimpleNamespace(
            moe=SimpleNamespace(enable_waterfill=False, ep_num_redundant_experts=0)
        )
        module = deepseek_v2_module
        with (
            patch.object(module, "_use_aiter", use_aiter),
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1, moe_ep_size=1),
            ),
            patch.object(module, "get_exec", return_value=exec_config),
            patch.object(topk_module, "get_exec", return_value=exec_config),
            patch.object(hash_topk_module, "get_exec", return_value=exec_config),
            patch.object(
                module, "is_shared_experts_fusion_disabled", return_value=True
            ),
            patch.object(module, "MoEGate", return_value=gate),
            patch.object(
                module, "get_moe_impl_class", return_value=lambda **kwargs: experts
            ),
            patch.object(
                module,
                "SboFlags",
                SimpleNamespace(fuse_shared_experts_inside_sbo=lambda: False),
            ),
        ):
            return module.DeepseekV2MoE(
                config=config, layer_id=0, is_deepseek_v4=deepseek_v4
            )

    def test_deepseek_moe_takes_the_fold_from_the_router_it_builds(self):
        """Every step after top-k reads routed_scaling_factor_fused_in_topk, so it
        must follow the router the block built: the quant method's fold flag, or
        on aiter a route that folds (biased grouped routing, DeepSeek-V4's
        sqrtsoftplus gate), never grouped routing without a bias or hash routing."""
        cases = (
            # (folded_in_topk, use_aiter, route, routed weights carry rsf)
            (False, False, {}, False),
            (True, False, {}, True),
            (False, True, {}, True),
            (False, True, {"correction_bias": False}, False),
            (False, True, {"deepseek_v4": True}, True),
            (False, True, {"deepseek_v4": True, "hash_layer": True}, False),
            (True, False, {"deepseek_v4": True, "hash_layer": True}, True),
        )
        for folded_in_topk, use_aiter, route, expected in cases:
            with self.subTest(
                folded_in_topk=folded_in_topk, use_aiter=use_aiter, **route
            ):
                moe = self._build_deepseek_moe(
                    folded_in_topk=folded_in_topk, use_aiter=use_aiter, **route
                )
                self.assertEqual(moe.routed_scaling_factor_fused_in_topk, expected)

    def _run_post_process_standard_path(self, *, scaling_factor):
        topk_ids = torch.tensor([[5, 40, 100, 256]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            fused_shared_experts_scaling_factor=scaling_factor,
            allow_routed_experts_capture=False,
        )
        router_logits = torch.zeros((1, 256), dtype=torch.float32)
        with (
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", False),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
        ):
            _out_ids, out_weights, _recorder_ids = topk_module._post_process_topk_ids(
                topk_ids.clone(),
                topk_weights.clone(),
                topk_config,
                router_logits,
                layer_id=0,
            )
        self.assertTrue(torch.equal(out_weights[0, :-1], topk_weights[0, :-1]))
        return out_weights[0, -1].item()

    def test_standard_ep_path_applies_shared_scaling_factor(self):
        # Regression: models pass 1/ep_size under standard EP (every rank
        # computes the fused shared expert and outputs are all-reduced), but
        # the standard CUDA post-process dropped the factor, so the shared
        # contribution was summed ep_size times (corrupt EP4 output on
        # BailingMoeV3, BF16 and FP8 alike).
        shared_weight = self._run_post_process_standard_path(scaling_factor=0.25)
        self.assertAlmostEqual(shared_weight, 0.25)

    def test_standard_path_without_factor_keeps_shared_weight(self):
        # TP mode passes no factor; the shared weight must pass through
        # unscaled (guards the predicate against degrading to always-scale).
        shared_weight = self._run_post_process_standard_path(scaling_factor=None)
        self.assertAlmostEqual(shared_weight, 1.0)

    def test_shared_expert_ids_route_to_home_rank(self):
        # Sanity: the shared slot id is placed at this rank's interleaved
        # position (ep_rank * num_local_experts + num_local_routed).
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 0.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
        )
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            out_ids, _ = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )
        num_local_routed = self.NUM_PHYSICAL_ROUTED // self.EP_SIZE  # 32
        num_local_experts = num_local_routed + 1  # 33
        expected_shared_id = self.EP_RANK * num_local_experts + num_local_routed
        self.assertEqual(out_ids[0, -1].item(), expected_shared_id)

    def test_glm53_idle_rank_preserves_routed_topk_contract(self):
        """An idle rank must return the pre-append top-8 shape; returning top-9
        here would make the AITER postprocessor append shared ID 288 twice."""
        topk = TopK.__new__(TopK)
        torch.nn.Module.__init__(topk)
        topk.topk_config = TopKConfig(top_k=9, num_fused_shared_experts=1)
        topk.enable_waterfill = False
        topk.waterfill_balancer = None

        with (
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=None),
            ),
            patch.object(topk_module, "is_allocation_symmetric", return_value=False),
            patch.object(
                topk_module,
                "use_symmetric_memory",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
        ):
            output = topk.empty_topk_output(torch.device("cpu"))

        self.assertEqual(output.topk_ids.shape, (0, 8))
        self.assertEqual(output.topk_weights.shape, (0, 8))


if __name__ == "__main__":
    unittest.main()
