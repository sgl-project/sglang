"""Routed and shared contributions must each receive their scale exactly once."""

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.runtime_context import get_model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.operations import _StateDict
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopK, TopKConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.model_executor.model_runner_components import moe_ep_setup
from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
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
        # The remap appends the fused shared column after the routed ones.
        topk_ids = torch.tensor([[5, 40, 100]], dtype=torch.int32)
        routed_weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
        topk_weights = routed_weights.unsqueeze(0)

        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
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
        # Both append paths must compensate for the same routing scale.
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
        """The overlap epilogue must apply the routed scale exactly once."""
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
        fused_shared=0,
        ep_size=1,
        tp_size=1,
        separate_shared=False,
        experts=None,
        alt_stream=None,
    ):
        config = SimpleNamespace(
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            n_shared_experts=fused_shared or int(separate_shared) or None,
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
            e_score_correction_bias=torch.zeros(16) if correction_bias else None,
            e_score_correction_bias_vl=None,
        )
        if experts is None:
            experts = SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=folded_in_topk,
                runner_backend=None,
                routing_method_type=None,
            )
        exec_config = SimpleNamespace(
            moe=SimpleNamespace(enable_waterfill=False, ep_num_redundant_experts=0)
        )
        module = deepseek_v2_module
        with (
            get_context().override_server_args(),
            patch.object(module, "_use_aiter", use_aiter),
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=tp_size, moe_ep_size=ep_size),
            ),
            patch.object(module, "get_exec", return_value=exec_config),
            patch.object(topk_module, "get_exec", return_value=exec_config),
            get_model().override(
                shared_experts=SharedExpertsConfig(
                    num_fused_shared_experts=fused_shared
                )
            ),
            patch.object(module, "MoEGate", return_value=gate),
            patch.object(
                module,
                "DeepseekV2MLP",
                return_value=SimpleNamespace(
                    gate_up_proj=SimpleNamespace(
                        output_size_per_partition=16,
                        quant_method=None,
                        weight=torch.empty(16, 8),
                    )
                ),
            ),
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
                config=config,
                layer_id=0,
                is_deepseek_v4=deepseek_v4,
                alt_stream=alt_stream,
            )

    def test_deepseek_moe_takes_the_fold_from_the_router_it_builds(self):
        """The MoE epilogue must follow the constructed router's folding policy."""
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

    def test_capture_admits_ascend_tp_and_flashinfer_megamoe(self):
        """Both constructed paths reach ordinary capture despite their dispatchers."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        x = torch.ones(2, 8)
        for backend, a2a, ep_size in (
            (MoeRunnerBackend.ASCEND, MoeA2ABackend.NONE, 1),
            (
                MoeRunnerBackend.FLASHINFER_MEGAMOE,
                MoeA2ABackend.FLASHINFER_MEGAMOE,
                2,
            ),
        ):
            with (
                self.subTest(backend=backend),
                get_flags().moe.override(runner_backend=backend, a2a_backend=a2a),
            ):
                experts = FusedMoE.__new__(FusedMoE)
                torch.nn.Module.__init__(experts)
                experts.runner_backend = backend
                experts.quant_method = SimpleNamespace(
                    preserves_input=lambda layer: True
                )
                experts.moe_runner_config = MoeRunnerConfig()
                experts.should_fuse_routed_scaling_factor_in_topk = True
                experts.routing_method_type = None
                moe = self._build_deepseek_moe(
                    folded_in_topk=True,
                    use_aiter=False,
                    tp_size=2,
                    ep_size=ep_size,
                    separate_shared=True,
                    experts=experts,
                    alt_stream=object(),
                )
                self.assertTrue(experts.moe_runner_config.inplace)
                self.assertFalse(moe._enable_a2a_moe)
                with (
                    model_capture_mode(),
                    patch.object(
                        deepseek_v2_module,
                        "is_in_breakable_cuda_graph",
                        return_value=False,
                    ),
                    patch.object(moe, "forward_normal", return_value=x) as serial,
                    patch.object(
                        moe, "forward_normal_dual_stream", return_value=x
                    ) as dual,
                ):
                    self.assertIs(moe(x), x)
                dual.assert_called_once()
                serial.assert_not_called()

    def test_dynamic_bias_preserves_the_final_routed_scale(self):
        """Per-token bias must preserve the final scale of the static AITER route."""
        hidden = torch.ones(2, 4)
        logits = torch.zeros(2, 16)
        rsf = self.ROUTED_SCALING_FACTOR

        def aiter_route(
            logits, bias, weights, ids, groups, selected, renormalize, scale
        ):
            weights.fill_((1.0 / weights.shape[1] if renormalize else 0.5) * scale)
            ids.copy_(torch.arange(ids.shape[1]).expand_as(ids))

        with (
            get_context().override_server_args(),
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", False),
            patch.object(
                topk_module, "biased_grouped_topk", topk_module.biased_grouped_topk_gpu
            ),
            patch.object(
                topk_module, "aiter_biased_grouped_topk", aiter_route, create=True
            ),
        ):
            for renormalize in (False, True):
                config = TopKConfig(
                    top_k=4,
                    use_grouped_topk=True,
                    num_expert_group=4,
                    topk_group=2,
                    renormalize=renormalize,
                    correction_bias=torch.zeros(16),
                    routed_scaling_factor=rsf,
                    scoring_func="sigmoid",
                    allow_routed_experts_capture=False,
                )
                for dynamic in (False, True):
                    with self.subTest(renormalize=renormalize, dynamic=dynamic):
                        output = topk_module.select_experts(
                            hidden,
                            logits,
                            config,
                            dynamic_expert_bias=torch.zeros_like(logits)
                            if dynamic
                            else None,
                        )
                        routed = output.topk_weights.sum(dim=-1, keepdim=True)
                        final = deepseek_v2_module._scale_routed_and_add_shared(
                            routed,
                            None,
                            routed_scaling_factor=rsf,
                            routed_scale_applied=topk_module.is_routed_scaling_factor_fused_in_topk(
                                config
                            ),
                        )
                        expected = torch.full_like(
                            final, rsf if renormalize else 2.0 * rsf
                        )
                        torch.testing.assert_close(final, expected)

    def test_per_token_bias_keeps_the_shared_coefficient_independent(self):
        """Dynamic routing must not scale shared output by the selected-score sum."""
        logits = torch.zeros(2, 16)
        rsf = self.ROUTED_SCALING_FACTOR
        for renormalize in (False, True):
            for folded in (False, True):
                with self.subTest(renormalize=renormalize, folded=folded):
                    weights, _ = topk_module.biased_grouped_topk_impl(
                        logits,
                        logits,
                        torch.zeros_like(logits),
                        topk=5,
                        renormalize=renormalize,
                        num_expert_group=4,
                        topk_group=2,
                        num_fused_shared_experts=1,
                        routed_scaling_factor=rsf,
                        apply_routed_scaling_factor_on_output=folded,
                    )
                    shared = weights[:, -1] * (1.0 if folded else rsf)
                    torch.testing.assert_close(shared, torch.ones_like(shared))

    def test_unbiased_aiter_global_shared_output_is_scaled_once(self):
        """Restoring the routed scale must retain one shared contribution, including EP."""
        from sglang.kernels.ops.moe import fused_moe_triton_kernels

        def append(ids, weights, count, scale, base, **kwargs):
            return (
                torch.cat((ids, ids.new_full((ids.shape[0], count), base)), dim=-1),
                torch.cat(
                    (weights, weights.new_full((weights.shape[0], count), scale)),
                    dim=-1,
                ),
            )

        with (
            get_context().override_server_args(),
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", False),
            patch.object(topk_module, "grouped_topk", topk_module.grouped_topk_gpu),
            patch.object(
                fused_moe_triton_kernels, "fused_append_shared_experts", append
            ),
        ):
            for ep_size in (1, 4):
                with self.subTest(ep_size=ep_size):
                    moe = self._build_deepseek_moe(
                        folded_in_topk=False,
                        use_aiter=True,
                        correction_bias=False,
                        fused_shared=1,
                        ep_size=ep_size,
                    )
                    moe.topk.topk_config.allow_routed_experts_capture = False
                    output = topk_module.select_experts(
                        torch.ones(1, 8),
                        torch.zeros(1, 16),
                        moe.topk.topk_config,
                    )
                    # Orthogonal expert outputs expose routed and shared coefficients.
                    combined = torch.stack(
                        (
                            output.topk_weights[:, :-1].sum(dim=-1),
                            output.topk_weights[:, -1],
                        ),
                        dim=-1,
                    )
                    final = deepseek_v2_module._scale_routed_and_add_shared(
                        combined,
                        None,
                        routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
                        routed_scale_applied=moe.routed_scaling_factor_fused_in_topk,
                    )
                    torch.testing.assert_close(
                        final,
                        torch.tensor([[self.ROUTED_SCALING_FACTOR, 1.0 / ep_size]]),
                    )

    def _run_post_process_standard_path(self, *, scaling_factor):
        topk_ids = torch.tensor([[5, 40, 100, 256]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
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
        topk_ids = torch.tensor([[5, 40, 100]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
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

    def _route_with_per_rank_slots(self, *, num_fused_shared_experts, logits, bias):
        topk_config = TopKConfig(
            top_k=8 + num_fused_shared_experts,
            shared_experts=SharedExpertsConfig(
                num_fused_shared_experts=num_fused_shared_experts
            ),
            renormalize=True,
            correction_bias=bias,
            torch_native=True,
            scoring_func="sigmoid",
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            allow_routed_experts_capture=False,
        )
        with (
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module,
                "has_per_rank_fused_shared_slots",
                side_effect=lambda num_fused: num_fused > 0,
            ),
            get_parallel().override(moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK),
        ):
            out = topk_module.select_experts(
                torch.empty(logits.shape[0], 1), logits, topk_config
            )
        return out.topk_ids, out.topk_weights

    def test_per_rank_slot_routes_like_no_fusion(self):
        """With per-rank shared slots the router must pick and renormalize the
        routed experts alone. Asked for top_k with the shared slot counted, it
        renormalized over one routed expert too many, which the shared column
        then overwrote, so the kept routed weights summed to about 0.9."""
        torch.manual_seed(0)
        logits = torch.randn(16, self.NUM_PHYSICAL_ROUTED)
        bias = 0.1 * torch.randn(self.NUM_PHYSICAL_ROUTED)
        ref_ids, ref_weights = self._route_with_per_rank_slots(
            num_fused_shared_experts=0, logits=logits, bias=bias
        )
        ids, weights = self._route_with_per_rank_slots(
            num_fused_shared_experts=1, logits=logits, bias=bias
        )

        self.assertEqual(ids.shape, (16, 9))
        torch.testing.assert_close(weights[:, :8].sum(-1), torch.ones(16))
        # Rank r's routed block starts at r * (num_local_routed + 1).
        num_local_routed = self.NUM_PHYSICAL_ROUTED // self.EP_SIZE
        expected_ids = ref_ids + ref_ids // num_local_routed
        order, ref_order = ids[:, :8].argsort(-1), expected_ids.argsort(-1)
        self.assertTrue(
            torch.equal(
                ids[:, :8].gather(-1, order), expected_ids.gather(-1, ref_order)
            )
        )
        torch.testing.assert_close(
            weights[:, :8].gather(-1, order), ref_weights.gather(-1, ref_order)
        )

    def test_glm53_idle_rank_preserves_routed_topk_contract(self):
        """An idle rank must return the pre-append top-8 shape; returning top-9
        here would make the AITER postprocessor append shared ID 288 twice."""
        with (
            get_context().override_server_args(enable_waterfill=False),
            get_model().override(
                shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
            ),
        ):
            topk = TopK(top_k=8, layer_id=0)

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


class TestDeepseekMoeEpilogue(CustomTestCase):
    def _moe(
        self,
        *,
        replicated=False,
        reduce=False,
        inplace=False,
        mutates_input=False,
        backend=MoeRunnerBackend.TRITON,
    ):
        def run_experts(hidden_states, *args, **kwargs):
            return hidden_states.mul_(2.5) if mutates_input else hidden_states * 2.5

        experts = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(experts)
        experts.forward = run_experts
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        experts.moe_runner_config = MoeRunnerConfig(inplace=inplace)
        experts.quant_method = UnquantizedFusedMoEMethod()
        if backend is not None:
            experts.quant_method.runner = MoeRunner(backend, experts.moe_runner_config)
        experts.runner_backend = backend
        experts.should_fuse_routed_scaling_factor_in_topk = False

        def gate(hidden_states, *args, **kwargs):
            return hidden_states

        gate.e_score_correction_bias_vl = None
        moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        vars(moe).update(
            experts=experts,
            gate=gate,
            topk=lambda *args, **kwargs: SimpleNamespace(
                format=topk_module.TopKOutputFormat.STANDARD
            ),
            num_fused_shared_experts=0,
            routed_scaling_factor=2.5,
            routed_scaling_factor_fused_in_topk=True,
            _shared_expert_tp1=replicated,
            _fuse_shared_experts_inside_sbo=False,
            _enable_a2a_moe=False,
            reduce_results=reduce,
            is_nextn=False,
            is_hash=False,
            layer_id=0,
            alt_stream=SimpleNamespace(wait_stream=lambda stream: None),
            _maybe_quant_moe_input_once=lambda x: None,
            _should_quant_routed_input_mxfp8=lambda x: False,
            _forward_shared_experts=lambda x, *args, **kwargs: x * 3,
            _can_dual_stream_graph=lambda x: False,
        )
        return moe

    def setUp(self):
        super().setUp()

        from contextlib import ExitStack

        from sglang.srt.layers.moe import utils

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("_is_cuda", True), ("_is_hip", False)):
            self.stack.enter_context(patch.object(deepseek_v2_module, name, value))
        self.stack.enter_context(
            patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(wait_stream=lambda s: None),
            )
        )
        self.stack.enter_context(
            patch("torch.cuda.stream", side_effect=lambda s: nullcontext())
        )
        self.stack.enter_context(
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE)
        )
        self.stack.enter_context(
            get_parallel().override(
                tp_size=2, tp_rank=0, moe_ep_size=1, moe_tp_size=2, dwdp_size=1
            )
        )
        self.stack.enter_context(
            patch.object(
                utils,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.stack.enter_context(
            patch(
                "sglang.srt.layers.moe.mega_moe.should_use_mega_moe", return_value=False
            )
        )

    def test_skip_shared_experts_survives_graph_and_capture_dispatch(self):
        """A shared result computed before gather must not be added again by capture."""
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        for graph in (False, True):
            with self.subTest(graph=graph):
                moe = self._moe()
                moe._can_dual_stream_graph = lambda x: graph
                with (
                    patch.object(
                        deepseek_v2_module, "get_is_capture_mode", return_value=True
                    ),
                    patch.object(
                        deepseek_v2_module,
                        "is_in_breakable_cuda_graph",
                        return_value=False,
                    ),
                    patch.object(
                        deepseek_v2_module,
                        "dsv2_flashinfer_moe_dual_stream_graph",
                        side_effect=lambda x, *args: moe.forward_normal_dual_stream(x),
                    ),
                ):
                    actual = moe(x, skip_shared_experts=True)
                torch.testing.assert_close(actual, x * 2.5)

    def test_replicated_shared_is_counted_once_with_owned_or_deferred_sum(self):
        """Summing two TP partials must count the replicated shared output once."""
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        for dual in (False, True):
            for reduce in (False, True):
                with self.subTest(dual=dual, reduce=reduce):
                    outputs = []
                    for rank in (0, 1):
                        moe = self._moe(replicated=True, reduce=reduce)
                        run = (
                            moe.forward_normal_dual_stream
                            if dual
                            else moe.forward_normal
                        )
                        with (
                            get_parallel().override(tp_rank=rank),
                            patch.object(
                                deepseek_v2_module,
                                "post_experts_all_reduce",
                                side_effect=lambda t: t * 2,
                            ),
                        ):
                            outputs.append(run(x.clone()))
                    expected = x * (2 * 2.5 + 3)
                    if reduce:
                        for output in outputs:
                            torch.testing.assert_close(output, expected)
                    else:
                        torch.testing.assert_close(sum(outputs), expected)

    def test_raw_pieces_bypass_tensor_only_graph_op(self):
        """A request for deferred pieces must not reduce or lose its shared operand."""
        x = torch.ones(2, 3)
        moe = self._moe(replicated=True)
        moe._can_dual_stream_graph = lambda x: True
        with (
            patch.object(deepseek_v2_module, "get_is_capture_mode", return_value=False),
            patch.object(
                deepseek_v2_module,
                "dsv2_flashinfer_moe_dual_stream_graph",
                side_effect=lambda x, *args: moe.forward_normal_dual_stream(x),
            ),
        ):
            pieces = moe(x, return_moe_output=True)
        self.assertIsInstance(pieces, deepseek_v2_module.MoEOutput)
        torch.testing.assert_close(pieces.get_merged(), x * 2.5)
        torch.testing.assert_close(pieces.shared, x * 3)

    def test_inplace_runner_reads_shared_before_overwriting_input(self):
        """Capture must not let an in-place routed write change the shared input."""
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        moe = self._moe(inplace=True, mutates_input=True)
        actual = moe.forward_normal_dual_stream(x.clone())
        torch.testing.assert_close(actual, x * 5.5)

    def test_capture_preserves_overlap_with_inplace_suggestion(self):
        """Input-preserving runners must retain routed-first capture scheduling."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )
        from sglang.srt.utils import dispose_tensor

        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        from sglang.srt.layers.moe.moe_runner import marlin  # noqa: F401

        for backend in (
            MoeRunnerBackend.DEEP_GEMM,
            MoeRunnerBackend.MARLIN,
            MoeRunnerBackend.TRITON_KERNELS,
        ):
            with self.subTest(backend=backend):
                moe = self._moe(inplace=True, backend=backend)
                calls = []

                def routed(hidden, *args, **kwargs):
                    calls.append("routed")
                    output = hidden * 2.5
                    if backend.is_deep_gemm():
                        dispose_tensor(hidden)
                    return output

                def shared(hidden, *args, **kwargs):
                    calls.append("shared")
                    return hidden * 3

                moe.experts.forward = routed
                moe._forward_shared_experts = shared
                with (
                    model_capture_mode(),
                    get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON),
                    patch.object(
                        deepseek_v2_module,
                        "is_in_breakable_cuda_graph",
                        return_value=False,
                    ),
                ):
                    actual = moe(x.clone())
                self.assertEqual(calls, ["routed", "shared"])
                torch.testing.assert_close(actual, x * 5.5)

    def test_experimental_aliases_preserve_base_capture_overlap(self):
        """Experimental aliases must retain routed-first capture without adapters."""
        from sglang.srt.layers.moe.moe_runner import (  # noqa: F401
            flashinfer_trtllm,
            marlin,
        )
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.marlin_lora_temp import (  # noqa: F401
            sgl_backend as marlin_alias,
        )
        from sglang.srt.lora.trtllm_lora_temp import (  # noqa: F401
            sgl_backend as trtllm_alias,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        for alias, stock in (
            (MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN, MoeRunnerBackend.MARLIN),
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
            ),
        ):
            for wrapped in (False, True):
                with self.subTest(alias=alias, wrapped=wrapped):
                    moe = self._moe(inplace=True, backend=alias)
                    base = moe.experts
                    stock_runner = MoeRunner(stock, base.moe_runner_config)
                    self.assertTrue(stock_runner.preserves_input)
                    calls = []

                    def routed(hidden, *args, **kwargs):
                        calls.append("routed")
                        return hidden * 2.5

                    def shared(hidden, *args, **kwargs):
                        calls.append("shared")
                        return hidden * 3

                    base.forward = routed
                    moe._forward_shared_experts = shared
                    if wrapped:
                        wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
                        torch.nn.Module.__init__(wrapper)
                        wrapper.base_layer = base
                        wrapper.quant_method = base.quant_method
                        wrapper.moe_runner_config = base.moe_runner_config
                        wrapper.lora_backend = SimpleNamespace(
                            get_batch_info=lambda *args: None
                        )
                        moe.experts = wrapper
                    with model_capture_mode():
                        actual = moe(x.clone())
                    self.assertEqual(calls, ["routed", "shared"])
                    self.assertEqual(base.preserves_input, stock_runner.preserves_input)
                    torch.testing.assert_close(actual, x * 5.5)

    def test_awq_amx_preserves_input_without_a_runner(self):
        """The direct AMX path must answer the layer query without a runner."""
        from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMoEMethod
        from sglang.srt.layers.quantization.awq.schemes.awq_cpu import (
            AWQIntelAMXMoEScheme,
        )

        config = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                experts = self._moe(inplace=inplace, backend=None).experts
                del experts.quant_method
                experts.quant_method = AWQMoEMethod(config)
                experts.scheme = AWQIntelAMXMoEScheme(config)
                experts.quant_method.create_moe_runner(
                    experts, experts.moe_runner_config
                )
                self.assertTrue(experts.preserves_input)

    def test_direct_quant_methods_preserve_capture_overlap(self):
        """Direct kernels need their own input contract even without a named runner."""
        from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesMoEMethod
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8_moe import (
            CompressedTensorsW4AFP8MoE,
        )
        from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFMoEMethod
        from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        config = SimpleNamespace(
            target_scheme_map={
                "Linear": {
                    "weights": SimpleNamespace(
                        num_bits=4, group_size=128, symmetric=True
                    )
                }
            },
            quant_format="pack-quantized",
        )
        cases = (
            (W4AFp8MoEMethod(W4AFp8Config()), None, True),
            (GGUFMoEMethod(GGUFConfig()), None, True),
            (
                CompressedTensorsFusedMoEMethod(config),
                CompressedTensorsW4AFP8MoE(config, None, None),
                True,
            ),
            # This direct method calls Triton with the inplace suggestion. Its
            # weight loader needs bitsandbytes; runner selection does not.
            (BitsAndBytesMoEMethod.__new__(BitsAndBytesMoEMethod), None, False),
        )
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        for method, scheme, preserves in cases:
            with self.subTest(method=type(method).__name__):
                moe = self._moe(inplace=True, backend=None)
                experts = moe.experts
                del experts.quant_method
                experts.quant_method = method
                experts.scheme = scheme
                experts.moe_runner_config = MoeRunnerConfig()
                method.create_moe_runner(experts, experts.moe_runner_config)
                experts.runner_backend = method.get_moe_runner_backend(experts)
                self.assertIsNone(experts.runner_backend)
                self.assertTrue(experts.moe_runner_config.inplace)
                calls = []

                def routed(hidden, *args, **kwargs):
                    calls.append("routed")
                    return hidden * 2.5 if preserves else hidden.mul_(2.5)

                def shared(hidden, *args, **kwargs):
                    calls.append("shared")
                    return hidden * 3

                experts.forward = routed
                moe._forward_shared_experts = shared
                with model_capture_mode():
                    actual = moe(x.clone())
                self.assertEqual(
                    calls, ["routed", "shared"] if preserves else ["shared", "routed"]
                )
                torch.testing.assert_close(actual, x * 5.5)

    def test_paged_experts_keep_capture_overlap(self):
        """Paged experts must overlap shared work using their effective runner config."""
        from sglang.srt.layers.moe.paged_experts import PagedExpertsMoEMethod
        from sglang.srt.layers.moe.paged_experts.formats import format_for
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        for fp8 in (False, True):
            for fused in (False, True):
                with (
                    self.subTest(fp8=fp8, fused=fused),
                    get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON),
                ):
                    base_method = (
                        Fp8MoEMethod(
                            Fp8Config(
                                is_checkpoint_fp8_serialized=True,
                                weight_block_size=[128, 128],
                            )
                        )
                        if fp8
                        else UnquantizedFusedMoEMethod()
                    )
                    method = PagedExpertsMoEMethod(
                        base_method=base_method,
                        expert_format=format_for(base_method),
                        num_experts=8,
                        num_resident=4,
                        graphs=True,
                    )
                    moe = self._moe(inplace=True)
                    experts = moe.experts
                    experts.quant_method = method
                    experts.moe_runner_config = MoeRunnerConfig(num_experts=8)
                    method.create_moe_runner(experts, experts.moe_runner_config)
                    experts.runner_backend = method.get_moe_runner_backend(experts)
                    if not fused:
                        method.runner.fused_func = None
                    self.assertTrue(experts.moe_runner_config.inplace)
                    self.assertFalse(method.runner.config.inplace)
                    calls = []

                    def routed(hidden, *args, **kwargs):
                        calls.append("routed")
                        return hidden * 2.5

                    def shared(hidden, *args, **kwargs):
                        calls.append("shared")
                        return hidden * 3

                    experts.forward = routed
                    moe._forward_shared_experts = shared
                    with model_capture_mode():
                        actual = moe(x.clone())
                    self.assertEqual(calls, ["routed", "shared"])
                    torch.testing.assert_close(actual, x * 5.5)

    def test_lora_wrapped_experts_preserve_capture_dispatch(self):
        """Wrapping routed experts must retain capture overlap with shared experts."""
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        moe = self._moe(inplace=True)
        base = moe.experts
        base.moe_runner_config.gemm1_alpha = None
        base.moe_tp_size = 2
        base.moe_tp_rank = 0
        base.intermediate_size_per_partition = 3
        base.num_experts = base.num_local_experts = 8
        base.runner = None
        base.quant_method.get_triton_quant_info = lambda layer: None
        base.dispatcher = SimpleNamespace(
            dispatch=lambda **kwargs: SimpleNamespace(**kwargs),
            combine=lambda combine_input: combine_input.hidden_states,
        )
        calls = []

        def routed(hidden, *args, **kwargs):
            calls.append("routed")
            return hidden * 2.5

        def lora_routed(dispatched, quant_info, *, lora_info):
            self.assertTrue(lora_info.has_active_lora)
            calls.append("lora")
            return SimpleNamespace(hidden_states=dispatched.hidden_states * 2.5)

        def shared(hidden, *args, **kwargs):
            calls.append("shared")
            return hidden * 3

        base.forward = routed
        moe._forward_shared_experts = shared
        batch = SimpleNamespace(
            lora_ranks=torch.tensor([1]),
            has_active_lora=True,
            moe_lora_info=SimpleNamespace(
                seg_indptr=torch.tensor([0, 2]),
                req_to_lora=torch.tensor([0]),
                adapter_enabled=torch.tensor([True]),
                token_lora_mapping=torch.tensor([0, 0]),
            ),
        )
        lora_backend = SimpleNamespace(
            name="triton",
            get_batch_info=lambda *args: batch,
            prefill_cuda_graph_batch_info=None,
            prefill_moe_cg_buffers=None,
            moe_cg_buffers=None,
        )
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
            moe.experts = FusedMoEWithLoRA(base, lora_backend)
        self.stack.enter_context(
            patch.object(moe.experts._lora_runner, "run", side_effect=lora_routed)
        )
        weights = torch.zeros(1, 8, 1, 3)
        moe.experts.set_lora_info(weights, weights, weights, weights)

        for has_batch in (True, False):
            lora_backend.get_batch_info = lambda *args: batch if has_batch else None
            calls.clear()
            with self.subTest(has_batch=has_batch), model_capture_mode():
                actual = moe(x.clone())
                self.assertEqual(calls, ["lora" if has_batch else "routed", "shared"])
                torch.testing.assert_close(actual, x * 5.5)

    def test_runner_override_does_not_inherit_builtin_input_guarantee(self):
        """An overriding core may overwrite input even when its builtin allocates output."""
        from sglang.srt.layers.moe.moe_runner import runner as runner_module
        from sglang.srt.layers.moe.moe_runner.base import DispatchMoeRunnerCore
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        class WritingCore(DispatchMoeRunnerCore):
            runner_backend = MoeRunnerBackend.MARLIN

            def run_from_dispatch(
                self, dispatch_output, quant_info, runner_config, hooks=None
            ):
                return StandardCombineInput(dispatch_output.hidden_states.mul_(2.5))

        moe = self._moe()
        with patch.dict(
            runner_module._CUSTOM_RUNNER_CORE_FACTORIES,
            {MoeRunnerBackend.MARLIN.value: WritingCore},
        ):
            runner = runner_module.MoeRunner(
                MoeRunnerBackend.MARLIN,
                moe.experts.moe_runner_config,
                lora_enabled=True,
            )
        moe.experts.quant_method.runner = runner
        moe.experts.runner_backend = runner.runner_backend
        moe.experts.forward = lambda x, *a, **kw: (
            runner.run(StandardDispatchOutput(x, None, None), None).hidden_states
        )
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        with model_capture_mode():
            actual = moe(x.clone())
        torch.testing.assert_close(actual, x * 5.5)

    def test_deepgemm_shared_read_precedes_eager_disposal(self):
        """Capture preservation must not enable overlap once disposal is active."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )
        from sglang.srt.utils import dispose_tensor

        x = torch.ones(2, 3)
        moe = self._moe(inplace=True, backend=MoeRunnerBackend.DEEP_GEMM)

        def routed(hidden, *args, **kwargs):
            output = hidden * 2.5
            dispose_tensor(hidden)
            return output

        moe.experts.forward = routed
        with model_capture_mode():
            torch.testing.assert_close(moe(x.clone()), x * 5.5)
        with get_flags().capture.override(disable_dispose_tensor=False):
            torch.testing.assert_close(
                moe.forward_normal_dual_stream(x.clone()), x * 5.5
            )

    def test_deepep_skip_leaves_complete_routed_output_unreduced(self):
        """An A2A combine already returns complete rows; skipping shared keeps them."""
        x = torch.ones(2, 3)
        moe = self._moe(reduce=True)
        moe._enable_a2a_moe = True
        moe.alt_stream = None
        with patch.object(
            deepseek_v2_module, "post_experts_all_reduce", side_effect=lambda t: t * 2
        ):
            actual = moe(x, forward_batch=SimpleNamespace(), skip_shared_experts=True)
        torch.testing.assert_close(actual, x * 2.5)

    def test_amx_skip_does_not_enter_fused_shared_kernel(self):
        """The AMX shared-add kernel cannot honor a request for routed output alone."""
        x = torch.ones(2, 3)
        moe = self._moe()
        moe.shared_experts = SimpleNamespace(gate_up_proj=object())
        # The fused operator's numerical contract; native AMX needs its own hardware.
        moe.forward_cpu = lambda hidden: hidden * 5.5
        with patch.object(
            deepseek_v2_module, "use_intel_amx_backend", return_value=True
        ):
            actual = moe.forward_normal(x, skip_shared_experts=True)
        torch.testing.assert_close(actual, x * 2.5)

    def test_mega_combine_keeps_scale_and_honors_skip(self):
        """A complete MegaMoE output must neither be scaled nor summed a second time."""
        from sglang.srt.layers.moe import mega_moe, mega_moe_flydsl

        x = torch.ones(2, 3)
        for flydsl in (False, True):
            for skip in (False, True):
                with self.subTest(flydsl=flydsl, skip=skip):
                    moe = self._moe(reduce=True)
                    moe.alt_stream = None
                    moe.mega_shared_l1_weights = None
                    backend = mega_moe_flydsl if flydsl else mega_moe
                    with (
                        patch.object(
                            mega_moe, "should_use_mega_moe", return_value=True
                        ),
                        patch.object(
                            mega_moe, "_use_amd_flydsl_mega_moe", return_value=flydsl
                        ),
                        patch.object(
                            backend,
                            "_run_mega_routed",
                            side_effect=lambda *args, **kwargs: x * 2.5,
                        ),
                        patch.object(
                            deepseek_v2_module,
                            "post_experts_all_reduce",
                            side_effect=lambda t: t * 2,
                        ),
                    ):
                        output = moe(x, skip_shared_experts=skip)
                    torch.testing.assert_close(output, x * (2.5 if skip else 5.5))

    def test_amx_fused_output_does_not_repeat_scale_or_shared_add(self):
        """The fused CPU result only owes its TP reduction."""
        x = torch.ones(2, 3)
        moe = self._moe(reduce=True, inplace=True, backend=None)
        moe.experts.forward = lambda hidden_states, *args, **kwargs: (
            hidden_states.clone()
        )
        projection = SimpleNamespace(weight=torch.ones(1))
        moe.shared_experts = SimpleNamespace(
            gate_up_proj=projection, down_proj=projection
        )
        moe.shared_experts_is_int8 = moe.shared_experts_is_fp8 = False
        with (
            patch.object(deepseek_v2_module, "_is_cuda", False),
            patch.object(
                deepseek_v2_module, "use_intel_amx_backend", return_value=True
            ),
            patch.object(
                torch.ops.sgl_kernel,
                "shared_expert_cpu",
                create=True,
                side_effect=lambda x, w1, w2, routed, scale, *args: (
                    routed * scale + x * 3
                ),
            ) as fused_shared,
            patch.object(
                deepseek_v2_module,
                "post_experts_all_reduce",
                side_effect=lambda t: t * 2,
            ),
        ):
            output = moe(x)
        fused_shared.assert_called_once()
        torch.testing.assert_close(output, x * 11)

    def test_deferred_finalize_hands_sum_to_boundary_once(self):
        """Returning pieces and returning a boundary finalize have distinct owners."""
        from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
            MoeDeferredFinalize,
        )
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
        from sglang.srt.runtime_context import get_forward

        x = torch.ones(2, 3)
        for pieces_requested in (False, True):
            with self.subTest(pieces_requested=pieces_requested):
                moe = self._moe()
                handle = flashinfer_trtllm.FlashInferTrtllmDeferredFinalizeOutput(
                    gemm2_out=x * 2.5,
                    expert_weights=torch.ones(2, 1),
                    expanded_idx_to_permuted_idx=torch.arange(
                        2, dtype=torch.int32
                    ).reshape(2, 1),
                    top_k=1,
                )
                moe.experts.supports_deferred_finalize = True
                moe.experts.forward_deferred_finalize = lambda *args, **kwargs: handle
                moe._deferred_finalize_max_tokens = 0
                moe.topk = lambda *args, **kwargs: SimpleNamespace(
                    format=topk_module.TopKOutputFormat.BYPASSED
                )
                with (
                    get_forward().scoped(defer_moe_finalize=True),
                    patch.object(
                        flashinfer_trtllm,
                        "finalize_flashinfer_trtllm_deferred_output",
                        side_effect=lambda output, shared: output.gemm2_out + shared,
                    ),
                    patch.object(
                        deepseek_v2_module,
                        "sum_post_experts_output",
                        side_effect=lambda t: t * 2,
                    ),
                ):
                    output = moe.forward_normal_dual_stream(
                        x, leave_output_sum=pieces_requested
                    )
                    if pieces_requested:
                        self.assertIsInstance(output, deepseek_v2_module.MoEOutput)
                        torch.testing.assert_close(output.get_merged(), x * 5.5)
                    else:
                        self.assertIsInstance(output, MoeDeferredFinalize)
                        torch.testing.assert_close(output.complete(), x * 11)


if __name__ == "__main__":
    unittest.main()
