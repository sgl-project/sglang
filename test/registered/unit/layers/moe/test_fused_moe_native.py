import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.nn import functional as F

from sglang.srt.layers.moe import topk
from sglang.srt.layers.moe.fused_moe_native import (
    fused_moe_forward_native,
    moe_forward_native,
)
from sglang.srt.layers.moe.topk import (
    ScaledStandardTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    select_experts,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_config(
    activation: str, is_gated: bool, routed_scaling_factor: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        activation=activation,
        is_gated=is_gated,
        apply_router_weight_on_input=False,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        routed_scaling_factor=routed_scaling_factor,
    )


def _reference_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    is_gated: bool,
) -> torch.Tensor:
    activations = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu2": lambda x: F.relu(x) ** 2,
    }
    output = torch.zeros_like(hidden_states)
    for token_index in range(hidden_states.shape[0]):
        for slot_index in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_index, slot_index])
            up = w13_weight[expert_id] @ hidden_states[token_index]
            if is_gated:
                gate, up = up.chunk(2)
                intermediate = activations[activation](gate) * up
            else:
                intermediate = activations[activation](up)
            output[token_index] += topk_weights[token_index, slot_index] * (
                w2_weight[expert_id] @ intermediate
            )
    return output


class TestFusedMoeNative(CustomTestCase):
    def test_noncontiguous_topk_ids(self):
        torch.manual_seed(0)
        layer = SimpleNamespace(
            num_experts=3,
            w13_weight=torch.randn(3, 4, 2),
            w2_weight=torch.randn(3, 2, 2),
        )
        hidden_states = torch.randn(3, 2)
        topk_weights = torch.tensor(
            [[0.75, 0.25], [0.4, 0.6], [0.9, 0.1]], dtype=torch.float32
        )
        padded_topk_ids = torch.tensor(
            [[0, -1, 2, -1], [1, -1, 0, -1], [2, -1, 1, -1]],
            dtype=torch.int64,
        )
        topk_ids = padded_topk_ids[:, ::2]
        self.assertFalse(topk_ids.is_contiguous())

        config = _make_config(activation="silu", is_gated=True)
        output = moe_forward_native(
            layer,
            hidden_states,
            (topk_weights, topk_ids, None),
            config,
        )
        reference = moe_forward_native(
            layer,
            hidden_states,
            (topk_weights, topk_ids.contiguous(), None),
            config,
        )

        torch.testing.assert_close(output, reference)

    def test_native_paths_match_reference(self):
        """Both native paths honor is_gated, including non-gated relu2 experts."""
        num_experts, hidden_size, intermediate_size, top_k = 4, 8, 6, 2
        cases = [
            ("silu", True),
            ("relu2", False),
            ("silu", False),
            ("gelu", False),
        ]
        for activation, is_gated in cases:
            with self.subTest(activation=activation, is_gated=is_gated):
                torch.manual_seed(0)
                w13_rows = intermediate_size * (2 if is_gated else 1)
                config = _make_config(activation=activation, is_gated=is_gated)
                layer = SimpleNamespace(
                    num_experts=num_experts,
                    w13_weight=torch.randn(num_experts, w13_rows, hidden_size),
                    w2_weight=torch.randn(num_experts, hidden_size, intermediate_size),
                    moe_runner_config=config,
                )
                hidden_states = torch.randn(5, hidden_size)
                topk_weights = torch.rand(5, top_k)
                topk_ids = torch.stack(
                    [torch.randperm(num_experts)[:top_k] for _ in range(5)]
                )
                reference = _reference_moe(
                    hidden_states,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation,
                    is_gated,
                )

                fused_output = fused_moe_forward_native(
                    layer,
                    SimpleNamespace(
                        hidden_states=hidden_states,
                        topk_output=(topk_weights, topk_ids, None),
                    ),
                ).hidden_states
                looped_output = moe_forward_native(
                    layer,
                    hidden_states,
                    (topk_weights, topk_ids, None),
                    config,
                )

                torch.testing.assert_close(fused_output, reference)
                torch.testing.assert_close(looped_output, reference)

    def test_fused_native_applies_routed_scaling_factor(self):
        """Matches the Triton runner, which scales the combined routed output
        unless TopK already folded the factor into topk_weights: because the
        layer fuses it there, or because the router applied it on its own."""
        num_experts, hidden_size, intermediate_size, top_k = 4, 8, 6, 2
        cases = [
            (activation, is_gated, fused_in_topk, output_cls)
            for activation, is_gated in (("relu2", False), ("silu", True))
            for fused_in_topk in (False, True)
            for output_cls in (StandardTopKOutput, ScaledStandardTopKOutput)
        ]
        for activation, is_gated, fused_in_topk, output_cls in cases:
            with self.subTest(
                activation=activation,
                is_gated=is_gated,
                fused_in_topk=fused_in_topk,
                output_cls=output_cls.__name__,
            ):
                torch.manual_seed(0)
                w13_rows = intermediate_size * (2 if is_gated else 1)
                layer = SimpleNamespace(
                    w13_weight=torch.randn(num_experts, w13_rows, hidden_size),
                    w2_weight=torch.randn(num_experts, hidden_size, intermediate_size),
                    moe_runner_config=_make_config(
                        activation=activation,
                        is_gated=is_gated,
                        routed_scaling_factor=2.5,
                    ),
                    should_fuse_routed_scaling_factor_in_topk=fused_in_topk,
                )
                hidden_states = torch.randn(5, hidden_size)
                topk_weights = torch.rand(5, top_k)
                topk_ids = torch.stack(
                    [torch.randperm(num_experts)[:top_k] for _ in range(5)]
                )
                reference = _reference_moe(
                    hidden_states,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation,
                    is_gated,
                )

                output = fused_moe_forward_native(
                    layer,
                    SimpleNamespace(
                        hidden_states=hidden_states,
                        topk_output=output_cls(topk_weights, topk_ids, None),
                    ),
                ).hidden_states

                folded_in_topk = (
                    fused_in_topk or output_cls.routed_scaling_factor_applied
                )
                expected = reference if folded_in_topk else 2.5 * reference
                torch.testing.assert_close(output, expected)

    def test_routed_scaling_factor_applied_once_per_router(self):
        """Through select_experts, the factor reaches the routed output exactly
        once: aiter's biased grouped kernel folds it into topk_weights, while the
        grouped router without a correction bias leaves it to the MoE."""
        num_experts, hidden_size, intermediate_size, top_k = 8, 8, 6, 2
        factor = 2.5
        # aiter_biased_grouped_topk scales the routed weights unconditionally.
        biased_weights = torch.tensor([[0.7, 0.3]])
        biased_ids = torch.tensor([[1, 5]])
        cases = [
            (use_aiter, has_bias)
            for use_aiter in (False, True)
            for has_bias in (False, True)
        ]
        for use_aiter, has_bias in cases:

            def biased_grouped_topk(**kwargs):
                scale = kwargs["routed_scaling_factor"] if use_aiter else 1.0
                return biased_weights * scale, biased_ids

            with (
                self.subTest(use_aiter=use_aiter, has_bias=has_bias),
                mock.patch.object(topk, "_use_aiter", use_aiter),
                mock.patch.object(topk, "biased_grouped_topk", biased_grouped_topk),
            ):
                torch.manual_seed(0)
                layer = SimpleNamespace(
                    w13_weight=torch.randn(num_experts, intermediate_size, hidden_size),
                    w2_weight=torch.randn(num_experts, hidden_size, intermediate_size),
                    moe_runner_config=_make_config(
                        activation="relu2",
                        is_gated=False,
                        routed_scaling_factor=factor,
                    ),
                    should_fuse_routed_scaling_factor_in_topk=False,
                )
                hidden_states = torch.randn(1, hidden_size)
                topk_output = select_experts(
                    hidden_states=hidden_states,
                    router_logits=torch.randn(1, num_experts),
                    topk_config=TopKConfig(
                        top_k=top_k,
                        use_grouped_topk=True,
                        topk_group=1,
                        num_expert_group=2,
                        correction_bias=torch.zeros(num_experts) if has_bias else None,
                        routed_scaling_factor=factor,
                    ),
                )
                self.assertEqual(
                    topk_output.routed_scaling_factor_applied, use_aiter and has_bias
                )
                unscaled_weights = (
                    biased_weights if has_bias else topk_output.topk_weights
                )
                reference = _reference_moe(
                    hidden_states,
                    layer.w13_weight,
                    layer.w2_weight,
                    unscaled_weights,
                    topk_output.topk_ids,
                    "relu2",
                    False,
                )

                output = fused_moe_forward_native(
                    layer,
                    SimpleNamespace(
                        hidden_states=hidden_states, topk_output=topk_output
                    ),
                ).hidden_states

                torch.testing.assert_close(output, factor * reference)

    def test_non_gated_rejects_unknown_activation(self):
        layer = SimpleNamespace(
            num_experts=2,
            w13_weight=torch.randn(2, 4, 3),
            w2_weight=torch.randn(2, 3, 4),
            moe_runner_config=_make_config(activation="situ", is_gated=False),
        )
        hidden_states = torch.randn(2, 3)
        topk_output = (torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64), None)
        with self.assertRaises(ValueError):
            fused_moe_forward_native(
                layer,
                SimpleNamespace(hidden_states=hidden_states, topk_output=topk_output),
            )


if __name__ == "__main__":
    unittest.main()
