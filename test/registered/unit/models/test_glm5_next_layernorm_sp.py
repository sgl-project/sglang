from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Rank 1 of a 2-rank TP group over 3 token rows: it holds row 2 plus one pad row.
FULL = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
PADDED = torch.cat([FULL, torch.zeros(1, 2)])
LOCAL = PADDED[2:4]
POSITIONS = torch.arange(3)


def _rank1_group():
    def reduce_scatter(output, full_partial):
        # Both ranks hold the same partial in this test, so the sum doubles it.
        output.copy_(full_partial[2:4] * 2)

    return SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        all_gather_into_tensor=lambda output, local: output.copy_(PADDED),
        reduce_scatter_tensor=MagicMock(side_effect=reduce_scatter),
    )


def _layer(self_attn, *, is_linear_attn):
    layer = SimpleNamespace(is_linear_attn=is_linear_attn, self_attn=self_attn)
    layer._run_attn = partial(glm5_next.Glm5NextDecoderLayer._run_attn, layer)
    return layer


def _run_attn_sp(layer, prev_topk_indices=None):
    group = _rank1_group()
    with (
        patch(
            "sglang.srt.layers.layernorm_sp.get_parallel",
            return_value=SimpleNamespace(tp_group=group),
        ),
        get_forward().scoped(sp_active=True),
    ):
        output = glm5_next.Glm5NextDecoderLayer._run_attn_sp(
            layer,
            positions=POSITIONS,
            hidden_states=LOCAL,
            forward_batch=None,
            zero_allocator=None,
            prev_topk_indices=prev_topk_indices,
        )
    group.reduce_scatter_tensor.assert_called_once()
    return output


class TestGlm5NextLayerNormSPAttention(CustomTestCase):
    def tearDown(self):
        get_attn_tp_context().clear_attn_inputs()

    def test_mla_reads_its_latent_from_the_gathered_rows(self):
        seen = {}

        class FakeMLA:
            def prepare_qkv_latent(self, hidden_states, forward_batch):
                return hidden_states * 3

            def __call__(self, *, hidden_states, input_on_attn_tp_slices, **kwargs):
                seen["sp_active"] = get_forward().sp_active
                seen["on_slices"] = input_on_attn_tp_slices
                seen["prev"] = kwargs["prev_topk_indices"]
                latent = get_attn_tp_context().fetch_qkv_latent()
                return hidden_states + latent, torch.tensor([7, 8, 9])

        mla = FakeMLA()
        # What the SP boundary leaves behind: this rank's rows only.
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(LOCAL, None, mla.prepare_qkv_latent)
        )
        prev = torch.tensor([1, 2, 3])
        output, topk = _run_attn_sp(_layer(mla, is_linear_attn=False), prev)

        torch.testing.assert_close(output, (PADDED * 4)[2:4] * 2)
        torch.testing.assert_close(topk, torch.tensor([7, 8, 9]))
        self.assertIs(seen["prev"], prev)
        self.assertFalse(seen["sp_active"])
        self.assertFalse(seen["on_slices"])

    def test_kda_sees_exactly_the_sequence_rows(self):
        seen = {}

        def kda(*, hidden_states, **kwargs):
            seen["rows"] = hidden_states.shape[0]
            return hidden_states * 2

        output, topk = _run_attn_sp(_layer(kda, is_linear_attn=True))

        self.assertEqual(seen["rows"], FULL.shape[0])
        torch.testing.assert_close(output, (PADDED * 2)[2:4] * 2)
        self.assertIsNone(topk)


def _symmetric_rank1_group():
    """Rank 1 of 2 where rank 0 holds the same rows, so a sum doubles them."""
    return SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        all_gather_into_tensor=lambda output, local: output.copy_(
            torch.cat([local, local])
        ),
        reduce_scatter_tensor=lambda output, full_partial: output.copy_(
            full_partial.tensor_split(2, dim=0)[1] * 2
        ),
    )


class TestGlm5NextLayerNormSPForward(CustomTestCase):
    """The decoder layer under SP: attention and the FFN see the sequence rows,
    and the FFN output is summed over TP only when it is a partial."""

    def _forward(self, mlp_cls):
        rows_seen = {}

        class FakeMlp(mlp_cls):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, hidden_states, forward_batch, allocator):
                rows_seen["mlp"] = hidden_states.shape[0]
                return hidden_states * 3

        def kda(*, hidden_states, **kwargs):
            rows_seen["attn"] = hidden_states.shape[0]
            return hidden_states * 2

        layer = glm5_next.Glm5NextDecoderLayer.__new__(glm5_next.Glm5NextDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.is_linear_attn = True
        layer.self_attn = kda
        layer.mlp = FakeMlp()
        same = lambda hidden_states, forward_batch, **kwargs: hidden_states
        layer.attn_boundary = SimpleNamespace(
            prepare=same, finish=same, input_on_attn_tp_slices=False
        )
        layer.ffn_boundary = SimpleNamespace(prepare=same, finish=same)
        with (
            patch(
                "sglang.srt.layers.layernorm_sp.get_parallel",
                return_value=SimpleNamespace(tp_group=_symmetric_rank1_group()),
            ),
            get_forward().scoped(sp_active=True),
        ):
            # 3 sequence rows; this rank holds row 2 and a pad row.
            output, _ = layer(POSITIONS, torch.tensor([[1.0], [0.0]]), None)
        self.assertEqual(rows_seen, {"attn": 3, "mlp": 3})
        return output[0, 0].item()

    def test_the_ffn_output_is_summed_only_when_it_owes_the_tp_sum(self):
        # Attention: 2x, summed over 2 ranks -> 4. FFN: 3x -> 12, summed -> 24.
        for mlp_cls, target, complete in (
            (glm5_next.Glm5NextMoE, "post_experts_output_is_complete", False),
            (glm5_next.Glm5NextMoE, "post_experts_output_is_complete", True),
            (glm5_next.Glm5NextMLP, "is_dense_ffn_fully_dp", False),
            (glm5_next.Glm5NextMLP, "is_dense_ffn_fully_dp", True),
        ):
            with (
                self.subTest(mlp=mlp_cls.__name__, complete=complete),
                patch.object(glm5_next, target, return_value=complete),
            ):
                self.assertEqual(self._forward(mlp_cls), 12.0 if complete else 24.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
