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


if __name__ == "__main__":
    import unittest

    unittest.main()
