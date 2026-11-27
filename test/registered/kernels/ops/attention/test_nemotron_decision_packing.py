import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.models.nemotron_decision import (
    NemotronForDecision,
    PackedDecisionAttention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestPackedDecisionAttention(CustomTestCase):
    def test_both_layouts_keep_bidirectional_attention_and_training_window(self):
        for layout in ("state_sees_own_question", "state_sees_no_questions"):
            with self.subTest(layout=layout):
                config = SimpleNamespace(
                    is_causal=False,
                    sliding_window=None,
                    decision_attention_layout=layout,
                    state_end_token_id=101,
                    max_position_embeddings=262144,
                    rope_parameters={"original_max_position_embeddings": 16384},
                )
                attention = SimpleNamespace(
                    attn=RadixAttention(4, 64, 0.125, 2, 0),
                    max_position_embeddings=config.max_position_embeddings,
                )
                model = SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=attention)],
                    start_layer=0,
                    end_layer=1,
                )
                module = "sglang.srt.models.nemotron_decision."
                with (
                    patch(module + "Ministral3Model", return_value=model),
                    patch(module + "NemotronDecisionModel", return_value=model),
                ):
                    NemotronForDecision._init_model(None, config)
                self.assertEqual(attention.attn.attn_type, AttentionType.ENCODER_ONLY)
                self.assertEqual(attention.max_position_embeddings, 16384)

    def test_paged_read_matches_independent_mask_after_layout_changes(self):
        packed = PackedDecisionAttention()
        # Keep the total token count equal so a stale plan produces wrong attention.
        cases = (
            (
                [[5, 3, 4], None, [2, 5]],
                [12, 7, 7],
                list(range(5))
                + list(range(5, 8))
                + list(range(5, 9))
                + list(range(7))
                + list(range(2))
                + list(range(2, 7)),
            ),
            (
                [[3, 2, 5], None, [4, 3]],
                [10, 9, 7],
                list(range(3))
                + list(range(3, 5))
                + list(range(3, 8))
                + list(range(9))
                + list(range(4))
                + list(range(4, 7)),
            ),
        )
        for layouts, lengths, expected_positions in cases:
            with self.subTest(layouts=layouts):
                count = sum(lengths)
                batch = SimpleNamespace(
                    decision_layouts=layouts, extend_seq_lens_cpu=lengths
                )
                positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
                packed.prepare(batch, positions)
                torch.testing.assert_close(
                    positions.cpu(), torch.tensor(expected_positions)
                )
                generator = torch.Generator(device="cuda").manual_seed(42)
                q = torch.randn(
                    count,
                    4,
                    64,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                k = torch.randn(
                    count,
                    2,
                    64,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                v = torch.randn(
                    count,
                    2,
                    64,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                layer = SimpleNamespace(
                    tp_q_head_num=4,
                    tp_k_head_num=2,
                    tp_v_head_num=2,
                    head_dim=64,
                    v_head_dim=64,
                    scaling=1 / math.sqrt(64),
                )
                actual = packed.run(q, k, v, layer).view_as(q)
                allowed = torch.zeros(count, count, dtype=torch.bool, device="cuda")
                offset = 0
                for length, layout in zip(lengths, layouts):
                    if layout is None:
                        allowed[offset : offset + length, offset : offset + length] = (
                            True
                        )
                    else:
                        segment = torch.cat(
                            [
                                torch.full((n,), i, device="cuda")
                                for i, n in enumerate(layout)
                            ]
                        )
                        allowed[offset : offset + length, offset : offset + length] = (
                            segment[None, :] == 0
                        ) | (segment[:, None] == segment[None, :])
                    offset += length
                expected = F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None],
                    k.repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    v.repeat_interleave(2, dim=1).transpose(0, 1)[None],
                    attn_mask=allowed[None, None],
                )[0].transpose(0, 1)
                torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
                question_start = sum(layouts[0][:2])
                question_end = lengths[0]
                changed = v.clone()
                changed[question_start:question_end] += 10
                unaffected = packed.run(q, k, changed, layer).view_as(q)
                torch.testing.assert_close(
                    unaffected[:question_start], actual[:question_start], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    unaffected[question_end:], actual[question_end:], atol=0, rtol=0
                )
                self.assertGreater(
                    (
                        unaffected[question_start:question_end]
                        - actual[question_start:question_end]
                    )
                    .abs()
                    .max()
                    .item(),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
