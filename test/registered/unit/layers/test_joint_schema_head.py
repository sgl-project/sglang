"""Unit tests for the joint schema head of Clef checkpoints: decision layouts, and
the batched forward against a per-request loop over the stock torch modules."""

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.joint_schema_head import (
    JointSchemaHead,
    JointSchemaPooler,
    LayoutQuestion,
    pack_decision_layout,
    parse_decision_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _questions(offset=0):
    return [
        LayoutQuestion(
            1,
            (10 + offset, 14 + offset),
            (
                (16 + offset, 17 + offset),
                (17 + offset, 19 + offset),
                (19 + offset, 21 + offset),
            ),
        ),
        LayoutQuestion(
            0,
            (22 + offset, 25 + offset),
            ((26 + offset, 28 + offset), (29 + offset, 31 + offset)),
        ),
    ]


class TestDecisionLayout(CustomTestCase):
    def test_round_trip(self):
        layout = pack_decision_layout(40, _questions())
        self.assertEqual(parse_decision_layout(layout, 40), _questions())

    def test_expanded_images_shift_every_span(self):
        layout = pack_decision_layout(40, _questions())
        self.assertEqual(parse_decision_layout(layout, 140), _questions(offset=100))

    def test_rejects_layouts_that_do_not_fit_the_prompt(self):
        layout = pack_decision_layout(40, _questions())
        invalid = {
            "shorter prompt": (layout, 39),
            "not ints": ([float(value) for value in layout], 40),
            "truncated": (layout[:-1], 40),
            "trailing values": (layout + [0], 40),
            "no questions": ([40, 0], 40),
            "empty span": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 5), ((6, 7),))]),
                40,
            ),
            "past the prompt": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 6), ((6, 41),))]),
                40,
            ),
            "unknown type": (
                pack_decision_layout(40, [LayoutQuestion(3, (5, 6), ((6, 7),))]),
                40,
            ),
        }
        for name, (values, num_tokens) in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_decision_layout(values, num_tokens)


class _RecordingHead:
    """Stands in for the head, scoring each option with its request's length."""

    def __init__(self):
        self.calls = []

    def forward_batch(self, hidden, input_ids, lengths, layouts, embedding_weight):
        self.calls.append(lengths)
        return [
            torch.full((sum(len(q.option_spans) for q in layout),), float(length))
            for length, layout in zip(lengths, layouts)
        ]


class TestJointSchemaPooler(CustomTestCase):
    def test_a_layout_cut_from_its_prompt_is_skipped_and_neighbors_are_scored(self):
        """A prompt shorter than its layout, as after a truncation, must not stop the
        scheduler or the other requests of its batch."""
        head = _RecordingHead()
        pooler = JointSchemaPooler(head, lambda: torch.zeros(1))
        layout = pack_decision_layout(40, _questions())
        lengths = [40, 35, 40]
        forward_batch = SimpleNamespace(
            decision_layouts=[layout, layout, layout],
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=[0, 0, 0],
            batch_size=3,
            input_ids=torch.zeros(sum(lengths), dtype=torch.long),
        )
        output = pooler(torch.zeros(sum(lengths), 8), forward_batch)
        self.assertEqual([len(scores) for scores in output.embeddings], [5, 0, 5])
        self.assertEqual(head.calls, [[40], [40]])


def _reference_logits(head, hidden, input_ids, questions, lm_head):
    """One request through the modules' own forward methods, as the checkpoint's loop does."""
    normalized = head.hidden_norm(hidden)
    memory = head.memory_projection(normalized).unsqueeze(0)
    global_vector = normalized[-1]

    def mean(values, span):
        return values[span[0] : span[1]].mean(dim=0)

    question_vectors = torch.stack(
        [mean(normalized, q.question_span) for q in questions]
    )
    queries, lexicals = [], []
    for index, question in enumerate(questions):
        context = torch.stack(
            [mean(normalized, span) for span in question.option_spans]
        )
        lexical = torch.stack(
            [
                lm_head[input_ids[start:end]].mean(dim=0)
                for start, end in question.option_spans
            ]
        )
        queries.append(
            head.option_context_projection(context)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors[index]).unsqueeze(0)
        )
        lexicals.append(lexical)
    routed = torch.cat(queries).unsqueeze(0)
    for layer in head.evidence_layers:
        attended, _ = layer.attention(
            layer.query_norm(routed),
            layer.memory_norm(memory),
            layer.memory_norm(memory),
            need_weights=False,
        )
        routed = routed + attended
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))
    options = list(torch.split(routed[0], [len(q.option_spans) for q in questions]))
    base_fields = head.question_projection(question_vectors)
    summaries = []
    for field, routed_options in zip(base_fields, options):
        weights = torch.softmax(
            routed_options @ field / math.sqrt(field.shape[-1]), dim=0
        )
        summaries.append((weights.unsqueeze(-1) * routed_options).sum(dim=0))
    fields = (
        base_fields
        + head.option_summary_norm(torch.stack(summaries))
        + head.global_projection(global_vector).unsqueeze(0)
        + head.type_embedding(
            torch.tensor([q.question_type for q in questions], device=hidden.device)
        )
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    fields = head.field_norm(fields[0])
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
    logits = []
    for index, (field, lexical, routed_options) in enumerate(
        zip(fields, lexicals, options)
    ):
        anchor = F.normalize(question_vectors[index] + global_vector, dim=-1)
        prior = prior_scale * (F.normalize(lexical, dim=-1) @ anchor)
        normed = head.option_norm(routed_options)
        repeated = field.unsqueeze(0).expand_as(normed)
        features = torch.cat(
            [repeated, normed, repeated * normed, torch.abs(repeated - normed)], dim=-1
        )
        residual = head.residual_scorer(features).squeeze(-1)
        cosine = F.cosine_similarity(repeated, normed, dim=-1)
        logits.append(
            prior
            + torch.sigmoid(head.residual_gate) * (joint_scale * cosine + residual)
        )
    return torch.cat(logits).float()


@unittest.skipUnless(torch.cuda.is_available(), "the batched head uses FlashAttention")
class TestJointSchemaHeadBatch(CustomTestCase):
    def test_batched_forward_matches_each_request_alone(self):
        torch.manual_seed(0)
        hidden_size, vocab = 256, 512
        head = JointSchemaHead(
            hidden_size, width=128, routing_layers=2, layers=2, heads=4, feedforward=256
        )
        for parameter in (
            head.prior_logit_scale,
            head.joint_logit_scale,
            head.residual_gate,
        ):
            parameter.data.fill_(0.5)
        head = head.to("cuda", torch.bfloat16).eval()
        lm_head = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.bfloat16)
        lengths = [40, 140, 64]
        layouts = [
            _questions(),
            _questions(offset=100),
            [
                LayoutQuestion(
                    2, (3, 9), tuple((10 + 2 * i, 12 + 2 * i) for i in range(20))
                )
            ],
        ]
        hidden = torch.randn(
            sum(lengths), hidden_size, device="cuda", dtype=torch.bfloat16
        )
        input_ids = torch.randint(0, vocab, (sum(lengths),), device="cuda")
        with torch.no_grad():
            batched = head.forward_batch(hidden, input_ids, lengths, layouts, lm_head)
            start = 0
            for length, questions, logits in zip(lengths, layouts, batched):
                end = start + length
                expected = _reference_logits(
                    head, hidden[start:end], input_ids[start:end], questions, lm_head
                )
                self.assertEqual(logits.shape, expected.shape)
                torch.testing.assert_close(logits, expected, atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(
                    logits.softmax(-1), expected.softmax(-1), atol=1e-2, rtol=0
                )
                start = end


if __name__ == "__main__":
    unittest.main()
