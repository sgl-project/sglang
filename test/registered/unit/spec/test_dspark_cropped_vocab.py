import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    VanillaMarkov,
    build_markov_head,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_markov_block,
    sample_draft_block,
    scatter_draft_vocab_logits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A draft that samples over its own narrower output vocabulary and maps the
# sampled ids onto the target vocabulary through the checkpoint's d2t table.
TARGET_VOCAB, DRAFT_VOCAB = 211, 64
RANK, HIDDEN, BS, GAMMA = 8, 16, 3, 4


def _draft_token_to_target() -> torch.Tensor:
    """A realistic injection: draft id -> target id, in the target vocabulary."""
    return torch.arange(DRAFT_VOCAB) * 3 % TARGET_VOCAB


def _identity_tp_sync() -> SimpleNamespace:
    return SimpleNamespace(sync=lambda site, tokens: tokens)


def _head(*, cropped: bool = True) -> VanillaMarkov:
    torch.manual_seed(0)
    return build_markov_head(
        SimpleNamespace(
            vocab_size=TARGET_VOCAB,
            markov_rank=RANK,
            hidden_size=HIDDEN,
            draft_vocab_size=DRAFT_VOCAB if cropped else TARGET_VOCAB,
        )
    ).float()


def _greedy_reference(head, base, hidden, anchor, table):
    """The block's own loop, written out: bias from the previous *target* token,
    argmax over the draft's output vocabulary, then mapped onto a target id."""
    state = head.init_state(hidden)
    prev = anchor.long()
    tokens, logits = [], []
    for step in range(GAMMA):
        bias, state = head.step_bias(prev, hidden_states=hidden[:, step], state=state)
        step_logits = base[:, step] + bias
        prev = table[step_logits.argmax(dim=-1)]
        tokens.append(prev)
        logits.append(step_logits)
    return torch.stack(tokens, dim=1), torch.stack(logits, dim=1)


class TestCroppedMarkovHead(CustomTestCase):
    def test_cropped_output_vocabulary_narrows_the_bias_only(self):
        head = _head()
        self.assertEqual(head.markov_w1.weight.shape, (TARGET_VOCAB, RANK))
        self.assertEqual(head.markov_w2.weight.shape, (DRAFT_VOCAB, RANK))
        self.assertEqual(head.vocab_size, TARGET_VOCAB)
        self.assertEqual(head.bias_vocab_size, DRAFT_VOCAB)
        self.assertEqual(
            head.step_bias(
                torch.zeros(BS, dtype=torch.long), hidden_states=None, state=None
            )[0].shape,
            (BS, DRAFT_VOCAB),
        )

    def test_a_full_vocabulary_draft_keeps_both_sides_equal(self):
        head = _head(cropped=False)
        self.assertEqual(head.markov_w2.weight.shape, (TARGET_VOCAB, RANK))
        self.assertEqual(head.bias_vocab_size, head.vocab_size)

    def test_a_wider_draft_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            build_markov_head(
                SimpleNamespace(
                    vocab_size=TARGET_VOCAB,
                    markov_rank=RANK,
                    hidden_size=HIDDEN,
                    draft_vocab_size=TARGET_VOCAB + 1,
                )
            )

    def test_cropping_is_only_supported_by_the_vanilla_head(self):
        for head_type in ("gated", "rnn"):
            with self.assertRaises(ValueError):
                build_markov_head(
                    SimpleNamespace(
                        vocab_size=TARGET_VOCAB,
                        markov_rank=RANK,
                        hidden_size=HIDDEN,
                        markov_head_type=head_type,
                        draft_vocab_size=DRAFT_VOCAB,
                    )
                )


class TestScatterDraftVocabLogits(CustomTestCase):
    def _check_wide_layout(self, out, draft_logits, table):
        self.assertEqual(out.shape, (*draft_logits.shape[:-1], TARGET_VOCAB))
        unmapped = torch.ones(TARGET_VOCAB, dtype=torch.bool)
        unmapped[table] = False
        self.assertEqual(float(out[..., unmapped].max()), float("-inf"))
        torch.testing.assert_close(out[..., table], draft_logits)
        # The draft's own distribution survives the widening: every unmapped
        # token weighs exactly zero in the softmax.
        torch.testing.assert_close(
            out.softmax(-1)[..., table], draft_logits.softmax(-1)
        )
        # Sampling in either layout picks the same token.
        torch.testing.assert_close(
            out.argmax(-1), table[draft_logits.argmax(-1)].reshape(out.shape[:-1])
        )

    def test_a_fresh_buffer_holds_the_draft_distribution(self):
        torch.manual_seed(0)
        draft_logits = torch.randn(BS, GAMMA, DRAFT_VOCAB)
        table = _draft_token_to_target()
        self._check_wide_layout(
            scatter_draft_vocab_logits(draft_logits, table, TARGET_VOCAB),
            draft_logits,
            table,
        )

    def test_a_reused_buffer_drops_the_previous_blocks_rows(self):
        torch.manual_seed(0)
        draft_logits = torch.randn(BS, GAMMA, DRAFT_VOCAB)
        table = _draft_token_to_target()
        out = torch.zeros(BS, GAMMA, TARGET_VOCAB)
        scatter_draft_vocab_logits(draft_logits, table, TARGET_VOCAB, out=out)
        self._check_wide_layout(out, draft_logits, table)


class TestDraftTokenToTargetLoading(CustomTestCase):
    """The checkpoint stores d2t delta-encoded; the draft keeps absolute ids."""

    class _Loader:
        _load_draft_token_to_target = DSparkDraftMixin._load_draft_token_to_target
        draft_output_vocab_size = DRAFT_VOCAB
        markov_head = SimpleNamespace(vocab_size=TARGET_VOCAB)

        def __init__(self):
            self.draft_token_to_target = torch.full(
                (DRAFT_VOCAB,), -1, dtype=torch.int64
            )

    def test_delta_entries_decode_to_absolute_target_ids(self):
        loader = self._Loader()
        table = _draft_token_to_target()
        loader._load_draft_token_to_target(
            (table - torch.arange(DRAFT_VOCAB)).to(torch.int64)
        )
        torch.testing.assert_close(loader.draft_token_to_target, table)

    def test_an_entry_outside_the_target_vocabulary_is_rejected(self):
        loader = self._Loader()
        delta = torch.zeros(DRAFT_VOCAB, dtype=torch.int64)
        # Draft id 5 would map onto target id 5 + (TARGET_VOCAB - 5).
        delta[DRAFT_VOCAB - 1] = TARGET_VOCAB
        with self.assertRaises(ValueError):
            loader._load_draft_token_to_target(delta)

    def test_a_table_of_the_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self._Loader()._load_draft_token_to_target(
                torch.zeros(DRAFT_VOCAB - 1, dtype=torch.int64)
            )


class TestCroppedDraftBlock(CustomTestCase):
    def _base(self):
        torch.manual_seed(0)
        return torch.randn(BS, GAMMA, DRAFT_VOCAB), torch.randn(BS, GAMMA, HIDDEN)

    def _anchor(self):
        torch.manual_seed(1)
        return torch.randint(0, TARGET_VOCAB, (BS,))

    def test_each_step_is_mapped_onto_a_target_id(self):
        head = _head()
        base, hidden = self._base()
        anchor, table = self._anchor(), _draft_token_to_target()
        tokens, logits = draft_markov_block(
            markov_head=head,
            base_logits=base,
            anchor_tokens=anchor,
            draft_hidden=hidden,
            step_temperatures=torch.zeros(BS),
            noise=DraftNoise(key=3, offset=torch.zeros((), dtype=torch.int64)),
            tp_sync=_identity_tp_sync(),
            greedy_draft=True,
            draft_token_to_target=table,
        )
        expected, _ = _greedy_reference(head, base, hidden, anchor, table)
        self.assertIsNone(logits)
        torch.testing.assert_close(tokens, expected)
        # Every drafted id is a target id, not one of the draft's own 64.
        self.assertTrue(bool((tokens >= DRAFT_VOCAB).any()))

    def test_the_kept_step_logits_span_the_target_vocabulary(self):
        head = _head()
        base, hidden = self._base()
        anchor, table = self._anchor(), _draft_token_to_target()
        block = sample_draft_block(
            base_logits=base,
            anchor_tokens=anchor,
            draft_hidden=hidden,
            sampling_info=None,
            markov_head=head,
            noise=DraftNoise(key=3, offset=torch.zeros((), dtype=torch.int64)),
            device="cpu",
            tp_sync=_identity_tp_sync(),
            greedy_draft=False,
            draft_token_to_target=table,
        )
        expected_tokens, expected_logits = _greedy_reference(
            head, base, hidden, anchor, table
        )
        self.assertEqual(block.corrected_logits.shape, (BS, GAMMA, TARGET_VOCAB))
        torch.testing.assert_close(block.draft_tokens, expected_tokens)
        torch.testing.assert_close(block.corrected_logits[..., table], expected_logits)
        torch.testing.assert_close(
            block.corrected_logits.softmax(-1)[..., table],
            expected_logits.softmax(-1),
        )

    def test_a_shared_vocabulary_draft_keeps_the_step_logits_unscattered(self):
        head = _head(cropped=False)
        base = torch.randn(BS, GAMMA, TARGET_VOCAB)
        hidden = torch.randn(BS, GAMMA, HIDDEN)
        block = sample_draft_block(
            base_logits=base,
            anchor_tokens=self._anchor(),
            draft_hidden=hidden,
            sampling_info=None,
            markov_head=head,
            noise=DraftNoise(key=3, offset=torch.zeros((), dtype=torch.int64)),
            device="cpu",
            tp_sync=_identity_tp_sync(),
            greedy_draft=False,
        )
        self.assertEqual(block.corrected_logits.shape, (BS, GAMMA, TARGET_VOCAB))


if __name__ == "__main__":
    unittest.main()
