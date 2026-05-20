# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StandaloneWorker._validate_vocab_compatibility.

All tests are CPU-only: no server, no GPU, no model loading needed.
Tokenizers are replaced with lightweight MagicMock fakes.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.speculative.standalone_worker import StandaloneWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_VOCAB = {"hello": 0, "world": 1, "foo": 2, "<eos>": 3}
_VOCAB_DIFFERENT_MAPPING = {"hello": 0, "world": 1, "foo": 99, "<eos>": 3}
_VOCAB_DIFFERENT_SIZE = {"hello": 0, "world": 1, "foo": 2, "<eos>": 3, "extra": 4}


def _tok(vocab: dict):
    tok = MagicMock()
    tok.get_vocab.return_value = dict(vocab)
    return tok


def _validate(target_size, draft_size, target_tok, draft_tok):
    StandaloneWorker._validate_vocab_compatibility(
        target_vocab_size=target_size,
        draft_vocab_size=draft_size,
        target_tokenizer=target_tok,
        draft_tokenizer=draft_tok,
    )


class TestStandaloneVocabCheck(CustomTestCase):
    def test_identical_vocab_passes(self):
        """Same size and same token mapping — should not raise."""
        _validate(len(_VOCAB), len(_VOCAB), _tok(_VOCAB), _tok(_VOCAB))

    def test_mismatched_vocab_size_raises(self):
        """Different vocab_size values must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate(
                len(_VOCAB),
                len(_VOCAB_DIFFERENT_SIZE),
                _tok(_VOCAB),
                _tok(_VOCAB_DIFFERENT_SIZE),
            )
        self.assertIn("vocab_size", str(ctx.exception))
        self.assertIn("TLI", str(ctx.exception))

    def test_same_size_different_mapping_raises(self):
        """Same vocab_size but different token strings must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate(
                len(_VOCAB),
                len(_VOCAB_DIFFERENT_MAPPING),
                _tok(_VOCAB),
                _tok(_VOCAB_DIFFERENT_MAPPING),
            )
        # vocab sizes happen to match here, so we hit the mapping check
        self.assertIn("TLI", str(ctx.exception))

    def test_tokenizer_without_get_vocab_skips_mapping_check(self):
        """Tokenizer types that lack get_vocab() (e.g. TiktokenTokenizer) must not raise."""
        no_get_vocab = MagicMock(spec=[])  # spec=[] means no attributes at all
        _validate(
            len(_VOCAB), len(_VOCAB), no_get_vocab, _tok(_VOCAB_DIFFERENT_MAPPING)
        )
        _validate(
            len(_VOCAB), len(_VOCAB), _tok(_VOCAB_DIFFERENT_MAPPING), no_get_vocab
        )

    def test_none_target_tokenizer_skips_mapping_check(self):
        """If target tokenizer is None, mapping check is skipped even if draft differs."""
        _validate(len(_VOCAB), len(_VOCAB), None, _tok(_VOCAB_DIFFERENT_MAPPING))

    def test_none_draft_tokenizer_skips_mapping_check(self):
        """If draft tokenizer is None, mapping check is skipped even if target differs."""
        _validate(len(_VOCAB), len(_VOCAB), _tok(_VOCAB_DIFFERENT_MAPPING), None)

    def test_both_tokenizers_none_skips_mapping_check(self):
        """Both tokenizers None (--skip-tokenizer-init) — only size is checked."""
        _validate(len(_VOCAB), len(_VOCAB), None, None)

    def test_error_message_contains_vocab_sizes(self):
        """ValueError for size mismatch should include both sizes."""
        with self.assertRaises(ValueError) as ctx:
            _validate(1000, 2000, None, None)
        msg = str(ctx.exception)
        self.assertIn("1000", msg)
        self.assertIn("2000", msg)

    def test_error_message_suggests_tli_for_size_mismatch(self):
        """Both size-mismatch and mapping-mismatch errors suggest TLI."""
        with self.assertRaises(ValueError) as ctx:
            _validate(1000, 2000, None, None)
        self.assertIn("TLI", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _validate(
                len(_VOCAB),
                len(_VOCAB_DIFFERENT_MAPPING),
                _tok(_VOCAB),
                _tok(_VOCAB_DIFFERENT_MAPPING),
            )
        self.assertIn("TLI", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
