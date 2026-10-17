"""Termination tracking of the outlines grammar backend."""

import unittest

from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.constrained.outlines_backend import OutlinesGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_VOCAB = {"<eos>": 0, "a": 1, "b": 2}
_EOS, _A, _B = 0, 1, 2


class TestOutlinesTermination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(_VOCAB, unk_token="<eos>")),
            eos_token="<eos>",
        )
        cls.tokenizer = TransformerTokenizer(tokenizer)

    def _grammar(self, regex):
        return OutlinesGrammar(RegexGuide.from_regex(regex, self.tokenizer), None)

    def _mask(self, grammar):
        # Like SamplingBatchInfo.update_regex_vocab_mask: a terminated grammar's
        # row keeps the allocated all-allow value.
        mask = grammar.allocate_vocab_mask(len(_VOCAB), 1, "cpu")
        if not grammar.finished and not grammar.is_terminated():
            grammar.fill_vocab_mask(mask, 0)
        return mask[0]

    def test_eos_terminates_and_unconstrains(self):
        for regex, prefix in (("ab", [_A, _B]), ("a+", [_A])):
            with self.subTest(regex=regex):
                grammar = self._grammar(regex)
                for token in prefix:
                    grammar.accept_token(token)
                # An accepting state is not terminated: "a+" may still extend.
                self.assertFalse(grammar.is_terminated())
                self.assertFalse(self._mask(grammar)[_EOS])

                grammar.accept_token(_EOS)
                self.assertTrue(grammar.is_terminated())
                # The scheduler syncs grammar.finished with the request (ignore_eos).
                grammar.finished = False
                self.assertFalse(self._mask(grammar).any())

                # Later tokens are not fed to the guide and keep it terminated.
                grammar.accept_token(_B)
                self.assertTrue(grammar.is_terminated())

    def test_invalid_token_does_not_terminate(self):
        # An invalid token also sends the guide to state -1, but only a consumed
        # EOS terminates; the grammar must stay constrained (EOS only).
        grammar = self._grammar("ab")
        grammar.accept_token(_B)
        self.assertEqual(grammar.state, -1)
        self.assertFalse(grammar.is_terminated())
        mask = self._mask(grammar)
        self.assertFalse(mask[_EOS])
        self.assertTrue(mask[_A] and mask[_B])


if __name__ == "__main__":
    unittest.main()
