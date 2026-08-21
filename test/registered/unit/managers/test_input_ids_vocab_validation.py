"""
Unit tests for prompt input_ids vocab-range validation in TokenizerManager.

A request whose prompt carried a token id outside [0, vocab_size) reached the
embedding index_select and tripped a device-side assert, which poisons the CUDA
context and aborts the scheduler -- i.e. one request could take down the server.
_validate_input_ids_in_vocab existed to prevent that but had no callers.

Covers:
  - _validate_input_ids_in_vocab rejects id >= vocab_size, id < 0 and non-ints
  - it accepts in-range ids, both flat and batched
  - _validate_one_request actually calls it (the regression that mattered)
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 32


def _make_tm() -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = MagicMock()
    tm.model_config.vocab_size = VOCAB_SIZE
    # _validate_one_request also does a length check before the vocab check.
    tm.context_len = 2048
    tm.num_reserved_tokens = 0
    tm.allow_auto_truncate = False
    tm.is_generation = True
    tm.server_args = MagicMock()
    tm.server_args.enable_lora = False
    # Assigned in __init__, so absent when the object is built with __new__.
    tm.validate_total_tokens = MagicMock()
    return tm


class TestValidateInputIdsInVocab(CustomTestCase):
    def test_accepts_in_range(self):
        tm = _make_tm()
        tm._validate_input_ids_in_vocab([0, 1, VOCAB_SIZE - 1], VOCAB_SIZE)

    def test_accepts_in_range_batched(self):
        tm = _make_tm()
        tm._validate_input_ids_in_vocab([[0, 1], [VOCAB_SIZE - 1]], VOCAB_SIZE)

    def test_rejects_id_at_or_above_vocab_size(self):
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_input_ids_in_vocab([VOCAB_SIZE], VOCAB_SIZE)
        with self.assertRaises(ValueError):
            tm._validate_input_ids_in_vocab([10**9], VOCAB_SIZE)

    def test_rejects_negative_id(self):
        # A negative id indexes the embedding from the far end and reaches the
        # same index_select, so the upper bound alone is not enough.
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_input_ids_in_vocab([-1], VOCAB_SIZE)

    def test_rejects_non_int(self):
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_input_ids_in_vocab([1.5], VOCAB_SIZE)

    def test_rejects_bad_id_inside_a_batch(self):
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_input_ids_in_vocab([[0, 1], [VOCAB_SIZE]], VOCAB_SIZE)


class TestValidateOneRequestCallsVocabCheck(CustomTestCase):
    """The guard existed; the regression was that nothing called it."""

    def _obj(self):
        obj = MagicMock(spec=GenerateReqInput)
        obj.sampling_params = {}
        obj.token_ids_logprob = None
        obj.return_hidden_states = False
        obj.input_embeds = None
        obj.lora_path = None
        return obj

    def test_out_of_range_prompt_is_rejected(self):
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_one_request(self._obj(), [0, VOCAB_SIZE])

    def test_negative_prompt_id_is_rejected(self):
        tm = _make_tm()
        with self.assertRaises(ValueError):
            tm._validate_one_request(self._obj(), [-1])

    def test_in_range_prompt_is_accepted(self):
        tm = _make_tm()
        tm._validate_one_request(self._obj(), [0, 1, VOCAB_SIZE - 1])


if __name__ == "__main__":
    unittest.main()
