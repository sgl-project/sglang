"""Msgpack round-trip tests for customized_info on batch outputs."""

import unittest
from array import array

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Shape produced by _GenerationStreamAccumulator: one row per request, padded
# with None for requests that did not report the key in this chunk.
_CUSTOMIZED_INFO = {
    "probe": [[None, None], [200, 201, 202], [None]],
    "score": [[0.5, None], [None, None, None], ["tag"]],
}

_COMMON_FIELDS = dict(
    rids=["r0", "r1", "r2"],
    finished_reasons=[None, None, None],
    output_ids=None,
    prompt_tokens=[1, 1, 1],
    completion_tokens=[2, 3, 1],
    reasoning_tokens=[0, 0, 0],
    cached_tokens=[0, 0, 0],
    input_token_logprobs_val=None,
    input_token_logprobs_idx=None,
    output_token_logprobs_val=None,
    output_token_logprobs_idx=None,
    input_top_logprobs_val=None,
    input_top_logprobs_idx=None,
    output_top_logprobs_val=None,
    output_top_logprobs_idx=None,
    input_token_ids_logprobs_val=None,
    input_token_ids_logprobs_idx=None,
    output_token_ids_logprobs_val=None,
    output_token_ids_logprobs_idx=None,
    output_token_entropy_val=None,
    output_hidden_states=None,
    routed_experts=None,
    indexer_topk=None,
    placeholder_tokens_idx=None,
    placeholder_tokens_val=None,
)


def _make_token_id_output(customized_info):
    return BatchTokenIDOutput(
        decoded_texts=["", "", ""],
        decode_ids=[array("i", [10]), array("i", [20]), array("i", [30])],
        read_offsets=[0, 0, 0],
        skip_special_tokens=[True, True, True],
        spaces_between_special_tokens=[True, True, True],
        no_stop_trim=[False, False, False],
        customized_info=customized_info,
        **_COMMON_FIELDS,
    )


def _make_str_output(customized_info):
    return BatchStrOutput(
        output_strs=["a", "b", "c"],
        customized_info=customized_info,
        **_COMMON_FIELDS,
    )


class TestCustomizedInfoMsgpack(CustomTestCase):
    def _round_trip(self, output):
        return msgpack_decode(msgpack_encode(output))

    def test_batch_token_id_output_round_trips(self):
        decoded = self._round_trip(_make_token_id_output(_CUSTOMIZED_INFO))

        self.assertIsInstance(decoded, BatchTokenIDOutput)
        self.assertEqual(decoded.customized_info, _CUSTOMIZED_INFO)

    def test_batch_str_output_round_trips(self):
        decoded = self._round_trip(_make_str_output(_CUSTOMIZED_INFO))

        self.assertIsInstance(decoded, BatchStrOutput)
        self.assertEqual(decoded.customized_info, _CUSTOMIZED_INFO)

    def test_none_round_trips(self):
        for make in (_make_token_id_output, _make_str_output):
            with self.subTest(make=make.__name__):
                decoded = self._round_trip(make(None))

                self.assertIsNone(decoded.customized_info)


if __name__ == "__main__":
    unittest.main()
