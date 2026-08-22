import pickle
import unittest
from array import array

import numpy as np
import torch

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_CUSTOMIZED_INFO = {
    "probe": [[None, True], [200, 201, 202], [False]],
    "score": [[0.5, None], [None, None, None], ["tag"]],
    "details": [[[1, "x"], {"ok": True}], [[], {}, None], [{"ratio": 0.25}]],
}

_NESTED_CUSTOMIZED_INFO = {
    "details": [
        [{"payload": [1, {"enabled": True, "tags": ["a", None]}]}],
        [[{"score": 0.5}, []]],
        [{"empty": {"items": []}}],
    ]
}

_NON_NATIVE_FACTORIES = {
    "array": lambda: array("i", [1, 2]),
    "ndarray": lambda: np.array([1, 2], dtype=np.int32),
    "tensor": lambda: torch.tensor([1, 2], dtype=torch.int32),
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
    output_token_sampling_mask=None,
    output_token_sampling_logprobs=None,
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
        self.assertEqual(
            [
                type(decoded.customized_info["probe"][0][0]),
                type(decoded.customized_info["probe"][0][1]),
                type(decoded.customized_info["probe"][1][0]),
                type(decoded.customized_info["score"][0][0]),
                type(decoded.customized_info["score"][2][0]),
                type(decoded.customized_info["details"][0][0]),
                type(decoded.customized_info["details"][0][1]),
            ],
            [type(None), bool, int, float, str, list, dict],
        )

    def test_batch_str_output_round_trips(self):
        decoded = self._round_trip(_make_str_output(_CUSTOMIZED_INFO))

        self.assertIsInstance(decoded, BatchStrOutput)
        self.assertEqual(decoded.customized_info, _CUSTOMIZED_INFO)

    def test_nested_json_values_round_trip(self):
        for make in (_make_token_id_output, _make_str_output):
            with self.subTest(make=make.__name__):
                decoded = self._round_trip(make(_NESTED_CUSTOMIZED_INFO))

                self.assertEqual(decoded.customized_info, _NESTED_CUSTOMIZED_INFO)

    def test_none_round_trips(self):
        for make in (_make_token_id_output, _make_str_output):
            with self.subTest(make=make.__name__):
                decoded = self._round_trip(make(None))

                self.assertIsNone(decoded.customized_info)

    def test_pickle_preserves_non_native_values(self):
        for make in (_make_token_id_output, _make_str_output):
            for name, factory in _NON_NATIVE_FACTORIES.items():
                with self.subTest(make=make.__name__, value=name):
                    original = factory()
                    output = make({"probe": [[original]]})

                    decoded = pickle.loads(
                        pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    actual = decoded.customized_info["probe"][0][0]

                    self.assertIs(type(actual), type(original))
                    if isinstance(original, torch.Tensor):
                        self.assertTrue(torch.equal(actual, original))
                    elif isinstance(original, np.ndarray):
                        np.testing.assert_array_equal(actual, original)
                    else:
                        self.assertEqual(actual, original)


if __name__ == "__main__":
    unittest.main()
