"""CPU tests for the caller's mm_token_expansion_start_len.

expanded prefix | tokens to expand -> start_len = prefix length in supplied IDs
positive start_len     -> require non-empty supplied IDs; 0 expands everything
start_len list         -> one per prompt, repeated for n > 1 like input_ids
single prompt          -> integer start_len only, whatever n is
batch / n > 1          -> broadcast a scalar start_len
text without IDs       -> omit start_len; tokenize later

"""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMMTokenExpansionStartLen(unittest.TestCase):
    def test_batch_and_parallel_sampling_alignment(self):
        for n, start_lens, expected_start_lens in (
            (1, 0, [0, 0]),
            (1, 1, [1, 1]),
            (1, [1, 3], [1, 3]),
            (2, 0, [0, 0]),
            (2, 1, [1, 1]),
            (2, [1, 3], [1, 3]),
        ):
            with self.subTest(n=n, start_lens=start_lens):
                request = GenerateReqInput(
                    input_ids=[[10, 11], [20, 21, 22]],
                    mm_token_expansion_start_len=start_lens,
                    sampling_params={"n": n},
                )
                request.normalize_batch_and_arguments()
                self.assertEqual(
                    [request[i].mm_token_expansion_start_len for i in range(2 * n)],
                    expected_start_lens * n,
                )
                self.assertEqual(
                    [request[i].input_ids for i in range(2 * n)],
                    [[10, 11], [20, 21, 22]] * n,
                )

        # One prompt sampled twice shares its scalar start_len.
        request = GenerateReqInput(
            input_ids=[10, 11],
            mm_token_expansion_start_len=1,
            sampling_params={"n": 2},
        )
        request.normalize_batch_and_arguments()
        self.assertEqual(
            [request[i].mm_token_expansion_start_len for i in range(2)], [1, 1]
        )
        self.assertEqual([request[i].input_ids for i in range(2)], [[10, 11], [10, 11]])

    def test_native_rejects_missing_ids_and_misaligned_start_len_lists(self):
        for inputs, start_len in (
            ({"text": "hello"}, 1),
            ({"text": ["hello", "world"]}, [0, 0]),
            ({"input_ids": []}, 1),
            ({"input_ids": [[10], [20, 21]]}, [0]),
            ({"input_ids": [10, 11]}, [1]),
            ({"input_ids": [10, 11], "sampling_params": {"n": 2}}, [1]),
        ):
            with self.subTest(inputs=inputs, start_len=start_len):
                request = GenerateReqInput(
                    **inputs, mm_token_expansion_start_len=start_len
                )
                with self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"):
                    request.normalize_batch_and_arguments()

    def test_chat_requires_input_ids(self):
        base = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        for input_ids, start_len in (
            (None, 1),
            ([], 1),
        ):
            with self.subTest(input_ids=input_ids, start_len=start_len):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest(
                        **base,
                        input_ids=input_ids,
                        mm_token_expansion_start_len=start_len,
                    )


if __name__ == "__main__":
    unittest.main()
