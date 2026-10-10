"""CPU tests for the caller's multimodal token expansion boundary.

  preserved prefix | tokens to expand   -> boundary in supplied IDs
  omitted boundary       -> stays None, a non-partial request
  explicit boundary      -> require non-empty supplied IDs, even for zero
  boundary list          -> validate batch alignment before repetition
  scalar boundary        -> defer type/range validation to token expansion
  single request         -> scalar boundary
  batch / n > 1          -> broadcast scalar or repeat per-prompt boundaries
  text without IDs       -> omit boundary; tokenize later

The boundary is independent of routed-expert and logprob response offsets.
"""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMMTokenExpansionStartLen(unittest.TestCase):
    def test_native_defaults_and_full_prefix(self):
        for inputs in ({"text": "hello"}, {"input_ids": [10, 11]}):
            for n in (1, 2):
                with self.subTest(inputs=inputs, n=n):
                    request = GenerateReqInput(**inputs, sampling_params={"n": n})
                    self.assertIsNone(request.mm_token_expansion_start_len)
                    request.normalize_batch_and_arguments()
                    self.assertEqual(
                        request.mm_token_expansion_start_len,
                        None if n == 1 else [None, None],
                    )

        request = GenerateReqInput(
            input_ids=[10, 11],
            mm_token_expansion_start_len=2,
            routed_experts_start_len=1,
        )
        request.normalize_batch_and_arguments()
        self.assertEqual(request.mm_token_expansion_start_len, 2)
        self.assertEqual(request.input_ids, [10, 11])
        self.assertEqual(request.routed_experts_start_len, 1)
        self.assertEqual(request.logprob_start_len, -1)

    def test_batch_and_parallel_sampling_alignment(self):
        for n in (1, 2):
            for starts, expected_starts in (
                (None, [None, None]),
                (1, [1, 1]),
                ([1, 3], [1, 3]),
            ):
                with self.subTest(n=n, starts=starts):
                    request = GenerateReqInput(
                        input_ids=[[10, 11], [20, 21, 22]],
                        mm_token_expansion_start_len=starts,
                        sampling_params={"n": n},
                    )
                    request.normalize_batch_and_arguments()
                    self.assertEqual(
                        [request[i].mm_token_expansion_start_len for i in range(2 * n)],
                        expected_starts * n,
                    )
                    self.assertEqual(
                        [request[i].input_ids for i in range(2 * n)],
                        [[10, 11], [20, 21, 22]] * n,
                    )

        for starts in (1, [1]):
            with self.subTest(single_prompt_parallel_boundaries=starts):
                request = GenerateReqInput(
                    input_ids=[10, 11],
                    mm_token_expansion_start_len=starts,
                    sampling_params={"n": 2},
                )
                request.normalize_batch_and_arguments()
                self.assertEqual(
                    [request[i].mm_token_expansion_start_len for i in range(2)], [1, 1]
                )
                self.assertEqual(
                    [request[i].input_ids for i in range(2)], [[10, 11], [10, 11]]
                )

    def test_native_rejects_missing_ids_and_misaligned_boundaries(self):
        for inputs, start in (
            ({"text": "hello"}, 0),
            ({"text": "hello"}, 1),
            ({"text": ["hello", "world"]}, [0, 0]),
            ({"input_ids": []}, 0),
            ({"input_ids": [10, 11]}, [1]),
            ({"input_ids": [[10], [20, 21]]}, [0]),
            ({"input_ids": [[10], [20, 21]]}, [0, 0, 0]),
            (
                {"input_ids": [[10], [20, 21]], "sampling_params": {"n": 2}},
                [0],
            ),
        ):
            with self.subTest(inputs=inputs, start=start):
                request = GenerateReqInput(**inputs, mm_token_expansion_start_len=start)
                with self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"):
                    request.normalize_batch_and_arguments()
                self.assertEqual(request.input_ids, inputs.get("input_ids"))
                self.assertEqual(request.mm_token_expansion_start_len, start)
                self.assertIsNone(request.rid)

    def test_native_defers_scalar_validation_to_expansion(self):
        for start in (-1, 3, 0.5):
            with self.subTest(start=start):
                request = GenerateReqInput(
                    input_ids=[10, 11], mm_token_expansion_start_len=start
                )
                request.normalize_batch_and_arguments()
                self.assertEqual(request.mm_token_expansion_start_len, start)

    def test_chat_requires_ids_and_defers_range_validation_to_expansion(self):
        base = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        self.assertIsNone(ChatCompletionRequest(**base).mm_token_expansion_start_len)
        for start in (0, 2, -1, 3):
            request = ChatCompletionRequest(
                **base, input_ids=[10, 11], mm_token_expansion_start_len=start
            )
            self.assertEqual(request.mm_token_expansion_start_len, start)

        for input_ids, start in (
            (None, 0),
            (None, 1),
            ([], 0),
        ):
            with self.subTest(input_ids=input_ids, start=start):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest(
                        **base, input_ids=input_ids, mm_token_expansion_start_len=start
                    )


if __name__ == "__main__":
    unittest.main()
