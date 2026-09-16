import unittest
from copy import deepcopy

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class CompletionLogprobsTest(unittest.TestCase):
    def test_token_ids_preserve_distinct_tokens_with_identical_text(self):
        result = to_openai_style_logprobs(
            input_token_logprobs=[(None, 1, "\ufffd")],
            input_top_logprobs=[None],
            output_token_logprobs=[(-0.1, 2, "\ufffd")],
            output_top_logprobs=[[(-0.1, 2, "\ufffd"), (-0.2, 3, "\ufffd")]],
            return_tokens_as_token_ids=True,
        )
        self.assertEqual(result.tokens, ["token_id:1", "token_id:2"])
        self.assertEqual(result.token_logprobs, [None, -0.1])
        self.assertEqual(
            result.top_logprobs, [None, {"token_id:2": -0.1, "token_id:3": -0.2}]
        )
        self.assertEqual(result.text_offset, [-1, -1])

    def test_default_and_false_keep_text(self):
        for options in ({}, {"return_tokens_as_token_ids": False}):
            with self.subTest(options=options):
                result = to_openai_style_logprobs(
                    output_token_logprobs=[(-0.1, 7, " hello")],
                    output_top_logprobs=[[(-0.1, 7, " hello")]],
                    **options,
                )
                self.assertEqual(result.tokens, [" hello"])
                self.assertEqual(result.top_logprobs, [{" hello": -0.1}])

    def test_empty_and_missing_top_logprobs(self):
        result = to_openai_style_logprobs(return_tokens_as_token_ids=True)
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.top_logprobs, [])
        result = to_openai_style_logprobs(
            output_token_logprobs=[(-0.1, 0, None)],
            output_top_logprobs=[[]],
            return_tokens_as_token_ids=True,
        )
        self.assertEqual(result.tokens, ["token_id:0"])
        self.assertEqual(result.top_logprobs, [{}])

    def test_request_accepts_token_prompts_and_independent_output_options(self):
        for prompt in ([1, 2], [[1, 2], [3]]):
            for option in (None, False, True):
                with self.subTest(prompt=prompt, option=option):
                    request = CompletionRequest(
                        model="test",
                        prompt=prompt,
                        logprobs=2,
                        return_token_ids=True,
                        return_tokens_as_token_ids=option,
                    )
                    self.assertEqual(request.prompt, prompt)
                    self.assertIs(request.return_tokens_as_token_ids, option)
                    self.assertTrue(request.return_token_ids)

    def test_input_only_output_only_and_missing_top_rows(self):
        for side in ("input", "output"):
            with self.subTest(side=side):
                result = to_openai_style_logprobs(
                    **{
                        f"{side}_token_logprobs": [(None, 0, None), (-0.25, 42, "")],
                        f"{side}_top_logprobs": [None, []],
                    },
                    return_tokens_as_token_ids=True,
                )
                self.assertEqual(result.tokens, ["token_id:0", "token_id:42"])
                self.assertEqual(result.token_logprobs, [None, -0.25])
                self.assertEqual(result.top_logprobs, [None, {}])
                self.assertEqual(result.text_offset, [-1, -1])

    def test_conversion_does_not_mutate_backend_data_or_share_results(self):
        data = {
            "input_token_logprobs": [(None, 1, "a")],
            "input_top_logprobs": [None],
            "output_token_logprobs": [(-0.1, 2, "b")],
            "output_top_logprobs": [[(-0.1, 2, "b"), (-0.2, 3, "b")]],
        }
        original = deepcopy(data)
        encoded = to_openai_style_logprobs(**data, return_tokens_as_token_ids=True)
        text = to_openai_style_logprobs(**data)
        encoded.tokens.append("sentinel")
        encoded.top_logprobs[-1]["sentinel"] = 0.0
        self.assertEqual(data, original)
        self.assertEqual(text.tokens, ["a", "b"])
        self.assertEqual(text.top_logprobs, [None, {"b": -0.2}])
        self.assertEqual(to_openai_style_logprobs().tokens, [])

    def test_request_defaults_and_flag_round_trip(self):
        defaults = CompletionRequest(model="test", prompt=[0, 42])
        self.assertIsNone(defaults.return_tokens_as_token_ids)
        self.assertFalse(defaults.return_token_ids)
        self.assertIsNone(defaults.logprobs)
        for option in (None, False, True):
            with self.subTest(option=option):
                request = defaults.model_copy(
                    update={"return_tokens_as_token_ids": option}
                )
                restored = CompletionRequest.model_validate_json(
                    request.model_dump_json()
                )
                self.assertIs(restored.return_tokens_as_token_ids, option)
                self.assertEqual(restored.prompt, [0, 42])
                self.assertIsNone(restored.logprobs)

    def test_request_rejects_invalid_logprob_token_flag(self):
        for option in ([], {}, "not-a-boolean"):
            with self.subTest(option=option), self.assertRaises(ValidationError):
                CompletionRequest(prompt=[1], return_tokens_as_token_ids=option)


if __name__ == "__main__":
    unittest.main()
