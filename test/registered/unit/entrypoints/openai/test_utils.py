from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest

import torch

from sglang.srt.entrypoints.openai.protocol import CompletionRequest, StreamOptions
from sglang.srt.entrypoints.openai.utils import (
    cached_tokens_details_from_dict,
    convert_embeds_to_tensors,
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    should_include_usage,
    to_openai_style_logprobs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class OpenAIUtilsTestCase(unittest.TestCase):
    def test_to_openai_style_logprobs_empty_inputs(self):
        logprobs = to_openai_style_logprobs()

        self.assertEqual(logprobs.tokens, [])
        self.assertEqual(logprobs.token_logprobs, [])
        self.assertEqual(logprobs.text_offset, [])
        self.assertEqual(logprobs.top_logprobs, [])

    def test_to_openai_style_logprobs_combines_input_and_output(self):
        logprobs = to_openai_style_logprobs(
            input_token_logprobs=[(-0.1, 1, "Hello")],
            output_token_logprobs=[(-0.2, 2, " world")],
            input_top_logprobs=[
                [(-0.1, 1, "Hello"), (-1.0, 3, "Hi")],
                None,
            ],
            output_top_logprobs=[
                [(-0.2, 2, " world")],
            ],
        )

        self.assertEqual(logprobs.tokens, ["Hello", " world"])
        self.assertEqual(logprobs.token_logprobs, [-0.1, -0.2])
        self.assertEqual(logprobs.text_offset, [-1, -1])
        self.assertEqual(
            logprobs.top_logprobs,
            [
                {"Hello": -0.1, "Hi": -1.0},
                None,
                {" world": -0.2},
            ],
        )

    def test_process_hidden_states_respects_request_flag(self):
        ret_item = {"meta_info": {"hidden_states": [[1.0], [2.0]]}}
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=False,
        )

        self.assertIsNone(process_hidden_states_from_ret(ret_item, request))

    def test_process_hidden_states_returns_last_state_or_empty_list(self):
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
        )

        self.assertEqual(
            process_hidden_states_from_ret(
                {"meta_info": {"hidden_states": [[1.0], [2.0]]}}, request
            ),
            [2.0],
        )
        self.assertEqual(
            process_hidden_states_from_ret(
                {"meta_info": {"hidden_states": [[1.0]]}}, request
            ),
            [],
        )
        self.assertIsNone(process_hidden_states_from_ret({"meta_info": {}}, request))

    def test_process_routed_experts_respects_request_flag(self):
        ret_item = {"meta_info": {"routed_experts": "expert-1"}}

        disabled_request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_routed_experts=False,
        )
        self.assertIsNone(process_routed_experts_from_ret(ret_item, disabled_request))

        enabled_request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_routed_experts=True,
        )
        self.assertEqual(
            process_routed_experts_from_ret(ret_item, enabled_request), "expert-1"
        )
        self.assertIsNone(
            process_routed_experts_from_ret({"meta_info": {}}, enabled_request)
        )

    def test_should_include_usage_defaults_without_stream_options(self):
        self.assertEqual(should_include_usage(None, False), (False, False))
        self.assertEqual(should_include_usage(None, True), (True, False))

    def test_should_include_usage_with_stream_options(self):
        self.assertEqual(
            should_include_usage(
                StreamOptions(include_usage=True, continuous_usage_stats=False), False
            ),
            (True, False),
        )
        self.assertEqual(
            should_include_usage(
                StreamOptions(include_usage=False, continuous_usage_stats=True), False
            ),
            (False, True),
        )
        self.assertEqual(
            should_include_usage(
                StreamOptions(include_usage=False, continuous_usage_stats=False), True
            ),
            (True, False),
        )

    def test_cached_tokens_details_from_dict_without_storage(self):
        details = cached_tokens_details_from_dict({"device": 3})

        self.assertEqual(details.device, 3)
        self.assertEqual(details.host, 0)
        self.assertIsNone(details.storage)
        self.assertIsNone(details.storage_backend)
        self.assertEqual(
            details.model_dump(exclude_none=True), {"device": 3, "host": 0}
        )

    def test_cached_tokens_details_from_dict_with_storage(self):
        details = cached_tokens_details_from_dict(
            {
                "device": 3,
                "host": 2,
                "storage": 1,
                "storage_backend": "file",
            }
        )

        self.assertEqual(details.device, 3)
        self.assertEqual(details.host, 2)
        self.assertEqual(details.storage, 1)
        self.assertEqual(details.storage_backend, "file")
        self.assertEqual(
            details.model_dump(exclude_none=True),
            {
                "device": 3,
                "host": 2,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def test_process_cached_tokens_details_from_ret(self):
        ret_item = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 4,
                    "host": 1,
                    "storage": 2,
                    "storage_backend": "disk",
                }
            }
        }
        disabled_request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_cached_tokens_details=False,
        )
        enabled_request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_cached_tokens_details=True,
        )

        self.assertIsNone(
            process_cached_tokens_details_from_ret(ret_item, disabled_request)
        )
        self.assertIsNone(
            process_cached_tokens_details_from_ret({"meta_info": {}}, enabled_request)
        )
        self.assertEqual(
            process_cached_tokens_details_from_ret(
                ret_item, enabled_request
            ).model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 2,
                "storage_backend": "disk",
            },
        )

    def test_convert_embeds_to_tensors_handles_none_and_empty_inputs(self):
        self.assertIsNone(convert_embeds_to_tensors(None))
        self.assertEqual(convert_embeds_to_tensors([]), [])
        self.assertEqual(convert_embeds_to_tensors([None, None]), [None, None])

    def test_convert_embeds_to_tensors_single_input(self):
        converted = convert_embeds_to_tensors([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(len(converted), 1)
        self.assertEqual(len(converted[0]), 2)
        self.assertTrue(torch.equal(converted[0][0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(converted[0][1], torch.tensor([3.0, 4.0])))
        self.assertEqual(converted[0][0].dtype, torch.float32)

    def test_convert_embeds_to_tensors_batch_input(self):
        converted = convert_embeds_to_tensors(
            [
                [[1.0, 2.0]],
                None,
                [[3.0, 4.0], [5.0, 6.0]],
            ]
        )

        self.assertEqual(len(converted), 3)
        self.assertTrue(torch.equal(converted[0][0], torch.tensor([1.0, 2.0])))
        self.assertIsNone(converted[1])
        self.assertTrue(torch.equal(converted[2][0], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(converted[2][1], torch.tensor([5.0, 6.0])))
        self.assertEqual(converted[2][0].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
