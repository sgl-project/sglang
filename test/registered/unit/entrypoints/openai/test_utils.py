"""Unit tests for srt/entrypoints/openai/utils.py"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    StreamOptions,
)
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
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestToOpenAIStyleLogprobs(CustomTestCase):
    def test_empty_logprobs(self):
        """Test with no logprobs provided."""
        result = to_openai_style_logprobs()
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(result.text_offset, [])
        self.assertEqual(result.top_logprobs, [])

    def test_input_token_logprobs_only(self):
        """Test with only input token logprobs."""
        input_logprobs = [
            (-0.5, None, "hello"),
            (-1.2, None, "world"),
        ]
        result = to_openai_style_logprobs(input_token_logprobs=input_logprobs)
        self.assertEqual(result.tokens, ["hello", "world"])
        self.assertEqual(result.token_logprobs, [-0.5, -1.2])
        self.assertEqual(result.text_offset, [-1, -1])
        self.assertEqual(result.top_logprobs, [])

    def test_output_token_logprobs_only(self):
        """Test with only output token logprobs."""
        output_logprobs = [
            (-0.3, None, "foo"),
            (-0.8, None, "bar"),
        ]
        result = to_openai_style_logprobs(output_token_logprobs=output_logprobs)
        self.assertEqual(result.tokens, ["foo", "bar"])
        self.assertEqual(result.token_logprobs, [-0.3, -0.8])

    def test_input_and_output_logprobs(self):
        """Test with both input and output logprobs."""
        input_logprobs = [(-0.5, None, "hello")]
        output_logprobs = [(-0.3, None, "world")]
        result = to_openai_style_logprobs(
            input_token_logprobs=input_logprobs, output_token_logprobs=output_logprobs
        )
        self.assertEqual(result.tokens, ["hello", "world"])
        self.assertEqual(result.token_logprobs, [-0.5, -0.3])

    def test_top_logprobs(self):
        """Test with top logprobs."""
        top_logprobs = [
            [(-0.1, None, "a"), (-0.5, None, "b")],
            [(-0.2, None, "x"), (-0.6, None, "y")],
        ]
        result = to_openai_style_logprobs(output_top_logprobs=top_logprobs)
        self.assertEqual(len(result.top_logprobs), 2)
        self.assertEqual(result.top_logprobs[0], {"a": -0.1, "b": -0.5})
        self.assertEqual(result.top_logprobs[1], {"x": -0.2, "y": -0.6})

    def test_top_logprobs_with_none(self):
        """Test top logprobs with None entries."""
        top_logprobs = [
            [(-0.1, None, "a")],
            None,
            [(-0.2, None, "x")],
        ]
        result = to_openai_style_logprobs(output_top_logprobs=top_logprobs)
        self.assertEqual(len(result.top_logprobs), 3)
        self.assertEqual(result.top_logprobs[0], {"a": -0.1})
        self.assertIsNone(result.top_logprobs[1])
        self.assertEqual(result.top_logprobs[2], {"x": -0.2})

    def test_combined_input_output_top_logprobs(self):
        """Test with all types of logprobs."""
        input_logprobs = [(-0.5, None, "hello")]
        output_logprobs = [(-0.3, None, "world")]
        input_top = [[(-0.1, None, "hi"), (-0.2, None, "hey")]]
        output_top = [[(-0.15, None, "earth"), (-0.25, None, "globe")]]

        result = to_openai_style_logprobs(
            input_token_logprobs=input_logprobs,
            output_token_logprobs=output_logprobs,
            input_top_logprobs=input_top,
            output_top_logprobs=output_top,
        )
        self.assertEqual(result.tokens, ["hello", "world"])
        self.assertEqual(result.token_logprobs, [-0.5, -0.3])
        self.assertEqual(len(result.top_logprobs), 2)


class TestProcessHiddenStatesFromRet(CustomTestCase):
    def test_return_hidden_states_true_multiple(self):
        """Test when return_hidden_states is True with multiple states."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_hidden_states = True
        ret_item = {"meta_info": {"hidden_states": [[1, 2, 3], [4, 5, 6]]}}

        result = process_hidden_states_from_ret(ret_item, request)
        self.assertEqual(result, [4, 5, 6])

    def test_return_hidden_states_true_single(self):
        """Test when only one hidden state is present."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_hidden_states = True
        ret_item = {"meta_info": {"hidden_states": [[1, 2, 3]]}}

        result = process_hidden_states_from_ret(ret_item, request)
        self.assertEqual(result, [])

    def test_return_hidden_states_none(self):
        """Test when hidden_states is None in meta_info."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_hidden_states = True
        ret_item = {"meta_info": {"hidden_states": None}}

        result = process_hidden_states_from_ret(ret_item, request)
        self.assertIsNone(result)

    def test_return_hidden_states_missing(self):
        """Test when hidden_states key is missing."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_hidden_states = True
        ret_item = {"meta_info": {}}

        result = process_hidden_states_from_ret(ret_item, request)
        self.assertIsNone(result)

    def test_return_hidden_states_empty_list(self):
        """Test when hidden_states is an empty list."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_hidden_states = True
        ret_item = {"meta_info": {"hidden_states": []}}

        result = process_hidden_states_from_ret(ret_item, request)
        self.assertEqual(result, [])


class TestShouldIncludeUsage(CustomTestCase):
    def test_no_stream_options_default_false(self):
        """Test with no stream_options and default False."""
        include, continuous = should_include_usage(None, False)
        self.assertFalse(include)
        self.assertFalse(continuous)

    def test_no_stream_options_default_true(self):
        """Test with no stream_options and default True."""
        include, continuous = should_include_usage(None, True)
        self.assertTrue(include)
        self.assertFalse(continuous)

    def test_stream_options_include_usage_true(self):
        """Test with stream_options.include_usage=True."""
        stream_options = StreamOptions(include_usage=True, continuous_usage_stats=False)
        include, continuous = should_include_usage(stream_options, False)
        self.assertTrue(include)
        self.assertFalse(continuous)

    def test_stream_options_continuous_usage_true(self):
        """Test with stream_options.continuous_usage_stats=True."""
        stream_options = StreamOptions(include_usage=False, continuous_usage_stats=True)
        include, continuous = should_include_usage(stream_options, False)
        self.assertFalse(include)
        self.assertTrue(continuous)

    def test_stream_options_both_true(self):
        """Test with both include_usage and continuous_usage_stats True."""
        stream_options = StreamOptions(include_usage=True, continuous_usage_stats=True)
        include, continuous = should_include_usage(stream_options, False)
        self.assertTrue(include)
        self.assertTrue(continuous)

    def test_stream_options_overrides_default(self):
        """Test stream_options interaction with server default."""
        stream_options = StreamOptions(
            include_usage=False, continuous_usage_stats=False
        )
        include, continuous = should_include_usage(stream_options, True)
        # OR logic: include_usage or default = False or True = True
        self.assertTrue(include)
        self.assertFalse(continuous)


class TestProcessRoutedExpertsFromRet(CustomTestCase):
    def test_return_routed_experts_true(self):
        """Test when return_routed_experts is True."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_routed_experts = True
        ret_item = {"meta_info": {"routed_experts": "expert_data"}}

        result = process_routed_experts_from_ret(ret_item, request)
        self.assertEqual(result, "expert_data")

    def test_return_routed_experts_none(self):
        """Test when routed_experts is None."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_routed_experts = True
        ret_item = {"meta_info": {"routed_experts": None}}

        result = process_routed_experts_from_ret(ret_item, request)
        self.assertIsNone(result)


class TestCachedTokensDetailsFromDict(CustomTestCase):
    def test_with_storage_backend(self):
        """Test conversion with storage backend fields."""
        details_dict = {
            "device": 100,
            "host": 50,
            "storage": 25,
            "storage_backend": "s3",
        }
        result = cached_tokens_details_from_dict(details_dict)
        self.assertEqual(result.device, 100)
        self.assertEqual(result.host, 50)
        self.assertEqual(result.storage, 25)
        self.assertEqual(result.storage_backend, "s3")

    def test_without_storage_backend(self):
        """Test conversion without storage backend fields."""
        details_dict = {"device": 100, "host": 50}
        result = cached_tokens_details_from_dict(details_dict)
        self.assertEqual(result.device, 100)
        self.assertEqual(result.host, 50)
        self.assertIsNone(result.storage)
        self.assertIsNone(result.storage_backend)

    def test_partial_storage_fields(self):
        """Test with only storage field, no backend."""
        details_dict = {"device": 100, "host": 50, "storage": 25}
        result = cached_tokens_details_from_dict(details_dict)
        self.assertEqual(result.device, 100)
        self.assertEqual(result.host, 50)
        self.assertEqual(result.storage, 25)
        self.assertIsNone(result.storage_backend)

    def test_defaults_to_zero(self):
        """Test that missing fields default to 0."""
        details_dict = {}
        result = cached_tokens_details_from_dict(details_dict)
        self.assertEqual(result.device, 0)
        self.assertEqual(result.host, 0)
        self.assertIsNone(result.storage)


class TestProcessCachedTokensDetailsFromRet(CustomTestCase):
    def test_return_cached_tokens_details_true(self):
        """Test when return_cached_tokens_details is True."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_cached_tokens_details = True
        ret_item = {"meta_info": {"cached_tokens_details": {"device": 10, "host": 5}}}

        result = process_cached_tokens_details_from_ret(ret_item, request)
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 10)
        self.assertEqual(result.host, 5)

    def test_cached_tokens_details_none(self):
        """Test when cached_tokens_details is None."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_cached_tokens_details = True
        ret_item = {"meta_info": {"cached_tokens_details": None}}

        result = process_cached_tokens_details_from_ret(ret_item, request)
        self.assertIsNone(result)

    def test_cached_tokens_details_missing(self):
        """Test when cached_tokens_details key is missing."""
        request = MagicMock(spec=ChatCompletionRequest)
        request.return_cached_tokens_details = True
        ret_item = {"meta_info": {}}

        result = process_cached_tokens_details_from_ret(ret_item, request)
        self.assertIsNone(result)


class TestConvertEmbedsToTensors(CustomTestCase):
    def test_none_input(self):
        """Test with None input."""
        result = convert_embeds_to_tensors(None)
        self.assertIsNone(result)

    def test_empty_list(self):
        """Test with empty list."""
        result = convert_embeds_to_tensors([])
        self.assertEqual(result, [])

    def test_all_none_entries(self):
        """Test with all None entries."""
        result = convert_embeds_to_tensors([None, None, None])
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_single_input_format(self):
        """Test single input format: [num_replacements][hidden_size]."""
        embeds = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        self.assertIsInstance(result[0][0], torch.Tensor)
        self.assertTrue(torch.equal(result[0][0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(result[0][1], torch.tensor([4.0, 5.0, 6.0])))

    def test_batch_format(self):
        """Test batch format: [num_inputs][num_replacements][hidden_size]."""
        embeds = [
            [[1.0, 2.0], [3.0, 4.0]],  # input 0
            [[5.0, 6.0], [7.0, 8.0]],  # input 1
        ]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(torch.equal(result[0][0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[1][1], torch.tensor([7.0, 8.0])))

    def test_batch_with_none_entries(self):
        """Test batch format with None entries."""
        embeds = [
            [[1.0, 2.0]],
            None,
            [[3.0, 4.0]],
        ]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNotNone(result[2])
        self.assertTrue(torch.equal(result[0][0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[2][0], torch.tensor([3.0, 4.0])))

    def test_tensor_dtype(self):
        """Test that tensors are created with float32 dtype."""
        embeds = [[1.0, 2.0]]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(result[0][0].dtype, torch.float32)

    def test_batch_first_entry_none(self):
        """Test batch format where first entry is None (tests first_non_none logic)."""
        embeds = [
            None,
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        ]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(torch.equal(result[1][0], torch.tensor([1.0, 2.0])))
        self.assertEqual(len(result[2]), 1)


if __name__ == "__main__":
    unittest.main()
