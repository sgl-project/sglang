"""Unit tests for sglang.srt.entrypoints.openai.utils — no server, no model loading."""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import torch

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    CompletionRequest,
    StreamOptions,
)
from sglang.srt.entrypoints.openai.utils import (
    convert_embeds_to_tensors,
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    should_include_usage,
    to_openai_style_logprobs,
)


class TestToOpenaiStyleLogprobs(unittest.TestCase):
    """Tests for to_openai_style_logprobs()."""

    def test_empty_inputs_returns_empty_logprobs(self):
        result = to_openai_style_logprobs()
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(result.top_logprobs, [])

    def test_input_token_logprobs_only(self):
        input_lp = [(-0.5, 42, "hello"), (-1.2, 43, " world")]
        result = to_openai_style_logprobs(input_token_logprobs=input_lp)
        self.assertEqual(result.tokens, ["hello", " world"])
        self.assertEqual(result.token_logprobs, [-0.5, -1.2])
        self.assertEqual(result.text_offset, [-1, -1])

    def test_output_token_logprobs_only(self):
        output_lp = [(-0.3, 10, "foo"), (-0.7, 11, "bar")]
        result = to_openai_style_logprobs(output_token_logprobs=output_lp)
        self.assertEqual(result.tokens, ["foo", "bar"])
        self.assertEqual(result.token_logprobs, [-0.3, -0.7])

    def test_input_and_output_token_logprobs_concatenated(self):
        input_lp = [(-0.1, 1, "a")]
        output_lp = [(-0.2, 2, "b"), (-0.3, 3, "c")]
        result = to_openai_style_logprobs(
            input_token_logprobs=input_lp, output_token_logprobs=output_lp
        )
        self.assertEqual(result.tokens, ["a", "b", "c"])
        self.assertEqual(result.token_logprobs, [-0.1, -0.2, -0.3])

    def test_input_top_logprobs(self):
        top_lp = [
            [(-0.1, 1, "yes"), (-2.0, 2, "no")],
            None,
        ]
        result = to_openai_style_logprobs(input_top_logprobs=top_lp)
        self.assertEqual(len(result.top_logprobs), 2)
        self.assertIn("yes", result.top_logprobs[0])
        self.assertEqual(result.top_logprobs[0]["yes"], -0.1)
        self.assertIsNone(result.top_logprobs[1])

    def test_output_top_logprobs(self):
        top_lp = [[(-0.5, 1, "ok")]]
        result = to_openai_style_logprobs(output_top_logprobs=top_lp)
        self.assertEqual(result.top_logprobs[0]["ok"], -0.5)

    def test_all_params_combined(self):
        input_lp = [(-0.1, 1, "in")]
        output_lp = [(-0.2, 2, "out")]
        in_top = [[(-0.1, 1, "in")]]
        out_top = [[(-0.2, 2, "out")]]
        result = to_openai_style_logprobs(
            input_token_logprobs=input_lp,
            output_token_logprobs=output_lp,
            input_top_logprobs=in_top,
            output_top_logprobs=out_top,
        )
        self.assertEqual(result.tokens, ["in", "out"])
        self.assertEqual(len(result.top_logprobs), 2)


class TestShouldIncludeUsage(unittest.TestCase):
    """Tests for should_include_usage()."""

    def test_no_stream_options_default_false(self):
        include, continuous = should_include_usage(None, False)
        self.assertFalse(include)
        self.assertFalse(continuous)

    def test_no_stream_options_default_true(self):
        include, continuous = should_include_usage(None, True)
        self.assertTrue(include)
        self.assertFalse(continuous)

    def test_stream_options_include_usage_true(self):
        opts = StreamOptions(include_usage=True, continuous_usage_stats=False)
        include, continuous = should_include_usage(opts, False)
        self.assertTrue(include)
        self.assertFalse(continuous)

    def test_stream_options_include_usage_false_default_true(self):
        # stream_options.include_usage=False but default=True → should still include
        opts = StreamOptions(include_usage=False, continuous_usage_stats=False)
        include, continuous = should_include_usage(opts, True)
        self.assertTrue(include)

    def test_stream_options_continuous_usage_stats_true(self):
        opts = StreamOptions(include_usage=True, continuous_usage_stats=True)
        include, continuous = should_include_usage(opts, False)
        self.assertTrue(include)
        self.assertTrue(continuous)

    def test_stream_options_both_false_default_false(self):
        opts = StreamOptions(include_usage=False, continuous_usage_stats=False)
        include, continuous = should_include_usage(opts, False)
        self.assertFalse(include)
        self.assertFalse(continuous)


class TestProcessHiddenStatesFromRet(unittest.TestCase):
    """Tests for process_hidden_states_from_ret()."""

    def _make_request(self, return_hidden_states=False):
        req = MagicMock(spec=CompletionRequest)
        req.return_hidden_states = return_hidden_states
        return req

    def test_returns_none_when_disabled(self):
        ret = {"meta_info": {"hidden_states": [[1.0, 2.0], [3.0, 4.0]]}}
        req = self._make_request(return_hidden_states=False)
        self.assertIsNone(process_hidden_states_from_ret(ret, req))

    def test_returns_none_when_no_hidden_states_key(self):
        ret = {"meta_info": {}}
        req = self._make_request(return_hidden_states=True)
        self.assertIsNone(process_hidden_states_from_ret(ret, req))

    def test_returns_last_element_for_multiple_states(self):
        states = [[1.0], [2.0], [3.0]]
        ret = {"meta_info": {"hidden_states": states}}
        req = self._make_request(return_hidden_states=True)
        result = process_hidden_states_from_ret(ret, req)
        self.assertEqual(result, [3.0])

    def test_returns_empty_list_for_single_state(self):
        # len == 1 → returns []
        ret = {"meta_info": {"hidden_states": [[1.0, 2.0]]}}
        req = self._make_request(return_hidden_states=True)
        result = process_hidden_states_from_ret(ret, req)
        self.assertEqual(result, [])


class TestProcessCachedTokensDetailsFromRet(unittest.TestCase):
    """Tests for process_cached_tokens_details_from_ret()."""

    def _make_request(self, return_cached=False):
        req = MagicMock(spec=CompletionRequest)
        req.return_cached_tokens_details = return_cached
        return req

    def test_returns_none_when_disabled(self):
        ret = {"meta_info": {"cached_tokens_details": {"device": 10, "host": 5}}}
        req = self._make_request(return_cached=False)
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_returns_none_when_no_details_in_meta(self):
        ret = {"meta_info": {}}
        req = self._make_request(return_cached=True)
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_returns_device_host_only(self):
        ret = {"meta_info": {"cached_tokens_details": {"device": 8, "host": 3}}}
        req = self._make_request(return_cached=True)
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 8)
        self.assertEqual(result.host, 3)
        self.assertIsNone(result.storage)

    def test_returns_with_storage_fields(self):
        ret = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 4,
                    "host": 2,
                    "storage": 10,
                    "storage_backend": "redis",
                }
            }
        }
        req = self._make_request(return_cached=True)
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertEqual(result.storage, 10)
        self.assertEqual(result.storage_backend, "redis")


class TestConvertEmbeddsToTensors(unittest.TestCase):
    """Tests for convert_embeds_to_tensors()."""

    def test_none_input(self):
        self.assertIsNone(convert_embeds_to_tensors(None))

    def test_empty_list(self):
        self.assertEqual(convert_embeds_to_tensors([]), [])

    def test_all_none_entries(self):
        result = convert_embeds_to_tensors([None, None])
        self.assertEqual(result, [None, None])

    def test_single_input_list_of_float_vectors(self):
        # Single input: List[List[float]] → [[tensor, ...]]
        embeds = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        self.assertIsInstance(result[0][0], torch.Tensor)
        self.assertTrue(torch.allclose(result[0][0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.allclose(result[0][1], torch.tensor([4.0, 5.0, 6.0])))

    def test_batch_input(self):
        # Batch: List[List[List[float]]] → [[tensor,...], [tensor,...]]
        embeds = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        ]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 1)
        self.assertTrue(torch.allclose(result[0][0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(result[1][0], torch.tensor([5.0, 6.0])))

    def test_batch_with_none_entries(self):
        embeds = [[[1.0, 2.0]], None, [[3.0, 4.0]]]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNotNone(result[2])

    def test_tensor_dtype_is_float32(self):
        embeds = [[1.0, 2.0]]
        result = convert_embeds_to_tensors(embeds)
        self.assertEqual(result[0][0].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
