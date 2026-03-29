"""Unit tests for entrypoints/openai/utils.py — no server, no model loading."""

from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    to_openai_style_logprobs,
)
from sglang.test.test_utils import CustomTestCase


class TestToOpenaiStyleLogprobs(CustomTestCase):

    def test_all_none_returns_empty_logprobs(self):
        result = to_openai_style_logprobs()
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.token_logprobs, [])

    def test_input_logprobs(self):
        result = to_openai_style_logprobs(
            input_token_logprobs=[(-0.5, 1, "hello"), (-1.0, 2, "world")]
        )
        self.assertEqual(result.tokens, ["hello", "world"])
        self.assertEqual(result.token_logprobs, [-0.5, -1.0])
        # text_offset is currently hardcoded to -1
        self.assertEqual(result.text_offset, [-1, -1])

    def test_input_and_output_concatenated(self):
        result = to_openai_style_logprobs(
            input_token_logprobs=[(-0.5, 1, "a")],
            output_token_logprobs=[(-0.3, 2, "b")],
        )
        self.assertEqual(result.tokens, ["a", "b"])
        self.assertEqual(result.token_logprobs, [-0.5, -0.3])

    def test_top_logprobs_converted_to_dict(self):
        """Each token list becomes {text: logprob} dict."""
        top = [[(-0.1, 10, "x"), (-0.5, 11, "y")]]
        result = to_openai_style_logprobs(output_top_logprobs=top)
        self.assertEqual(result.top_logprobs[0], {"x": -0.1, "y": -0.5})

    def test_none_entry_in_top_logprobs(self):
        """A None in the top_logprobs list should be preserved as None."""
        result = to_openai_style_logprobs(output_top_logprobs=[None, [(-0.2, 5, "z")]])
        self.assertIsNone(result.top_logprobs[0])
        self.assertEqual(result.top_logprobs[1], {"z": -0.2})


class TestProcessHiddenStates(CustomTestCase):

    def _make_request(self, return_hidden_states=False):
        req = Mock()
        req.return_hidden_states = return_hidden_states
        return req

    def test_flag_disabled_skips_extraction(self):
        ret = {"meta_info": {"hidden_states": [[1, 2], [3, 4]]}}
        self.assertIsNone(
            process_hidden_states_from_ret(ret, self._make_request(False))
        )

    def test_missing_key_returns_none(self):
        ret = {"meta_info": {}}
        self.assertIsNone(process_hidden_states_from_ret(ret, self._make_request(True)))

    def test_multi_element_returns_last(self):
        """Source code does hidden_states[-1] when len > 1."""
        ret = {"meta_info": {"hidden_states": [[1, 2], [3, 4], [5, 6]]}}
        result = process_hidden_states_from_ret(ret, self._make_request(True))
        self.assertEqual(result, [5, 6])

    def test_single_element_returns_empty_list(self):
        """Source code returns [] when len(hidden_states) == 1."""
        ret = {"meta_info": {"hidden_states": [[1, 2]]}}
        result = process_hidden_states_from_ret(ret, self._make_request(True))
        self.assertEqual(result, [])


class TestProcessRoutedExperts(CustomTestCase):

    def test_no_attribute_returns_none(self):
        req = Mock(spec=[])
        ret = {"meta_info": {"routed_experts": "data"}}
        self.assertIsNone(process_routed_experts_from_ret(ret, req))

    def test_flag_enabled_extracts_value(self):
        req = Mock()
        req.return_routed_experts = True
        ret = {"meta_info": {"routed_experts": "expert_data"}}
        self.assertEqual(process_routed_experts_from_ret(ret, req), "expert_data")

    def test_flag_enabled_missing_key(self):
        req = Mock()
        req.return_routed_experts = True
        ret = {"meta_info": {}}
        self.assertIsNone(process_routed_experts_from_ret(ret, req))


class TestProcessCachedTokensDetails(CustomTestCase):

    def test_no_attribute_returns_none(self):
        req = Mock(spec=[])
        ret = {"meta_info": {"cached_tokens_details": {"device": 5}}}
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_without_storage_field(self):
        req = Mock()
        req.return_cached_tokens_details = True
        ret = {"meta_info": {"cached_tokens_details": {"device": 10, "host": 5}}}
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertEqual(result.device, 10)
        self.assertEqual(result.host, 5)
        self.assertIsNone(result.storage)

    def test_with_storage_field(self):
        """When 'storage' key is present, L3 cache fields are populated."""
        req = Mock()
        req.return_cached_tokens_details = True
        ret = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 10,
                    "host": 5,
                    "storage": 3,
                    "storage_backend": "disk",
                }
            }
        }
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertEqual(result.storage, 3)
        self.assertEqual(result.storage_backend, "disk")

    def test_none_details_returns_none(self):
        req = Mock()
        req.return_cached_tokens_details = True
        ret = {"meta_info": {"cached_tokens_details": None}}
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_missing_key_returns_none(self):
        req = Mock()
        req.return_cached_tokens_details = True
        ret = {"meta_info": {}}
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))


if __name__ == "__main__":
    unittest.main()
