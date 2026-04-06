"""Unit tests for srt/entrypoints/openai/utils.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.protocol import CachedTokensDetails, LogProbs
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    to_openai_style_logprobs,
)
from sglang.test.test_utils import CustomTestCase


class TestToOpenaiStyleLogprobs(CustomTestCase):
    """Tests for to_openai_style_logprobs()."""

    def test_all_none_returns_empty_logprobs(self):
        """All arguments None should return an empty LogProbs."""
        result = to_openai_style_logprobs()
        self.assertIsInstance(result, LogProbs)
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(result.text_offset, [])
        self.assertEqual(result.top_logprobs, [])

    def test_input_token_logprobs_only(self):
        """Input token logprobs are appended correctly."""
        input_token_logprobs = [
            (-0.5, 101, "hello"),
            (-1.2, 102, " world"),
        ]
        result = to_openai_style_logprobs(input_token_logprobs=input_token_logprobs)

        self.assertEqual(result.tokens, ["hello", " world"])
        self.assertAlmostEqual(result.token_logprobs[0], -0.5)
        self.assertAlmostEqual(result.token_logprobs[1], -1.2)
        self.assertEqual(result.text_offset, [-1, -1])

    def test_output_token_logprobs_only(self):
        """Output token logprobs are appended correctly."""
        output_token_logprobs = [
            (-0.3, 200, "foo"),
        ]
        result = to_openai_style_logprobs(output_token_logprobs=output_token_logprobs)

        self.assertEqual(result.tokens, ["foo"])
        self.assertAlmostEqual(result.token_logprobs[0], -0.3)

    def test_input_and_output_token_logprobs_concatenated(self):
        """Both input and output token logprobs are concatenated in order."""
        input_token_logprobs = [(-0.1, 1, "a")]
        output_token_logprobs = [(-0.2, 2, "b"), (-0.3, 3, "c")]
        result = to_openai_style_logprobs(
            input_token_logprobs=input_token_logprobs,
            output_token_logprobs=output_token_logprobs,
        )

        self.assertEqual(result.tokens, ["a", "b", "c"])
        self.assertEqual(len(result.token_logprobs), 3)

    def test_input_top_logprobs(self):
        """Input top logprobs are converted to {text: logprob} dicts."""
        input_top_logprobs = [
            [(-0.1, 10, "x"), (-0.5, 11, "y")],
            [(-0.2, 20, "z")],
        ]
        result = to_openai_style_logprobs(input_top_logprobs=input_top_logprobs)

        self.assertEqual(len(result.top_logprobs), 2)
        self.assertAlmostEqual(result.top_logprobs[0]["x"], -0.1)
        self.assertAlmostEqual(result.top_logprobs[0]["y"], -0.5)
        self.assertAlmostEqual(result.top_logprobs[1]["z"], -0.2)

    def test_output_top_logprobs(self):
        """Output top logprobs are converted similarly."""
        output_top_logprobs = [
            [(-0.4, 30, "w")],
        ]
        result = to_openai_style_logprobs(output_top_logprobs=output_top_logprobs)

        self.assertEqual(len(result.top_logprobs), 1)
        self.assertAlmostEqual(result.top_logprobs[0]["w"], -0.4)

    def test_top_logprobs_none_entry(self):
        """None entries in top_logprobs are preserved as None."""
        output_top_logprobs = [
            None,
            [(-0.1, 1, "a")],
        ]
        result = to_openai_style_logprobs(output_top_logprobs=output_top_logprobs)

        self.assertEqual(len(result.top_logprobs), 2)
        self.assertIsNone(result.top_logprobs[0])
        self.assertIsNotNone(result.top_logprobs[1])

    def test_input_and_output_top_logprobs_concatenated(self):
        """Both input and output top logprobs are concatenated."""
        input_top_logprobs = [[(-0.1, 1, "a")]]
        output_top_logprobs = [[(-0.2, 2, "b")]]
        result = to_openai_style_logprobs(
            input_top_logprobs=input_top_logprobs,
            output_top_logprobs=output_top_logprobs,
        )

        self.assertEqual(len(result.top_logprobs), 2)

    def test_all_arguments_provided(self):
        """All four arguments provided together."""
        input_tok = [(-0.1, 1, "in")]
        output_tok = [(-0.2, 2, "out")]
        input_top = [[(-0.3, 3, "top_in")]]
        output_top = [[(-0.4, 4, "top_out")]]

        result = to_openai_style_logprobs(
            input_token_logprobs=input_tok,
            output_token_logprobs=output_tok,
            input_top_logprobs=input_top,
            output_top_logprobs=output_top,
        )

        self.assertEqual(result.tokens, ["in", "out"])
        self.assertEqual(len(result.top_logprobs), 2)

    def test_empty_lists(self):
        """Empty lists produce empty LogProbs fields."""
        result = to_openai_style_logprobs(
            input_token_logprobs=[],
            output_token_logprobs=[],
            input_top_logprobs=[],
            output_top_logprobs=[],
        )
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.top_logprobs, [])


class TestProcessHiddenStatesFromRet(CustomTestCase):
    """Tests for process_hidden_states_from_ret()."""

    def _make_request(self, return_hidden_states=False):
        req = MagicMock()
        req.return_hidden_states = return_hidden_states
        return req

    def test_disabled_returns_none(self):
        """When return_hidden_states is False, always returns None."""
        ret_item = {"meta_info": {"hidden_states": [[0.1, 0.2]]}}
        result = process_hidden_states_from_ret(
            ret_item, self._make_request(return_hidden_states=False)
        )
        self.assertIsNone(result)

    def test_no_hidden_states_in_meta_returns_none(self):
        """When meta_info has no hidden_states key, returns None."""
        ret_item = {"meta_info": {}}
        result = process_hidden_states_from_ret(
            ret_item, self._make_request(return_hidden_states=True)
        )
        self.assertIsNone(result)

    def test_multiple_hidden_states_returns_last(self):
        """With multiple hidden states, returns the last one."""
        hidden = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        ret_item = {"meta_info": {"hidden_states": hidden}}
        result = process_hidden_states_from_ret(
            ret_item, self._make_request(return_hidden_states=True)
        )
        self.assertEqual(result, [0.5, 0.6])

    def test_single_hidden_state_returns_empty_list(self):
        """With exactly one hidden state (len == 1), returns empty list."""
        hidden = [[0.1, 0.2]]
        ret_item = {"meta_info": {"hidden_states": hidden}}
        result = process_hidden_states_from_ret(
            ret_item, self._make_request(return_hidden_states=True)
        )
        self.assertEqual(result, [])

    def test_two_hidden_states_returns_last(self):
        """With two hidden states (len > 1), returns hidden_states[-1]."""
        hidden = [[0.1], [0.9]]
        ret_item = {"meta_info": {"hidden_states": hidden}}
        result = process_hidden_states_from_ret(
            ret_item, self._make_request(return_hidden_states=True)
        )
        self.assertEqual(result, [0.9])


class TestProcessRoutedExpertsFromRet(CustomTestCase):
    """Tests for process_routed_experts_from_ret()."""

    def _make_request(self, return_routed_experts=False):
        req = MagicMock()
        req.return_routed_experts = return_routed_experts
        return req

    def test_disabled_returns_none(self):
        """When return_routed_experts is False, returns None."""
        ret_item = {"meta_info": {"routed_experts": "some_data"}}
        result = process_routed_experts_from_ret(
            ret_item, self._make_request(return_routed_experts=False)
        )
        self.assertIsNone(result)

    def test_enabled_returns_data(self):
        """When enabled, returns the routed_experts from meta_info."""
        expected = "expert_routing_info"
        ret_item = {"meta_info": {"routed_experts": expected}}
        result = process_routed_experts_from_ret(
            ret_item, self._make_request(return_routed_experts=True)
        )
        self.assertEqual(result, expected)

    def test_enabled_missing_key_returns_none(self):
        """When enabled but key missing from meta_info, returns None."""
        ret_item = {"meta_info": {}}
        result = process_routed_experts_from_ret(
            ret_item, self._make_request(return_routed_experts=True)
        )
        self.assertIsNone(result)

    def test_attribute_not_present_returns_none(self):
        """When request has no return_routed_experts attribute, returns None."""
        req = object()  # plain object with no attributes
        ret_item = {"meta_info": {"routed_experts": "data"}}
        result = process_routed_experts_from_ret(ret_item, req)
        self.assertIsNone(result)


class TestProcessCachedTokensDetailsFromRet(CustomTestCase):
    """Tests for process_cached_tokens_details_from_ret()."""

    def _make_request(self, return_cached_tokens_details=False):
        req = MagicMock()
        req.return_cached_tokens_details = return_cached_tokens_details
        return req

    def test_disabled_returns_none(self):
        """When return_cached_tokens_details is False, returns None."""
        ret_item = {"meta_info": {"cached_tokens_details": {"device": 10, "host": 5}}}
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=False)
        )
        self.assertIsNone(result)

    def test_no_details_in_meta_returns_none(self):
        """When meta_info has no cached_tokens_details, returns None."""
        ret_item = {"meta_info": {}}
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=True)
        )
        self.assertIsNone(result)

    def test_device_and_host_only(self):
        """Without storage field, returns CachedTokensDetails with device+host."""
        ret_item = {"meta_info": {"cached_tokens_details": {"device": 100, "host": 50}}}
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=True)
        )
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 100)
        self.assertEqual(result.host, 50)
        self.assertIsNone(result.storage)
        self.assertIsNone(result.storage_backend)

    def test_with_l3_storage_fields(self):
        """With storage field present, returns CachedTokensDetails with all fields."""
        ret_item = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 80,
                    "host": 20,
                    "storage": 200,
                    "storage_backend": "redis",
                }
            }
        }
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=True)
        )
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 80)
        self.assertEqual(result.host, 20)
        self.assertEqual(result.storage, 200)
        self.assertEqual(result.storage_backend, "redis")

    def test_missing_device_host_defaults_to_zero(self):
        """When device/host keys are missing, they default to 0."""
        ret_item = {"meta_info": {"cached_tokens_details": {}}}
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=True)
        )
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 0)
        self.assertEqual(result.host, 0)

    def test_storage_present_but_zero(self):
        """Storage field present with value 0 still includes storage."""
        ret_item = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 10,
                    "host": 5,
                    "storage": 0,
                    "storage_backend": "disk",
                }
            }
        }
        result = process_cached_tokens_details_from_ret(
            ret_item, self._make_request(return_cached_tokens_details=True)
        )
        self.assertEqual(result.storage, 0)
        self.assertEqual(result.storage_backend, "disk")

    def test_attribute_not_present_returns_none(self):
        """When request has no return_cached_tokens_details attribute, returns None."""
        req = object()  # plain object with no attributes
        ret_item = {"meta_info": {"cached_tokens_details": {"device": 1, "host": 2}}}
        result = process_cached_tokens_details_from_ret(ret_item, req)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
