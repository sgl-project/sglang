"""
Unit tests for sglang.srt.entrypoints.openai.utils.

Covers:
  - to_openai_style_logprobs
  - process_hidden_states_from_ret
  - process_routed_experts_from_ret
  - process_cached_tokens_details_from_ret

All tests run on CPU with no server or model weights required.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Stub out sgl_kernel before any sglang import so tests run on CPU-only runners.
for _mod in ("sgl_kernel", "sgl_kernel.kvcacheio"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sglang.srt.entrypoints.openai.protocol import CachedTokensDetails
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    to_openai_style_logprobs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**kwargs):
    """Minimal request-like object with the fields used by utils.py."""

    class _Req:
        return_hidden_states = False
        return_routed_experts = False
        return_cached_tokens_details = False

    req = _Req()
    for k, v in kwargs.items():
        setattr(req, k, v)
    return req


def _make_ret(meta_info: dict) -> dict:
    return {"meta_info": meta_info}


# ---------------------------------------------------------------------------
# to_openai_style_logprobs
# ---------------------------------------------------------------------------


class TestToOpenaiStyleLogprobs(unittest.TestCase):
    def test_all_none_returns_empty_logprobs(self):
        result = to_openai_style_logprobs()
        self.assertEqual(result.tokens, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(result.top_logprobs, [])
        self.assertEqual(result.text_offset, [])

    def test_output_token_logprobs_only(self):
        # Each entry is (logprob, token_id, token_text)
        output = [(-0.1, 42, "Hello"), (-0.5, 99, " world")]
        result = to_openai_style_logprobs(output_token_logprobs=output)
        self.assertEqual(result.tokens, ["Hello", " world"])
        self.assertEqual(result.token_logprobs, [-0.1, -0.5])
        # text_offset is always -1 (not yet supported)
        self.assertEqual(result.text_offset, [-1, -1])

    def test_input_token_logprobs_only(self):
        input_lp = [(-0.2, 1, "Hi"), (-0.3, 2, "!")]
        result = to_openai_style_logprobs(input_token_logprobs=input_lp)
        self.assertEqual(result.tokens, ["Hi", "!"])
        self.assertEqual(result.token_logprobs, [-0.2, -0.3])

    def test_input_then_output_appended_in_order(self):
        input_lp = [(-0.1, 1, "A")]
        output_lp = [(-0.2, 2, "B"), (-0.3, 3, "C")]
        result = to_openai_style_logprobs(
            input_token_logprobs=input_lp,
            output_token_logprobs=output_lp,
        )
        self.assertEqual(result.tokens, ["A", "B", "C"])
        self.assertEqual(result.token_logprobs, [-0.1, -0.2, -0.3])

    def test_output_top_logprobs(self):
        # top_logprobs: list of lists; each inner list is (logprob, token_id, token_text)
        top = [[(-0.1, 1, "yes"), (-2.0, 2, "no")], None]
        result = to_openai_style_logprobs(output_top_logprobs=top)
        self.assertEqual(result.top_logprobs[0], {"yes": -0.1, "no": -2.0})
        self.assertIsNone(result.top_logprobs[1])

    def test_input_top_logprobs(self):
        top = [[(-0.5, 10, "tok")]]
        result = to_openai_style_logprobs(input_top_logprobs=top)
        self.assertEqual(result.top_logprobs, [{"tok": -0.5}])

    def test_combined_token_and_top_logprobs(self):
        output_lp = [(-0.1, 1, "A")]
        output_top = [[(-0.1, 1, "A"), (-1.0, 2, "B")]]
        result = to_openai_style_logprobs(
            output_token_logprobs=output_lp,
            output_top_logprobs=output_top,
        )
        self.assertEqual(result.tokens, ["A"])
        self.assertEqual(result.top_logprobs[0]["A"], -0.1)

    def test_text_offset_always_minus_one(self):
        output = [(-0.1, 1, "x"), (-0.2, 2, "y"), (-0.3, 3, "z")]
        result = to_openai_style_logprobs(output_token_logprobs=output)
        self.assertEqual(result.text_offset, [-1, -1, -1])


# ---------------------------------------------------------------------------
# process_hidden_states_from_ret
# ---------------------------------------------------------------------------


class TestProcessHiddenStates(unittest.TestCase):
    def test_returns_none_when_flag_false(self):
        req = _make_request(return_hidden_states=False)
        ret = _make_ret({"hidden_states": [[1, 2, 3]]})
        self.assertIsNone(process_hidden_states_from_ret(ret, req))

    def test_returns_none_when_hidden_states_absent(self):
        req = _make_request(return_hidden_states=True)
        ret = _make_ret({})
        self.assertIsNone(process_hidden_states_from_ret(ret, req))

    def test_returns_none_when_hidden_states_is_none(self):
        req = _make_request(return_hidden_states=True)
        ret = _make_ret({"hidden_states": None})
        self.assertIsNone(process_hidden_states_from_ret(ret, req))

    def test_single_hidden_state_returns_empty_list(self):
        # len == 1: the branch returns [] (not the single element)
        req = _make_request(return_hidden_states=True)
        ret = _make_ret({"hidden_states": [[0.1, 0.2]]})
        self.assertEqual(process_hidden_states_from_ret(ret, req), [])

    def test_multiple_hidden_states_returns_last(self):
        states = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        req = _make_request(return_hidden_states=True)
        ret = _make_ret({"hidden_states": states})
        self.assertEqual(process_hidden_states_from_ret(ret, req), [0.5, 0.6])


# ---------------------------------------------------------------------------
# process_routed_experts_from_ret
# ---------------------------------------------------------------------------


class TestProcessRoutedExperts(unittest.TestCase):
    def test_returns_none_when_flag_false(self):
        req = _make_request(return_routed_experts=False)
        ret = _make_ret({"routed_experts": "expert_data"})
        self.assertIsNone(process_routed_experts_from_ret(ret, req))

    def test_returns_value_when_flag_true(self):
        req = _make_request(return_routed_experts=True)
        ret = _make_ret({"routed_experts": "expert_data"})
        self.assertEqual(process_routed_experts_from_ret(ret, req), "expert_data")

    def test_returns_none_when_key_missing(self):
        req = _make_request(return_routed_experts=True)
        ret = _make_ret({})
        self.assertIsNone(process_routed_experts_from_ret(ret, req))

    def test_request_without_attribute_treated_as_false(self):
        # Objects without return_routed_experts attr default to False via getattr
        class _MinReq:
            pass

        ret = _make_ret({"routed_experts": "data"})
        self.assertIsNone(process_routed_experts_from_ret(ret, _MinReq()))


# ---------------------------------------------------------------------------
# process_cached_tokens_details_from_ret
# ---------------------------------------------------------------------------


class TestProcessCachedTokensDetails(unittest.TestCase):
    def test_returns_none_when_flag_false(self):
        req = _make_request(return_cached_tokens_details=False)
        ret = _make_ret({"cached_tokens_details": {"device": 10, "host": 5}})
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_returns_none_when_details_absent(self):
        req = _make_request(return_cached_tokens_details=True)
        ret = _make_ret({})
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_returns_none_when_details_is_none(self):
        req = _make_request(return_cached_tokens_details=True)
        ret = _make_ret({"cached_tokens_details": None})
        self.assertIsNone(process_cached_tokens_details_from_ret(ret, req))

    def test_device_host_only(self):
        req = _make_request(return_cached_tokens_details=True)
        ret = _make_ret({"cached_tokens_details": {"device": 8, "host": 3}})
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 8)
        self.assertEqual(result.host, 3)
        self.assertIsNone(result.storage)
        self.assertIsNone(result.storage_backend)

    def test_with_storage_fields(self):
        req = _make_request(return_cached_tokens_details=True)
        ret = _make_ret(
            {
                "cached_tokens_details": {
                    "device": 4,
                    "host": 2,
                    "storage": 6,
                    "storage_backend": "s3",
                }
            }
        )
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertIsInstance(result, CachedTokensDetails)
        self.assertEqual(result.device, 4)
        self.assertEqual(result.host, 2)
        self.assertEqual(result.storage, 6)
        self.assertEqual(result.storage_backend, "s3")

    def test_missing_device_host_default_to_zero(self):
        req = _make_request(return_cached_tokens_details=True)
        ret = _make_ret({"cached_tokens_details": {}})
        result = process_cached_tokens_details_from_ret(ret, req)
        self.assertEqual(result.device, 0)
        self.assertEqual(result.host, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
