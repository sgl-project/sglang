import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)


class TestGetExplicitSamplingKeys(unittest.TestCase):
    """Test get_explicit_sampling_keys() for ChatCompletionRequest and CompletionRequest."""

    def test_chat_no_explicit_params(self):
        """When user only sends messages, no sampling keys should be explicit."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
        )
        keys = req.get_explicit_sampling_keys()
        self.assertEqual(keys, set())

    def test_chat_explicit_temperature(self):
        """When user sets temperature, it should appear in explicit keys."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("temperature", keys)

    def test_chat_explicit_multiple_params(self):
        """When user sets multiple params, all should appear in explicit keys."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.1,
            max_completion_tokens=100,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("temperature", keys)
        self.assertIn("top_p", keys)
        self.assertIn("presence_penalty", keys)
        self.assertIn("max_new_tokens", keys)

    def test_chat_seed_maps_to_sampling_seed(self):
        """The 'seed' field should map to 'sampling_seed' key."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            seed=42,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("sampling_seed", keys)

    def test_completion_no_explicit_params(self):
        """CompletionRequest with only prompt should have no explicit sampling keys."""
        req = CompletionRequest(prompt="hello")
        keys = req.get_explicit_sampling_keys()
        self.assertEqual(keys, set())

    def test_completion_explicit_temperature(self):
        """CompletionRequest with explicit temperature."""
        req = CompletionRequest(prompt="hello", temperature=0.5)
        keys = req.get_explicit_sampling_keys()
        self.assertIn("temperature", keys)

    def test_completion_max_tokens_maps_to_max_new_tokens(self):
        """CompletionRequest 'max_tokens' should map to 'max_new_tokens'."""
        req = CompletionRequest(prompt="hello", max_tokens=50)
        keys = req.get_explicit_sampling_keys()
        self.assertIn("max_new_tokens", keys)


class TestResponsesGetExplicitSamplingKeys(unittest.TestCase):
    """Test get_explicit_sampling_keys() for ResponsesRequest."""

    def test_responses_no_explicit_params(self):
        """When user only sends input, no sampling keys should be explicit."""
        req = ResponsesRequest(
            model="test-model",
            input="hello",
        )
        keys = req.get_explicit_sampling_keys()
        self.assertEqual(keys, set())

    def test_responses_explicit_temperature(self):
        """When user sets temperature, it should appear in explicit keys."""
        req = ResponsesRequest(
            model="test-model",
            input="hello",
            temperature=0.5,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("temperature", keys)

    def test_responses_explicit_multiple_params(self):
        """When user sets multiple params, all should appear in explicit keys."""
        req = ResponsesRequest(
            model="test-model",
            input="hello",
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=100,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("temperature", keys)
        self.assertIn("top_p", keys)
        self.assertIn("max_new_tokens", keys)

    def test_responses_max_output_tokens_maps_to_max_new_tokens(self):
        """ResponsesRequest 'max_output_tokens' should map to 'max_new_tokens'."""
        req = ResponsesRequest(
            model="test-model",
            input="hello",
            max_output_tokens=200,
        )
        keys = req.get_explicit_sampling_keys()
        self.assertIn("max_new_tokens", keys)


class TestPreferredSamplingParamsMerge(unittest.TestCase):
    """Test the merge logic between preferred_sampling_params and request params."""

    def _merge(self, preferred, request_params, explicit_keys):
        """Simulate the merge logic from tokenizer_manager._create_tokenized_object."""
        if preferred:
            sampling_kwargs = dict(preferred)
            for k, v in request_params.items():
                if explicit_keys is None or k in explicit_keys:
                    sampling_kwargs[k] = v
        else:
            sampling_kwargs = request_params
        return sampling_kwargs

    def test_no_preferred_params(self):
        """Without preferred params, request params are used as-is."""
        request_params = {"temperature": 1.0, "max_new_tokens": 16}
        result = self._merge(None, request_params, set())
        self.assertEqual(result, request_params)

    def test_preferred_not_overridden_by_defaults(self):
        """Preferred params should NOT be overridden by non-explicit request defaults."""
        preferred = {"temperature": 0.8, "max_new_tokens": 100}
        request_params = {
            "temperature": 1.0,  # default, not explicit
            "max_new_tokens": None,  # default, not explicit
            "presence_penalty": 0.0,  # default, not explicit
        }
        explicit_keys = set()  # user didn't set any sampling params

        result = self._merge(preferred, request_params, explicit_keys)
        self.assertEqual(result["temperature"], 0.8)
        self.assertEqual(result["max_new_tokens"], 100)

    def test_explicit_user_value_overrides_preferred(self):
        """User-explicit values should override preferred params."""
        preferred = {"temperature": 0.8, "max_new_tokens": 100}
        request_params = {
            "temperature": 0.5,  # user explicitly set this
            "max_new_tokens": None,  # default, not explicit
            "presence_penalty": 0.0,  # default, not explicit
        }
        explicit_keys = {"temperature"}  # only temperature was explicit

        result = self._merge(preferred, request_params, explicit_keys)
        self.assertEqual(result["temperature"], 0.5)  # overridden by user
        self.assertEqual(result["max_new_tokens"], 100)  # kept from preferred

    def test_explicit_keys_none_falls_back_to_old_behavior(self):
        """When explicit_keys is None (e.g., /generate API), all keys override."""
        preferred = {"temperature": 0.8, "max_new_tokens": 100}
        request_params = {
            "temperature": 1.0,
            "max_new_tokens": 16,
        }
        explicit_keys = None  # no info about explicit keys

        result = self._merge(preferred, request_params, explicit_keys)
        self.assertEqual(result["temperature"], 1.0)  # overridden (old behavior)
        self.assertEqual(result["max_new_tokens"], 16)  # overridden (old behavior)

    def test_preferred_params_fill_missing_keys(self):
        """Preferred params should provide values for keys not in request."""
        preferred = {"temperature": 0.8, "top_k": 50, "max_new_tokens": 200}
        request_params = {"presence_penalty": 0.1}
        explicit_keys = {"presence_penalty"}

        result = self._merge(preferred, request_params, explicit_keys)
        self.assertEqual(result["temperature"], 0.8)
        self.assertEqual(result["top_k"], 50)
        self.assertEqual(result["max_new_tokens"], 200)
        self.assertEqual(result["presence_penalty"], 0.1)

    def test_user_explicitly_sets_zero(self):
        """User explicitly setting 0.0 should override preferred (not treated as default)."""
        preferred = {"presence_penalty": 0.5}
        request_params = {"presence_penalty": 0.0}
        explicit_keys = {"presence_penalty"}  # user explicitly set it to 0.0

        result = self._merge(preferred, request_params, explicit_keys)
        self.assertEqual(result["presence_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
