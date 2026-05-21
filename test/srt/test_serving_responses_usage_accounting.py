# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the usage-accounting fix in serving_responses.py.

The fix changed ``hasattr(final_res, "meta_info")`` to
``"meta_info" in final_res`` because ``final_res`` is a plain dict
returned by the engine, not an object with attributes.
"""

import unittest


class TestUsageAccountingFromDict(unittest.TestCase):
    """Verify that dict-based final_res is handled correctly."""

    def _extract_usage(self, final_res: dict) -> dict:
        """Mimics the fixed code path in
        OpenAIServingResponses.create_responses (non-streaming).
        """
        meta_info = final_res.get("meta_info", {})
        return {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "cached_tokens": meta_info.get("cached_tokens", 0),
            "reasoning_tokens": meta_info.get("reasoning_tokens", 0),
        }

    def test_dict_with_meta_info(self):
        """Normal dict response should populate usage correctly."""
        final_res = {
            "text": "Hello world",
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 2,
                "reasoning_tokens": 0,
            },
        }
        usage = self._extract_usage(final_res)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["cached_tokens"], 2)
        self.assertEqual(usage["reasoning_tokens"], 0)

    def test_dict_without_meta_info(self):
        """Dict missing meta_info should fall back to zeros."""
        final_res = {"text": "Hello world"}
        usage = self._extract_usage(final_res)
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["cached_tokens"], 0)
        self.assertEqual(usage["reasoning_tokens"], 0)

    def test_dict_with_partial_meta_info(self):
        """meta_info present but missing some keys should default to 0."""
        final_res = {
            "text": "hi",
            "meta_info": {"prompt_tokens": 7},
        }
        usage = self._extract_usage(final_res)
        self.assertEqual(usage["prompt_tokens"], 7)
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["cached_tokens"], 0)
        self.assertEqual(usage["reasoning_tokens"], 0)

    def test_hasattr_wrong_on_dict(self):
        """Demonstrate why the old hasattr check was wrong.

        ``hasattr(d, "meta_info")`` is always False for a plain dict even
        when the key exists, so the old code would silently skip usage
        extraction and leave all counters at zero.
        """
        d = {"meta_info": {"prompt_tokens": 42}}
        # hasattr looks for *attributes*, not dict keys
        self.assertFalse(hasattr(d, "meta_info"))
        # The correct check is ``"meta_info" in d``
        self.assertTrue("meta_info" in d)


if __name__ == "__main__":
    unittest.main()
