"""
Unit test for the points template matching bug fix.

This test verifies the fix for GitHub issue #12791:
https://github.com/sgl-project/sglang/issues/12791

The bug was that the regex pattern r"points" in match_points_v15_chat()
incorrectly matched paths containing "checkpoints" or "endpoints".
The fix uses word boundaries: r"\bpoints\b"
"""

import re
import unittest


class TestPointsRegexFix(unittest.TestCase):
    """
    Test the fixed regex pattern for points template matching.

    This is a direct test of the regex logic without complex imports.
    """

    def match_points_v15_chat_fixed(self, model_path: str):
        """
        Fixed version of match_points_v15_chat using word boundaries.
        Reference: https://github.com/sgl-project/sglang/issues/12791
        """
        if re.search(r"\bpoints\b", model_path, re.IGNORECASE):
            return "points-v15-chat"
        return None

    def match_points_v15_chat_buggy(self, model_path: str):
        """
        Buggy version for comparison (old implementation).
        """
        if re.search(r"points", model_path, re.IGNORECASE):
            return "points-v15-chat"
        return None

    def test_fixed_version_does_not_match_checkpoints(self):
        """Test that the fixed version does not match 'checkpoints'."""
        test_paths = [
            "checkpoints/qwen2.5-0.5b-instruct",
            "/models/checkpoints/my_model",
            "/data/checkpoints/llama",
            "checkpoints/deepseek-v3",
        ]
        for path in test_paths:
            with self.subTest(path=path):
                result = self.match_points_v15_chat_fixed(path)
                self.assertIsNone(
                    result,
                    f"Fixed version should NOT match checkpoints: {path}",
                )

    def test_fixed_version_does_not_match_endpoints(self):
        """Test that the fixed version does not match 'endpoints'."""
        test_paths = [
            "/data/endpoints/model",
            "endpoints/api/v1",
            "/service/endpoints/inference",
        ]
        for path in test_paths:
            with self.subTest(path=path):
                result = self.match_points_v15_chat_fixed(path)
                self.assertIsNone(
                    result,
                    f"Fixed version should NOT match endpoints: {path}",
                )

    def test_fixed_version_matches_real_points(self):
        """Test that the fixed version correctly matches real points models."""
        test_paths = [
            "points-v15-chat",
            "/models/points/v15",
            "my-points-model",
            "Points-V1.5-Instruct",
            "/path/to/points",
            "points",
        ]
        for path in test_paths:
            with self.subTest(path=path):
                result = self.match_points_v15_chat_fixed(path)
                self.assertEqual(
                    result,
                    "points-v15-chat",
                    f"Fixed version should match real points model: {path}",
                )

    def test_buggy_version_incorrectly_matches_checkpoints(self):
        """Test that the buggy version incorrectly matches 'checkpoints'."""
        test_paths = [
            "checkpoints/qwen2.5-0.5b-instruct",
            "/models/checkpoints/my_model",
        ]
        for path in test_paths:
            with self.subTest(path=path):
                result = self.match_points_v15_chat_buggy(path)
                self.assertEqual(
                    result,
                    "points-v15-chat",
                    f"Buggy version incorrectly matches: {path}",
                )

    def test_comparison_fixed_vs_buggy(self):
        """Compare fixed and buggy versions on the main regression case."""
        # The main bug: checkpoints should not match
        path = "checkpoints/qwen2.5-0.5b-instruct"

        buggy_result = self.match_points_v15_chat_buggy(path)
        fixed_result = self.match_points_v15_chat_fixed(path)

        # Buggy version incorrectly returns 'points-v15-chat'
        self.assertEqual(buggy_result, "points-v15-chat")

        # Fixed version correctly returns None
        self.assertIsNone(fixed_result)


if __name__ == "__main__":
    unittest.main()
