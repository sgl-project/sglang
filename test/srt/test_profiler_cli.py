"""
Unit tests for command-line profiler with merge_profiles parameter.

Usage:
    python test_profiler_cli.py
    python -m unittest test_profiler_cli.py -v
"""

import argparse
import unittest
from unittest.mock import MagicMock, patch

from sglang.profiler import _run_profile, run_profile


class TestProfilerCLI(unittest.TestCase):
    """Test cases for command-line profiler integration."""

    def test_run_profile_merge_profiles_parameter(self):
        """Test that run_profile accepts merge_profiles parameter."""
        # Test default value
        with patch("sglang.profiler._run_profile") as mock_run:
            run_profile(
                url="http://localhost:30000",
                num_steps=5,
                activities=["CPU", "GPU"],
                merge_profiles=False,
            )
            mock_run.assert_called_once()
            # Check that merge_profiles is passed as the last positional argument
            args = mock_run.call_args[0]
            self.assertEqual(
                len(args), 7
            )  # url, num_steps, activities, output_dir, profile_name, profile_by_stage, merge_profiles
            self.assertFalse(args[6])  # merge_profiles is the 7th argument (index 6)

        # Test explicit True value
        with patch("sglang.profiler._run_profile") as mock_run:
            run_profile(
                url="http://localhost:30000",
                num_steps=5,
                activities=["CPU", "GPU"],
                merge_profiles=True,
            )
            mock_run.assert_called_once()
            # Check that merge_profiles is passed as the last positional argument
            args = mock_run.call_args[0]
            self.assertEqual(
                len(args), 7
            )  # url, num_steps, activities, output_dir, profile_name, profile_by_stage, merge_profiles
            self.assertTrue(args[6])  # merge_profiles is the 7th argument (index 6)

    def test_run_profile_merge_profiles_default(self):
        """Test that merge_profiles defaults to False."""
        with patch("sglang.profiler._run_profile") as mock_run:
            run_profile(
                url="http://localhost:30000", num_steps=5, activities=["CPU", "GPU"]
            )
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            self.assertEqual(len(args), 7)  # All parameters including merge_profiles
            self.assertFalse(args[6])  # merge_profiles defaults to False

    def test_run_profile_json_data_includes_merge_profiles(self):
        """Test that _run_profile includes merge_profiles in JSON data."""
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"server": "info"}  # Mock JSON response
            mock_post.return_value = mock_response
            mock_get.return_value = mock_response

            _run_profile(
                url="http://localhost:30000",
                num_steps=5,
                activities=["CPU", "GPU"],
                merge_profiles=True,
            )

            # Verify the request was made with correct JSON data
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            json_data = call_args[1]["json"]

            self.assertIn("merge_profiles", json_data)
            self.assertTrue(json_data["merge_profiles"])

    def test_run_profile_json_data_merge_profiles_false(self):
        """Test that _run_profile includes merge_profiles=False in JSON data."""
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"server": "info"}  # Mock JSON response
            mock_post.return_value = mock_response
            mock_get.return_value = mock_response

            _run_profile(
                url="http://localhost:30000",
                num_steps=5,
                activities=["CPU", "GPU"],
                merge_profiles=False,
            )

            # Verify the request was made with correct JSON data
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            json_data = call_args[1]["json"]

            self.assertIn("merge_profiles", json_data)
            self.assertFalse(json_data["merge_profiles"])

    def test_command_line_arguments_merge_profiles(self):
        """Test that command-line argument parser includes merge_profiles."""
        from sglang.profiler import ArgumentParser

        # Create parser like in the main module
        parser = ArgumentParser(description="Benchmark the online serving throughput.")
        parser.add_argument("--url", type=str, default="http://localhost:30000")
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--profile-name", type=str, default=None)
        parser.add_argument("--num-steps", type=int, default=5)
        parser.add_argument(
            "--profile-by-stage",
            action=argparse.BooleanOptionalAction,
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--cpu", action=argparse.BooleanOptionalAction, type=bool, default=True
        )
        parser.add_argument(
            "--gpu", action=argparse.BooleanOptionalAction, type=bool, default=True
        )
        parser.add_argument(
            "--mem", action=argparse.BooleanOptionalAction, type=bool, default=False
        )
        parser.add_argument(
            "--rpd", action=argparse.BooleanOptionalAction, type=bool, default=False
        )
        parser.add_argument(
            "--merge-profiles",
            action=argparse.BooleanOptionalAction,
            type=bool,
            default=False,
        )

        # Test parsing with merge_profiles=True
        args = parser.parse_args(["--merge-profiles"])
        self.assertTrue(args.merge_profiles)

        # Test parsing with merge_profiles=False
        args = parser.parse_args(["--no-merge-profiles"])
        self.assertFalse(args.merge_profiles)

        # Test parsing without merge_profiles (should default to False)
        args = parser.parse_args([])
        self.assertFalse(args.merge_profiles)

    def test_command_line_arguments_merge_profiles_help(self):
        """Test that merge_profiles argument has proper help text."""
        from sglang.profiler import ArgumentParser

        parser = ArgumentParser(description="Benchmark the online serving throughput.")
        parser.add_argument(
            "--merge-profiles",
            action=argparse.BooleanOptionalAction,
            type=bool,
            default=False,
            help="Whether to merge profiles from all ranks into a single trace file",
        )

        # Get help text
        help_text = parser.format_help()
        self.assertIn("--merge-profiles", help_text)
        # The help text is wrapped, so check for the key part
        self.assertIn("merge profiles from all ranks", help_text)

    def test_integration_with_existing_parameters(self):
        """Test that merge_profiles works with existing parameters."""
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"server": "info"}  # Mock JSON response
            mock_post.return_value = mock_response
            mock_get.return_value = mock_response

            _run_profile(
                url="http://localhost:30000",
                num_steps=10,
                activities=["CPU", "GPU", "MEM"],
                output_dir="/tmp/custom",
                profile_name="test_profile",
                profile_by_stage=True,
                merge_profiles=True,
            )

            # Verify all parameters are included
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            json_data = call_args[1]["json"]

            self.assertEqual(json_data["num_steps"], "10")
            self.assertEqual(json_data["activities"], ["CPU", "GPU", "MEM"])
            # The output_dir gets modified by the profiler to include profile_name and timestamp
            self.assertTrue(json_data["output_dir"].startswith("/tmp/custom"))
            self.assertTrue(json_data["profile_by_stage"])
            self.assertTrue(json_data["merge_profiles"])


if __name__ == "__main__":
    unittest.main()
