"""
Test weight version functionality.

This test suite verifies the weight_version feature implementation including:
1. Default weight_version setting
2. /get_weight_version endpoint
3. /update_weight_version endpoint
4. /generate request meta_info contains weight_version
5. OpenAI API response metadata contains weight_version
"""

import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)


class TestWeightVersion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """Start server once for all tests with custom weight version."""
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:30000"
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--weight-version",
                "test_version_1.0",
                "--attention-backend",
                "flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """Terminate server after all tests complete."""
        if cls.process:
            cls.process.terminate()

    def test_weight_version_comprehensive(self):
        """Comprehensive test for all weight_version functionality."""

        response = requests.get(f"{self.base_url}/get_model_info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("weight_version", data)
        self.assertEqual(data["weight_version"], "test_version_1.0")

        response = requests.get(f"{self.base_url}/get_weight_version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("weight_version", data)
        self.assertEqual(data["weight_version"], "test_version_1.0")

        request_data = {
            "text": "Hello, how are you?",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 5,
            },
        }
        response = requests.post(f"{self.base_url}/generate", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("meta_info", data)
        self.assertIn("weight_version", data["meta_info"])
        self.assertEqual(data["meta_info"]["weight_version"], "test_version_1.0")

        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions", json=request_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertIn("weight_version", data["metadata"])
        self.assertEqual(data["metadata"]["weight_version"], "test_version_1.0")

        request_data = {
            "model": self.model,
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0.0,
        }
        response = requests.post(f"{self.base_url}/v1/completions", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertIn("weight_version", data["metadata"])
        self.assertEqual(data["metadata"]["weight_version"], "test_version_1.0")

        update_data = {
            "new_version": "updated_version_2.0",
            "abort_all_requests": False,
        }
        response = requests.post(
            f"{self.base_url}/update_weight_version", json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["new_version"], "updated_version_2.0")

        response = requests.get(f"{self.base_url}/get_weight_version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["weight_version"], "updated_version_2.0")

        gen_data = {
            "text": "Test persistence",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 3},
        }
        response = requests.post(f"{self.base_url}/generate", json=gen_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["meta_info"]["weight_version"], "updated_version_2.0")

        chat_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 3,
            "temperature": 0.0,
        }
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=chat_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["metadata"]["weight_version"], "updated_version_2.0")

        update_data = {"new_version": "final_version_3.0", "abort_all_requests": True}
        response = requests.post(
            f"{self.base_url}/update_weight_version", json=update_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["new_version"], "final_version_3.0")

        # Check /get_weight_version
        response = requests.get(f"{self.base_url}/get_weight_version")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["weight_version"], "final_version_3.0")

        # Check /get_model_info
        response = requests.get(f"{self.base_url}/get_model_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["weight_version"], "final_version_3.0")

        # Check /generate meta_info
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Final test",
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 2},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["meta_info"]["weight_version"], "final_version_3.0"
        )

        # Check OpenAI chat metadata
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Final"}],
                "max_tokens": 2,
                "temperature": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["metadata"]["weight_version"], "final_version_3.0"
        )

        print("All weight_version functionality tests passed!")

    def test_update_weight_version_with_weight_updates(self):
        """Test that weight_version can be updated along with weight updates using real model data."""
        print("Testing weight_version update with real weight operations...")

        # Get current model info for reference
        model_info_response = requests.get(f"{self.base_url}/get_model_info")
        self.assertEqual(model_info_response.status_code, 200)
        current_model_path = model_info_response.json()["model_path"]

        update_data = {
            "model_path": current_model_path,
            "load_format": "auto",
            "abort_all_requests": False,
            "weight_version": "disk_update_v2.0.0",
        }

        response = requests.post(
            f"{self.base_url}/update_weights_from_disk", json=update_data
        )
        self.assertEqual(
            response.status_code,
            200,
            f"update_weights_from_disk failed with status {response.status_code}",
        )

        # Verify version was updated
        version_response = requests.get(f"{self.base_url}/get_weight_version")
        self.assertEqual(version_response.status_code, 200)
        self.assertEqual(
            version_response.json()["weight_version"], "disk_update_v2.0.0"
        )

        print("Weight update with weight_version test completed!")


if __name__ == "__main__":
    unittest.main()
