import json
import unittest

from sglang.srt.managers.io_struct import ProfileReq
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestProfileMergerHTTPAPI(CustomTestCase):
    def test_profile_req_merge_profiles_json_serialization(self):
        # Test with merge_profiles=True
        req = ProfileReq(
            output_dir="/tmp/test",
            num_steps=5,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            merge_profiles=True,
        )

        # Convert to dict (as would happen in HTTP request)
        req_dict = {
            "output_dir": req.output_dir,
            "num_steps": req.num_steps,
            "activities": req.activities,
            "profile_by_stage": req.profile_by_stage,
            "merge_profiles": req.merge_profiles,
        }

        # Test JSON serialization
        json_str = json.dumps(req_dict)
        parsed_data = json.loads(json_str)

        self.assertTrue(parsed_data["merge_profiles"])
        self.assertEqual(parsed_data["output_dir"], "/tmp/test")
        self.assertEqual(parsed_data["num_steps"], 5)
        self.assertEqual(parsed_data["activities"], ["CPU", "GPU"])
        self.assertTrue(parsed_data["profile_by_stage"])

    def test_profile_req_merge_profiles_json_deserialization(self):
        # Test JSON data as would come from HTTP request
        json_data = {
            "output_dir": "/tmp/test",
            "num_steps": 10,
            "activities": ["CPU", "GPU", "MEM"],
            "profile_by_stage": False,
            "merge_profiles": True,
        }

        # Create ProfileReq from dict (as HTTP server would do)
        req = ProfileReq(**json_data)

        self.assertTrue(req.merge_profiles)
        self.assertEqual(req.output_dir, "/tmp/test")
        self.assertEqual(req.num_steps, 10)
        self.assertEqual(req.activities, ["CPU", "GPU", "MEM"])
        self.assertFalse(req.profile_by_stage)

    def test_profile_req_merge_profiles_default_value(self):
        # Test with minimal data
        json_data = {"output_dir": "/tmp/test"}

        req = ProfileReq(**json_data)
        self.assertFalse(req.merge_profiles)

    def test_profile_req_merge_profiles_explicit_false(self):
        json_data = {"output_dir": "/tmp/test", "merge_profiles": False}

        req = ProfileReq(**json_data)
        self.assertFalse(req.merge_profiles)

    def test_http_api_parameter_flow(self):
        # Simulate HTTP request data
        request_data = {
            "output_dir": "/tmp/test",
            "num_steps": 5,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "merge_profiles": True,
        }

        # Create ProfileReq as HTTP server would
        obj = ProfileReq(**request_data)

        # Verify the parameter is set correctly
        self.assertTrue(obj.merge_profiles)
        self.assertEqual(obj.output_dir, "/tmp/test")
        self.assertEqual(obj.num_steps, 5)
        self.assertEqual(obj.activities, ["CPU", "GPU"])
        self.assertTrue(obj.profile_by_stage)

    def test_http_api_parameter_validation(self):
        # Test with True
        json_data = {"merge_profiles": True}
        req = ProfileReq(**json_data)
        self.assertTrue(req.merge_profiles)

        # Test with False
        json_data = {"merge_profiles": False}
        req = ProfileReq(**json_data)
        self.assertFalse(req.merge_profiles)

        # Test with string "true" (should be converted by JSON parser)
        json_data = {"merge_profiles": "true"}
        req = ProfileReq(**json_data)
        self.assertEqual(req.merge_profiles, "true")  # String, not boolean

    def test_http_api_backward_compatibility(self):
        # Test minimal request (no merge_profiles)
        json_data = {}
        req = ProfileReq(**json_data)
        self.assertFalse(req.merge_profiles)  # Should default to False

        # Test with other parameters but no merge_profiles
        json_data = {
            "output_dir": "/tmp/test",
            "num_steps": 5,
            "activities": ["CPU", "GPU"],
        }
        req = ProfileReq(**json_data)
        self.assertFalse(req.merge_profiles)  # Should default to False

    def test_http_api_parameter_combinations(self):
        test_cases = [
            {
                "name": "minimal with merge_profiles",
                "data": {"merge_profiles": True},
                "expected_merge": True,
            },
            {
                "name": "full parameters with merge_profiles=True",
                "data": {
                    "output_dir": "/tmp/test",
                    "num_steps": 10,
                    "activities": ["CPU", "GPU", "MEM"],
                    "profile_by_stage": True,
                    "with_stack": True,
                    "record_shapes": True,
                    "merge_profiles": True,
                },
                "expected_merge": True,
            },
            {
                "name": "full parameters with merge_profiles=False",
                "data": {
                    "output_dir": "/tmp/test",
                    "num_steps": 10,
                    "activities": ["CPU", "GPU", "MEM"],
                    "profile_by_stage": False,
                    "with_stack": False,
                    "record_shapes": False,
                    "merge_profiles": False,
                },
                "expected_merge": False,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                req = ProfileReq(**test_case["data"])
                self.assertEqual(req.merge_profiles, test_case["expected_merge"])


if __name__ == "__main__":
    unittest.main()
