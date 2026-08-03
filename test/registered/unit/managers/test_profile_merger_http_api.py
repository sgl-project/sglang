import unittest

from sglang.srt.managers.io_struct import ProfileReq
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestProfileMergerHTTPAPI(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
