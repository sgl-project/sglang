import json
import unittest

from sglang.srt.managers.io_struct import ProfileReqInput


class TestProfileMergerHTTPAPI(unittest.TestCase):
    def test_profile_req_input_merge_profiles_json_serialization(self):
        # Test with merge_profiles=True
        req_input = ProfileReqInput(
            output_dir="/tmp/test",
            num_steps=5,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            merge_profiles=True,
        )

        # Convert to dict (as would happen in HTTP request)
        req_dict = {
            "output_dir": req_input.output_dir,
            "num_steps": req_input.num_steps,
            "activities": req_input.activities,
            "profile_by_stage": req_input.profile_by_stage,
            "merge_profiles": req_input.merge_profiles,
        }

        # Test JSON serialization
        json_str = json.dumps(req_dict)
        parsed_data = json.loads(json_str)

        self.assertTrue(parsed_data["merge_profiles"])
        self.assertEqual(parsed_data["output_dir"], "/tmp/test")
        self.assertEqual(parsed_data["num_steps"], 5)
        self.assertEqual(parsed_data["activities"], ["CPU", "GPU"])
        self.assertTrue(parsed_data["profile_by_stage"])

    def test_profile_req_input_merge_profiles_json_deserialization(self):
        # Test JSON data as would come from HTTP request
        json_data = {
            "output_dir": "/tmp/test",
            "num_steps": 10,
            "activities": ["CPU", "GPU", "MEM"],
            "profile_by_stage": False,
            "merge_profiles": True,
        }

        # Create ProfileReqInput from dict (as HTTP server would do)
        req_input = ProfileReqInput(**json_data)

        self.assertTrue(req_input.merge_profiles)
        self.assertEqual(req_input.output_dir, "/tmp/test")
        self.assertEqual(req_input.num_steps, 10)
        self.assertEqual(req_input.activities, ["CPU", "GPU", "MEM"])
        self.assertFalse(req_input.profile_by_stage)

    def test_profile_req_input_merge_profiles_default_value(self):
        # Test with minimal data
        json_data = {"output_dir": "/tmp/test"}

        req_input = ProfileReqInput(**json_data)
        self.assertFalse(req_input.merge_profiles)

    def test_profile_req_input_merge_profiles_explicit_false(self):
        json_data = {"output_dir": "/tmp/test", "merge_profiles": False}

        req_input = ProfileReqInput(**json_data)
        self.assertFalse(req_input.merge_profiles)

    def test_http_api_parameter_flow(self):
        # Simulate HTTP request data
        request_data = {
            "output_dir": "/tmp/test",
            "num_steps": 5,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "merge_profiles": True,
        }

        # Create ProfileReqInput as HTTP server would
        obj = ProfileReqInput(**request_data)

        # Verify the parameter is set correctly
        self.assertTrue(obj.merge_profiles)
        self.assertEqual(obj.output_dir, "/tmp/test")
        self.assertEqual(obj.num_steps, 5)
        self.assertEqual(obj.activities, ["CPU", "GPU"])
        self.assertTrue(obj.profile_by_stage)

    def test_http_api_parameter_validation(self):
        # Test with True
        json_data = {"merge_profiles": True}
        req_input = ProfileReqInput(**json_data)
        self.assertTrue(req_input.merge_profiles)

        # Test with False
        json_data = {"merge_profiles": False}
        req_input = ProfileReqInput(**json_data)
        self.assertFalse(req_input.merge_profiles)

        # Test with string "true" (should be converted by JSON parser)
        json_data = {"merge_profiles": "true"}
        req_input = ProfileReqInput(**json_data)
        self.assertEqual(req_input.merge_profiles, "true")  # String, not boolean

    def test_http_api_backward_compatibility(self):
        # Test minimal request (no merge_profiles)
        json_data = {}
        req_input = ProfileReqInput(**json_data)
        self.assertFalse(req_input.merge_profiles)  # Should default to False

        # Test with other parameters but no merge_profiles
        json_data = {
            "output_dir": "/tmp/test",
            "num_steps": 5,
            "activities": ["CPU", "GPU"],
        }
        req_input = ProfileReqInput(**json_data)
        self.assertFalse(req_input.merge_profiles)  # Should default to False

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
                req_input = ProfileReqInput(**test_case["data"])
                self.assertEqual(req_input.merge_profiles, test_case["expected_merge"])


if __name__ == "__main__":
    unittest.main()
