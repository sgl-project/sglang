import re
import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogRequestLevel0(TestNPULoggingBase):
    """Testcase: Verify that the logs include request logs with the corresponding verbosity level when --log-requests-level is set.

    [Test Category] Parameter
    [Test Target] --log-requests-level
    """

    log_requests_level = 0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.finish_message_level_dict = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*, video_data=None.*, sampling_params=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*, video_data=None.*, sampling_params=.*",
        }
        cls.finish_message = (
            r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, .*"
        )
        cls.keyword_output_id_start = (
            "'output_ids': ["  # Start delimiter for token ID array
        )
        cls.keyword_output_id_end = "], 'meta_info'"

        cls.other_args.extend(["--log-requests"])
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])
        cls.launch_server()

    def test_log_requests_level(self):
        """
        Validate that log content complies with expectations for different --log-requests-level configurations.

        Core Functionality:
            1. Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
               - Max 100 new tokens for --log-requests-level ≤ 1 (reduce generation time for low-detail logging)
               - Max 2500 new tokens for --log-requests-level ≥ 2 (exceeds 2048 to test truncation behavior)
            2. Verify the log file contains level-specific keywords matching the target log_requests_level
            3. Validate token count preservation rules in logs:
               - Level 2: Logs are truncated to retain only 2048 tokens (partial input/output)
               - Level 3: Logs retain all generated tokens (full input/output)
               - Levels ≤1: No token count validation (only metadata/sampling params logged)
        """
        # Step 1: Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
        max_new_token = 2500 if self.log_requests_level >= 2 else 100
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"just return me a long string, generate as much as possible.",
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
            },
        )
        self.assertEqual(response.status_code, 200)

        # Step 2: Verify the log file contains level-specific keywords matching the target log_requests_level
        self.out_log_file.seek(0)
        content = self.wait_for_log_content()
        self.assertTrue(len(content) > 0)
        self.assertIsNotNone(
            re.search(
                self.finish_message_level_dict[str(self.log_requests_level)], content
            )
        )
        # The total number of generated tokens should equal the configured maximum number of generated tokens
        lines = self.get_lines_with_keyword(self.out_log_name, self.finish_message)
        self.assertGreater(len(lines), 0, "Did not find finish message in log.")
        finish_message = lines[-1]["content"]
        self.assertIn(f"'completion_tokens': {max_new_token}", finish_message)

        # Step 3: Validate token count preservation rules in logs:
        if self.log_requests_level >= 2:
            # Extract the content of output_ids to count the number of generated tokens recorded in the logs
            output_ids_start_index = finish_message.find(
                self.keyword_output_id_start
            ) + len(self.keyword_output_id_start)
            output_ids_end_index = finish_message.find(self.keyword_output_id_end)
            output_ids_list_str = finish_message[
                output_ids_start_index:output_ids_end_index
            ].strip()
            if self.log_requests_level == 2:
                # When --log-requests-level=2, the log records a maximum of 2048 tokens (truncated content)
                self.assertIn("] ... [", output_ids_list_str)
                output_ids_list_str = output_ids_list_str.replace("] ... [", ", ")
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count == 2048)
            else:
                # When --log-requests_level=3, the log records all generated token content (no truncation)
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count == max_new_token)


class TestNPULogRequestLevel1(TestNPULogRequestLevel0):
    log_requests_level = 1


class TestNPULogRequestLevel2(TestNPULogRequestLevel0):
    log_requests_level = 2


class TestNPULogRequestLevel3(TestNPULogRequestLevel0):
    log_requests_level = 3


if __name__ == "__main__":
    unittest.main()
