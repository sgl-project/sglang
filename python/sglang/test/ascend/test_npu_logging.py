import os
import re
import tempfile
import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestNPULoggingBase(CustomTestCase):
    """Testcase：Test base class to verify whether the parameters in the logging function are correct.

    Description:
        Includes methods for initializing data and methods for verifying the correctness of the logging function.
    """

    @staticmethod
    def get_lines_with_keyword(filename, keyword):
        """Find and return lines matching a regex keyword from a specified file, with line numbers and content.

        Function Description:
            Reads the target file line by line, uses the input keyword as a regular expression pattern to match each line's content.
            For each line that matches the regex pattern, encapsulates the line number (1-indexed) and content into a dictionary,
            and finally returns a list of dictionaries containing all matched lines.

        Args:
            filename (str): Path to the file to be read
            keyword (str): Regular expression pattern for matching

        Returns:
            List[Dict[str, Union[str, int]]]
                List of dictionaries for matched lines, each dictionary contains two key-value pairs:
                - "line_number": int - Line number of the matched line (starts from 1)
                - "content": str - Full text content of the matched line
        """
        results = []
        try:
            with open(filename, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    if re.match(keyword, line):
                        results.append(
                            {
                                "line_number": line_num,
                                "content": line.strip(),
                            }
                        )
            return results
        except Exception as e:
            print(f"error:{e}")
            return []

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--log-requests",
        ]
        cls.out_log_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
        cls.out_log_name = cls.out_log_file_obj.name
        cls.out_log_file = cls.out_log_file_obj
        cls.err_log_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
        cls.err_log_name = cls.err_log_file_obj.name
        cls.err_log_file = cls.err_log_file_obj
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        os.remove(cls.out_log_name)
        cls.err_log_file.close()
        os.remove(cls.err_log_name)

    @classmethod
    def launch_server(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def inference_once(self, max_tokens=32):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "Failed to call generate API")
        self.assertIn("Paris", response.text, "Inference out error.")

    def wait_for_log_content(self, timeout=30):
        """Wait for and return the content of the specified log file, with timeout handling.

        Function Description:
            Continuously reads the target log file in a loop within the set timeout period,
            avoids assertion failures caused by reading the log too early before log writing is completed.
            Returns the log content immediately once the file has non-empty content,
            otherwise waits and retries reading at intervals until the timeout is reached.

        Args:
            timeout (int, optional): Maximum waiting time in seconds, default value is 30 seconds.

        Returns:
            str
                Full text content read from the log file:
                - Non-empty string if log content is detected within the timeout period
                - Empty string if no log content is found after the timeout expires
        """
        start_time = time.time()
        content = ""
        while time.time() - start_time < timeout:
            with open(self.out_log_file.name, "r", encoding="utf-8") as f:
                content = f.read()
            if content:
                break
            time.sleep(0.5)
        return content
