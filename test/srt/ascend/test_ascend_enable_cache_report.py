import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEnableCacheReport(CustomTestCase):
    def test_enable_cache_report(self):
        other_args = (
            [
                "--enable-cache-report",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else [
                "--enable-cache-report",
            ]
        )
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)
        try:
            for i in range(3):
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "just return me a string with of 5000 characters,just return me a string with of 5000 characters,just return me a string with of 5000 characters,just return me a string with of 5000 characters,just return me a string with of 5000 characters,just return me a string with of 5000 characters,just return me a string with of 5000 characters, 20, 30, 40, 50,60,70,4250,11, 2031, 11, 17837, 15, 11, 220, 2636, 15, 11, 220, 5067, 15, 11, 220, 7007, 15, 11, 220, 4728, 15, 11, 220, 7467, 15, 11, 220, 1041, 410, 11, 220, 5120, 410, 11, 220, 4364, 410, 11, 220, 5894, 410, 11, 220, 6860, 410, 11, 220, 3965, 410, 11, 220, 6330, 410, 11, 220, 8258, 410, 11, 220, 5245, 410, 11, 220, 7028, 410, 11, 220, 1049, 410, 11, 220, 8848, 410, 11, 220, 8610, 410, 11, 220, 9870, 410, 11, 220, 8273, 410, 11, 220, 5154, 410, 11, 220, 11387, 410, 11, 220, 10914, 410, 11, 220, 11209, 410, 11, 220, 13754, 410, 11, 220, 3101, 410, 11, 220, 12226, 410, 11, 220, 9588, 410, 11, 220, 10568, 410, 11, 220, 13679, 410, 11, 220, 8652, 410, 11, 220, 6843, 410, 11, 220, 14648, 410, 11, 220, 13897, 410, 11, 220, 15515, 410, 11, 220, 3443, 410, 11, 220, 14487, 410, 11, 220, 12819, 410, 11, 220, 14245, 410, 11, 220, 14868, 410, 11, 220, 10617, 410, 11, 220, 16551, 410, 11, 220, 17711, 410, 11, 220, 11738, 410, 11, 220, 18518, 410, 11, 220, 2636, 410, 11, 220, 15633, 410, 11, 220, 15830, 410, 11, 220, 17252, 410, 11, 220, 17048, 410, 11, 220, 13506, 410, 11, 220, 17698, 410, 11, 220, 18712, 410, 11, 220, 18216, 410, 11, 220, 20615, 410, 11, 220, 5067, 410, 11, 220, 17608, 410, 11, 220, 17416, 410, 11, 220, 18660, 410, 11, 220, 14033, 410, 11, 220, 13655, 410, 11, 220, 19274, 410, 11, 220, 21218, 410, 11, 220, 17814, 410, 11, 220, 21741, 410, 11, 220, 7007, 410, 11, 220, 19027, 410, 11, 220, 13104, 410, 11, 220, 20785, 410, 11, 220, 21112, 410, 11, 220, 11711, 410, 11, 220, 19104, 410, 11, 220, 20772, 410, 11, 220, 19423, 410, 11, 220, 22876, 410, 11, 220, 4728, 410, 11, 220, 19232, 410, 11, 220, 18248, 410, 11, 220, 21221, 410, 11, 220, 19899, 410, 11, 220, 16217, 410, 11, 220, 18670, 410, 11, 220, 22440, 410, 11, 220, 19272, 410, 11, 220, 21381, 410, 11, 220, 7467, 410, 11, 220, 21056, 410, 11, 220, 18485, 410, 11, 220, 19306, 410, 11, 220, 21251, 410, 11, 220, 15862, 410, 11, 220, 16415, 410, 11, 220, 21133, 410, 11, 220, 19068, 410, 11, 220, 19146, 410, 11, 220, 1041, 931, 11, 220, 4645, 931, 11, 220, 4278, 931, 11, 220, 6889, 931, 11, 220, 6849, 931, 11, 220, 6550, 931, 11, 220, 7461, 931, 11, 220, 7699, 931, 11, 220, 6640, 931, 11, 220, 7743, 931, 11, 220, 5120, 931, 11, 220, 5037, 931, 11, 220, 7261, 931, 11, 220,",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 1000,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                if i == 2:
                    print("------------------------look------", response.json())
                    self.assertTrue(
                        int(response.json()["meta_info"]["cached_tokens"]) > 0
                    )
            response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["enable_cache_report"], True)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":

    unittest.main()
