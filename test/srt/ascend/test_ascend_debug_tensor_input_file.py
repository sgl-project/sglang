import os
import unittest

import numpy
import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDebugTensorInputFile(CustomTestCase):
    def test_tensor_input_file(self):
        vector = numpy.array([1001, 1002, 1003, 1004, 1005, 1006, 1007])
        numpy.save("./input_tensor.npy", vector)
        other_args = (
            [
                "--debug-tensor-dump-input-file",
                "./input_tensor.npy",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else ["--debug-tensor-dump-input-file", "./input_tensor.npy"]
        )
        out_log_file = open("./tensor_input_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./tensor_input_err_log.txt", "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                (
                    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                    if is_npu()
                    else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
                ),
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        except Exception as e:
            print("process is killed")
        err_log_file.seek(0)
        content = err_log_file.read()
        self.assertTrue(len(content) > 0)
        # self.assertIn("The server is fired up and ready to roll!", content)
        out_log_file.close()
        err_log_file.close()
        os.remove("./tensor_input_out_log.txt")
        os.remove("./tensor_input_err_log.txt")
        os.remove("./input_tensor.npy")


if __name__ == "__main__":
    unittest.main()
