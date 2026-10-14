import glob
import os
import re
import shutil
import tempfile
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNpuDebugTensorDumps(CustomTestCase):
    """Tensor dump verification of all parameter combinations in TP+PP scenarios

    [Test Category] Functional
    [Test Target] Debug Tensor Dumps on NPU
    --debug-tensor-dump-output-folder; --debug-tensor-dump-layers
    """

    TP_SIZE = 2
    PP_SIZE = 2
    TP_DIR_NUM = TP_SIZE * PP_SIZE
    dump_folder = tempfile.mkdtemp(prefix="tensor_folder")

    model = QWEN3_32B_WEIGHTS_PATH
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        TP_SIZE,
        "--pp-size",
        PP_SIZE,
        "--debug-tensor-dump-output-folder",
        dump_folder,
        "--debug-tensor-dump-layers",
        "2",
        "3",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        """Set up the test class by launching the server with the specified configuration."""
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.base_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.dump_folder):
            shutil.rmtree(cls.dump_folder)

    def test_debug_tensor_dump(self):
        """Send multiple requests."""
        text1 = "The capital of France is"
        for i in range(3):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text1,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)

        # Verify that the debug-tensor-dump-output-folder configuration is effective and that the tensor folder exists.
        res = glob.glob(f"{self.dump_folder}/TP*_PP*_Rank*_pid*")
        self.assertEqual(len(res), self.TP_DIR_NUM)

        # Verify the directory structure of tensor_dump
        subdirs = [
            d
            for d in os.listdir(self.dump_folder)
            if os.path.isdir(os.path.join(self.dump_folder, d))
        ]
        self.assertGreater(len(subdirs), 0)

        pp0_dirs = [d for d in subdirs if re.search(r"PP0[^0-9]", d)]
        self.assertGreater(len(pp0_dirs), 0)

        # Verify that the contents of tensor_dump exist as .pt format files.
        first_pp0_dir = os.path.join(self.dump_folder, pp0_dirs[0])
        files = os.listdir(first_pp0_dir)
        pass_files = [f for f in files if f.startswith("Pass") and f.endswith(".pt")]
        self.assertGreater(len(pass_files), 0)

        # Verify that the tensor dump file contains tensor data.
        pt_file = os.path.join(first_pp0_dir, pass_files[0])
        tensor_data = torch.load(pt_file)
        self.assertIn("model.layers.2.input_layernorm", tensor_data)
        self.assertIn("model.layers.3.input_layernorm", tensor_data)
        self.assertIn("model.layers.4.input_layernorm", tensor_data)
        self.assertGreater(len(tensor_data), 0)


if __name__ == "__main__":
    unittest.main()
