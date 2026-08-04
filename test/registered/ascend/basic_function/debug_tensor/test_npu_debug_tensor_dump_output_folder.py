import glob
import unittest
from abc import ABC

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH, run_command
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

TP_SIZE = 4
PP_SIZE = 1
TP_DIR_NUM = TP_SIZE * PP_SIZE
FILE_PATTERN_PP0 = "./TP0_PP0_Rank0_pid*"
FILE_PATTERN_PP1 = "./TP0_PP1_Rank4_pid*"
PT_FILE_NAME = "Pass00000.pt"


class TestDebugTensorDumpOutputFolderBase(ABC):
    """
    Testcase：Verify that the configuration parameters --debug-tensor-dump-output-folder and --debug-tensor-dump-layers correctly generate tensor .pt files.

    [Test Category] Parameter
    [Test Target] --debug-tensor-dump-output-folder; --debug-tensor-dump-layers
    """

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
        "--debug-tensor-dump-output-folder",
        "./",
        "--skip-server-warmup",
    ]
    other_args = []

    @classmethod
    def setUpClass(cls):
        """Set up the test class by launching the server with the specified configuration."""
        cls._cleanup_directories()

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.base_args + cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        cls._cleanup_directories()

    @classmethod
    def _cleanup_directories(cls):
        """Remove test directories with retry mechanism."""
        for _ in range(3):
            run_command("rm -rf ./TP*_PP*")
            result = run_command("ls -d ./TP*_PP* 2>/dev/null || echo ''")
            if not result.strip():
                break

    def sending_request(self):
        """Send a request to the server and count the number of generated tensor dump directories."""
        text1 = "The capital of France is"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
            },
        )
        res = run_command("ls -d TP*_PP*_Rank*_pid* | wc -l")
        return response, res

    def get_layers_from_tensor_file(self, file_pattern, file_name=PT_FILE_NAME):
        """Extract layer indices from the saved tensor dump file."""
        matching_files = glob.glob(file_pattern)
        model_layers_list = []

        if matching_files:
            tensor_file_path = matching_files[0]
            tensor_data = torch.load(
                tensor_file_path + "/" + file_name, map_location="cpu"
            )

            for idx, key in enumerate(tensor_data.keys(), 1):
                print(f"{idx}. {key}")
                if "model.layers." in key:
                    model_layers_list.append(key.split(".")[2])

        model_layers_list = sorted(set(int(x) for x in model_layers_list))
        print(model_layers_list)
        return model_layers_list


class TestDebugTensorDumpOutputFolder(
    TestDebugTensorDumpOutputFolderBase, CustomTestCase
):
    """
    Testcase： Verify that tensor dumps are generated for all layers when no specific layers are specified.
    """

    def test_debug_tensor_dump_output_folder(self):
        """Test that tensor dumps are generated for all 64 layers across two pipeline stages."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), TP_DIR_NUM)

        model_layers_list = self.get_layers_from_tensor_file(FILE_PATTERN_PP0)
        # Keep it consistent with num_hidden_layers in the model's config.json file.
        self.assertEqual(len(model_layers_list), 64)
        self.assertEqual(model_layers_list, list(range(64)))


class TestDumpLayersSingle(TestDebugTensorDumpOutputFolderBase, CustomTestCase):
    """
    Testcase： Verify that tensor dumps are generated only for a single specified layer.
    """

    other_args = [
        "--debug-tensor-dump-layers",
        "1",
    ]

    def test_dump_layers_single(self):
        """Test that tensor dumps are generated only for layer 1."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), TP_DIR_NUM)

        model_layers_list = self.get_layers_from_tensor_file(FILE_PATTERN_PP0)
        self.assertEqual(len(model_layers_list), 1)
        self.assertEqual(model_layers_list[0], 1)


class TestDumpLayersMultiple(TestDebugTensorDumpOutputFolderBase, CustomTestCase):
    """
    Testcase： Verify that tensor dumps are generated for multiple consecutive layers.
    """

    other_args = [
        "--debug-tensor-dump-layers",
        "2",
        "3",
        "4",
    ]

    def test_dump_layers_multiple(self):
        """Test that tensor dumps are generated for layers 2, 3, and 4."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), TP_DIR_NUM)

        model_layers_list = self.get_layers_from_tensor_file(FILE_PATTERN_PP0)
        self.assertEqual(len(model_layers_list), 3)
        self.assertEqual(model_layers_list[0], 2)
        self.assertEqual(model_layers_list[1], 3)
        self.assertEqual(model_layers_list[2], 4)


class TestDumpLayersNonConsecutiveLayers(
    TestDebugTensorDumpOutputFolderBase, CustomTestCase
):
    """
    Testcase： Verify that tensor dumps are generated for multiple non-consecutive layers.
    """

    other_args = [
        "--debug-tensor-dump-layers",
        "0",
        "5",
        "10",
    ]

    def test_dump_layers_non_consecutive_layers(self):
        """Test that tensor dumps are generated for layers 0, 5, and 10."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), TP_DIR_NUM)

        model_layers_list = self.get_layers_from_tensor_file(FILE_PATTERN_PP0)
        self.assertEqual(len(model_layers_list), 3)
        self.assertEqual(model_layers_list[0], 0)
        self.assertEqual(model_layers_list[1], 5)
        self.assertEqual(model_layers_list[2], 10)


class TestDumpLayersOutOFRange(TestDebugTensorDumpOutputFolderBase, CustomTestCase):
    """
    Testcase： Verify that no tensor dumps are generated when an out-of-range layer is specified.
    """

    other_args = [
        "--debug-tensor-dump-layers",
        "500",
    ]

    def test_dump_layers_out_of_range(self):
        """Test that no tensor dumps are generated when specifying layer 500 (out of range)."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), TP_DIR_NUM)

        model_layers_list = self.get_layers_from_tensor_file(FILE_PATTERN_PP0)
        self.assertEqual(len(model_layers_list), 0)


class TestDebugTensorNoDumpOutputFolder(
    TestDebugTensorDumpOutputFolderBase, CustomTestCase
):
    """
    Testcase： Verify that no tensor dump directories are created when --debug-tensor-dump-output-folder is not specified.
    """

    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        TP_SIZE,
        "--debug-tensor-dump-layers",
        "1",
        "--skip-server-warmup",
    ]

    def test_no_dump_output_folder(self):
        """Test that no tensor dump directories are created when output folder is not specified."""
        response, res = self.sending_request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(res), 0)


if __name__ == "__main__":
    unittest.main()
