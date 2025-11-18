"""
Basic tests for KT-kernel integration

Tests basic inference workflow with different GPU configurations.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

from .utils import (
    TEST_PROMPTS,
    get_kt_env,
    get_kt_model_paths,
    get_kt_server_args,
    run_inference,
)


class TestKTBasic1GPU(CustomTestCase):
    """
    Test basic inference workflow with 1 GPU configuration

    Configuration:
    - tensor_parallel_size: 1
    - kt_num_gpu_experts: 1
    - kt_cpuinfer: 60
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30000"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=1,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=1,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],  # Only show errors
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_inference_workflow(self):
        """Test basic inference with 5 prompts"""
        prompts = TEST_PROMPTS[:5]
        max_tokens = 50

        outputs = run_inference(
            self.base_url, prompts, max_tokens=max_tokens, temperature=0.0
        )

        # Validate responses
        self.assertEqual(
            len(outputs),
            len(prompts),
            "Number of outputs doesn't match number of prompts",
        )

        for i, output in enumerate(outputs):
            self.assertIsInstance(output, str, f"Output {i} is not a string")
            self.assertGreater(len(output), 0, f"Output {i} is empty")
            print(f"✓ Prompt {i+1}: {len(output)} chars generated")


class TestKTBasic4GPU(CustomTestCase):
    """
    Test basic inference workflow with 4 GPU configuration

    Configuration:
    - tensor_parallel_size: 4
    - kt_num_gpu_experts: 80
    - kt_cpuinfer: 60
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30001"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=80,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=4,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],  # Only show errors
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_inference_workflow(self):
        """Test basic inference with 5 prompts"""
        prompts = TEST_PROMPTS[:5]
        max_tokens = 50

        outputs = run_inference(
            self.base_url, prompts, max_tokens=max_tokens, temperature=0.0
        )

        # Validate responses
        self.assertEqual(
            len(outputs),
            len(prompts),
            "Number of outputs doesn't match number of prompts",
        )

        for i, output in enumerate(outputs):
            self.assertIsInstance(output, str, f"Output {i} is not a string")
            self.assertGreater(len(output), 0, f"Output {i} is empty")
            print(f"✓ Prompt {i+1}: {len(output)} chars generated")


class TestKTBasic8GPU(CustomTestCase):
    """
    Test basic inference workflow with 8 GPU configuration

    Configuration:
    - tensor_parallel_size: 8
    - kt_num_gpu_experts: 200
    - kt_cpuinfer: 60
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30002"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=200,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=8,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],  # Only show errors
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_inference_workflow(self):
        """Test basic inference with 5 prompts"""
        prompts = TEST_PROMPTS[:5]
        max_tokens = 50

        outputs = run_inference(
            self.base_url, prompts, max_tokens=max_tokens, temperature=0.0
        )

        # Validate responses
        self.assertEqual(
            len(outputs),
            len(prompts),
            "Number of outputs doesn't match number of prompts",
        )

        for i, output in enumerate(outputs):
            self.assertIsInstance(output, str, f"Output {i} is not a string")
            self.assertGreater(len(output), 0, f"Output {i} is empty")
            print(f"✓ Prompt {i+1}: {len(output)} chars generated")


if __name__ == "__main__":
    unittest.main()
