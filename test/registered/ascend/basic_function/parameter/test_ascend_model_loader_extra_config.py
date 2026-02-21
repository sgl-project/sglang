import json
import os
import unittest
from abc import ABC
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="run failed",
)

MULTITHREAD_OUT_LOG = "./multi_thread_out_log.txt"
MULTITHREAD_ERR_LOG = "./multi_thread_err_log.txt"
CHECKPOINT_OUT_LOG = "./checkpoint_out_log.txt"
CHECKPOINT_ERR_LOG = "./checkpoint_err_log.txt"


class BaseModelLoaderTest(ABC):
    """Test base class"""

    models = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
    ]
    out_file = None
    err_file = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        os.environ["HCCL_BUFFSIZE"] = "200"
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "24"
        os.environ["USE_VLLM_CUSTOM_ALLREDUCE"] = "1"
        os.environ["HCCL_EXEC_TIMEOUT"] = "200"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        os.environ["SGLANG_ENBLE_TORCH_COMILE"] = "1"
        os.environ["AUTO_USE_UC_MEMORY"] = "0"
        os.environ["P2P_HCCL_BUFFSIZE"] = "20"
        os.environ["SGLANG_IS_IN_CI"] = "False"
        env = os.environ.copy()

        # Start the service first to prevent caching from affecting model load time.
        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
            env=env,
        )
        kill_process_tree(cls.process.pid)

        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
            env=env,
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

        if hasattr(cls, "out_file") and cls.out_file:
            cls.out_file.close()
        if hasattr(cls, "err_file") and cls.err_file:
            cls.err_file.close()


class TestNOModelLoaderExtraConfig(BaseModelLoaderTest, CustomTestCase):
    """Get the model loading time when the --model-loader-extra-config parameter is not configured."""

    log_info = "Loading safetensors"
    out_file = open(CHECKPOINT_OUT_LOG, "w+", encoding="utf-8")
    err_file = open(CHECKPOINT_ERR_LOG, "w+", encoding="utf-8")

    def test_no_model_loader_extra_config(self):
        self.err_file.seek(0)
        content = self.err_file.read()
        # "When the --model-loader-extra-config parameter is not configured, the startup log contains the 'Loading safetensors' string."
        self.assertIn(self.log_info, content)


class TestModelLoaderExtraConfig(BaseModelLoaderTest, CustomTestCase):
    """Testcase: After configuring the --model-loader-extra-config parameter, the model loading time will be shortened.

    [Test Category] Parameter
    [Test Target] --model-loader-extra-config
    """

    accuracy = 0.5
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 2}),
    ]
    log_info = "Multi-thread"
    out_file = open(MULTITHREAD_OUT_LOG, "w+", encoding="utf-8")
    err_file = open(MULTITHREAD_ERR_LOG, "w+", encoding="utf-8")

    def test_model_loader_extra_config(self):
        self.err_file.seek(0)
        content = self.err_file.read()
        # "When the --model-loader-extra-config parameter is configured, the startup log contains the 'Multi-thread' string."
        self.assertIn(self.log_info, content)

    def test_model_loading_time_reduced(self):
        def get_loading_seconds(filename, pattern):
            cmd = f"grep '{pattern}' {filename} | tail -1"
            line = run_command(cmd)
            if not line:
                return 0
            line = line.strip()
            print(f"{pattern}: {line}")
            if not line:
                return 0
            mm_ss = line.split("[")[1].split("<")[0]
            m, s = map(int, mm_ss.split(":"))
            return m * 60 + s

        # Get loading times
        multi_thread_seconds = get_loading_seconds(
            MULTITHREAD_ERR_LOG, "Multi-thread loading shards"
        )
        checkpoint_seconds = get_loading_seconds(
            CHECKPOINT_ERR_LOG, "Loading safetensors checkpoint shards"
        )

        print(
            f"Multi-thread: {multi_thread_seconds}s, Loading safetensors: {checkpoint_seconds}s."
        )

        # "Model loading time is reduced."
        self.assertGreater(checkpoint_seconds, multi_thread_seconds)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.models} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestNOModelLoaderExtraConfig("test_no_model_loader_extra_config"))
    suite.addTest(TestModelLoaderExtraConfig("test_model_loader_extra_config"))
    suite.addTest(TestModelLoaderExtraConfig("test_model_loading_time_reduced"))
    suite.addTest(TestModelLoaderExtraConfig("test_gsm8k"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
