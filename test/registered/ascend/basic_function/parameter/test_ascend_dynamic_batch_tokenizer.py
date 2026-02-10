import unittest
import logging
import requests  
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "4",
    "--enable-dynamic-batch-tokenizer",
    "--dynamic-batch-tokenizer-batch-size", "4",
    "--dynamic-batch-tokenizer-batch-timeout", "0", 
    "--log-level", "debug"
]
MODEL_NAME = QWEN3_32B_WEIGHTS_PATH

def launch_server_with_tokenizer_timeout(model_name, base_url, tokenizer_timeout, other_args_base):
    other_args = other_args_base.copy()
    if "--dynamic-batch-tokenizer-batch-timeout" in other_args:
        idx = other_args.index("--dynamic-batch-tokenizer-batch-timeout") + 1
        other_args[idx] = str(tokenizer_timeout)
    
    process = popen_launch_server(
        model_name,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 
        other_args=other_args,
    )
    return process

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class BaseQwenTest(CustomTestCase):
    # Base test class for Qwen3-32B model accuracy validation on Ascend backend
    accuracy = 0.86

    def _run_gsm8k_test(self, scenario):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)

        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'accuracy {metrics["accuracy"]} < {self.accuracy}',
        )
        
        server_info = requests.get(self.base_url + "/get_server_info")
        logger.info(f"{scenario}: server_info={server_info}")

class TestQwenPPTieWeightsAccuracyTokenizerTimeout0(BaseQwenTest):
    """Testcase: Verify Qwen3-32B model accuracy on GSM8K with dynamic batch tokenizer timeout set to 0.

    [Test Category] Parameter
    [Test Target] --dynamic-batch-tokenizer-batch-timeout;--enable-dynamic-batch-tokenizer;--dynamic-batch-tokenizer-batch-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=0, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_0(self):
        self._run_gsm8k_test("tokenizer_timeout=0")

class TestQwenPPTieWeightsAccuracyTokenizerTimeout1(BaseQwenTest):
    """Testcase: Verify Qwen3-32B model accuracy on GSM8K with dynamic batch tokenizer timeout set to 1.

    [Test Category] Parameter
    [Test Target] --dynamic-batch-tokenizer-batch-timeout;--enable-dynamic-batch-tokenizer;--dynamic-batch-tokenizer-batch-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=1, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_1(self):
        self._run_gsm8k_test("tokenizer_timeout=1")

if __name__ == "__main__":
    unittest.main()
