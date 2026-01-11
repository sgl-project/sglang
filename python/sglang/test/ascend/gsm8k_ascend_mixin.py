import os
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class GSM8KAscendMixin(ABC):
    model = ""
    accuracy = 0.00
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
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
        env = os.environ.copy()

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
