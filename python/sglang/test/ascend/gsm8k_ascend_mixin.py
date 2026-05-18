import os
import subprocess
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    write_github_step_summary,
)


class GSM8KAscendMixin(ABC):
    model = ""

    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]
    server_cmd = ""
    gsm8k_num_shots = 5
    num_questions = 200

    env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
        "HCCL_BUFFSIZE": "200",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
        "USE_VLLM_CUSTOM_ALLREDUCE": "1",
        "HCCL_EXEC_TIMEOUT": "200",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_ENBLE_TORCH_COMILE": "1",
        "AUTO_USE_UC_MEMORY": "0",
        "P2P_HCCL_BUFFSIZE": "20",
    }

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout_for_server_launch,
                other_args=cls.other_args,
                env=cls.env,
            )
            cls.server_cmd = subprocess.list2cmdline(cls.process.args)
        except Exception as e:
            write_github_step_summary(f"Failed to launch server for {cls.model}: {e}")
            raise AssertionError(f"Test failed for {cls.model}: {e}")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        accuracy_threshold = getattr(self, "accuracy", 0.00)
        output_throughput_threshold = getattr(self, "output_throughput", 0.00)

        model_metrics = {
            "server": self.server_cmd,
            "client": "few_shot_gsm8k",
            "accuracy_threshold": getattr(self, "accuracy", "N/A"),
            "output_throughput_threshold": getattr(self, "output_throughput", "N/A"),
        }

        try:
            args = SimpleNamespace(
                max_tokens=512,
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                num_examples=200,
                num_threads=128,
                num_shots=self.gsm8k_num_shots,
            )
            metrics = run_eval(args)
            model_metrics["accuracy"] = metrics["accuracy"]
            model_metrics["output_throughput"] = metrics["output_throughput"]
            self.assertGreaterEqual(
                metrics["score"],
                accuracy_threshold,
                f'Accuracy of {self.model} is {str(metrics["score"])}, is lower than {accuracy_threshold}',
            )
            self.assertGreaterEqual(
                metrics["output_throughput"],
                output_throughput_threshold,
                f'Output throughput of {self.model} is {str(metrics["output_throughput"])}, is lower than {output_throughput_threshold}',
            )
        except Exception as e:
            model_metrics["error"] = e
            self.fail(f"Test failed for {self.model}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})
