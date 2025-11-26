import os
import subprocess
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None


class TestQwen3_32B(CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
    dataset = (
        "/root/.cache/modelscope/hub/datasets/Howeee/GSM8K-in1500-bs1536/test.jsonl"
    )
    accuracy = 0.05

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--trust-remote-code",
                "--nnodes",
                "1",
                "--node-rank",
                "0",
                "--quantization",
                "w8a8_int8",
                "--max-running-requests",
                "68",
                "--context-length",
                "8192",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "32768",
                "--max-prefill-tokens",
                "28000",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--cuda-graph-bs",
                "4",
                "8",
                "16",
                "68",
                "--tp-size",
                "4",
            ]
            if is_npu()
            else []
        )
        if is_npu():
            env = os.environ.copy()
            env.update(
                {
                    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                    "STREAMS_PER_DEVICE": "32",
                }
            )
        else:
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_k8s(self):
        # args = SimpleNamespace(
        #     num_shots=5,
        #     data_path=None,
        #     num_questions=200,
        #     max_new_tokens=512,
        #     parallel=128,
        #     host="http://127.0.0.1",
        #     port=int(self.base_url.split(":")[-1]),
        # )
        # metrics = run_eval(args)
        # self.assertGreater(
        #     metrics["accuracy"],
        #     self.accuracy,
        #     f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        # )
        port = self.base_url.split(":")[-1]
        run_command("rm -rf ./benchmark")
        run_command("pip3 install nltk==3.8")
        run_command("git clone https://gitee.com/aisbench/benchmark.git")
        run_command(
            f'sed -i \'s#path="[^"]*"#path="{self.model}"#\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            'sed -i \'s/model="[^"]*"/model="Qwen3"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            "sed -i 's/request_rate = [^\"]*/request_rate = 5.5,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            'sed -i \'s/host_ip = "[^"]*"/host_ip = "127.0.0.1"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            f"sed -i 's/host_port = [^\"]*/host_port = {port},/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            "sed -i 's/max_out_len = [^\"]*/max_out_len = 1500,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            "sed -i 's/batch_size=[^\"]*/batch_size=410,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            r"""sed -i '/generation_kwargs = dict(/,/),/c\        generation_kwargs = dict(\n            temperature = 0,\n            ignore_eos = True,\n        ),'  ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"""
        )
        run_command("mkdir ./benchmark/ais_bench/datasets/gsm8k")
        run_command(f"\cp {self.dataset} ./benchmark/ais_bench/datasets/gsm8k/")
        run_command("touch ./benchmark/ais_bench/datasets/gsm8k/train.jsonl")
        ais_res = run_command("pip3 install -e ./benchmark/")
        print(str(ais_res))
        cat_res = run_command(
            "cat ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        print("cat_res is " + str(cat_res))
        metrics = run_command(
            "ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf | tee ./gsm8k_deepseek_log.txt"
        )
        print("metrics is " + str(metrics))
        tpot = run_command(
            "cat ./gsm8k_deepseek_log.txt | grep TPOT | awk '{print $6}'"
        )
        output_token_throughput = run_command(
            "cat ./gsm8k_deepseek_log.txt | grep 'Output Token Throughput' | awk '{print $8}'"
        )
        self.assertGreaterEqual(
            float(output_token_throughput),
            1109,
        )
        self.assertLessEqual(
            float(tpot),
            55,
        )


if __name__ == "__main__":
    unittest.main()
