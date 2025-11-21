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


class TestDeepseek_w8a8(CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
    accuracy = 0.05

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--tp",
                "16",
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "w8a8_int8",
                "--watchdog-timeout",
                "9000",
                "--host",
                "127.0.0.1",
                "--port",
                "6699",
                "--cuda-graph-bs",
                "8",
                "16",
                "24",
                "28",
                "32",
                "--mem-fraction-static",
                "0.87",
                "--max-running-requests",
                "128",
                "--context-length",
                "8188",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--max-prefill-tokens",
                "6000",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--enable-dp-attention",
                "--dp-size",
                "4",
                "--enable-dp-lm-head",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--dtype",
                "bfloat16",
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
                    "HCCL_SOCKET_IFNAME": "lo",
                    "GLOO_SOCKET_IFNAME": "lo",
                    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
                    "HCCL_BUFFSIZE": "850",
                    "SGLANG_USE_MLAPO": "1",
                    "SGLANG_USE_FIA_NZ": "1",
                    "ENABLE_MOE_NZ": "1",
                }
            )
            # run_command("pip uninstall modelscope")
        else:
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3000,
            other_args=other_args,
            env=env, 
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_k8s(self):
        run_command("rm -rf ./benchmark")
        run_command("pip3 install nltk==3.8")
        run_command("git clone https://gitee.com/aisbench/benchmark.git")
        run_command(
            'sed -i \'s#path="[^"]*"#path="/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-W8A8"#\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            'sed -i \'s/model="[^"]*"/model="DeepSeek-R1-W8A8"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            "sed -i 's/request_rate = [^\"]*/request_rate = 13.5,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            'sed -i \'s/host_ip = "[^"]*"/host_ip = "127.0.0.1"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            f"sed -i 's/host_port = [^\"]*/host_port = 6699,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            "sed -i 's/max_out_len = [^\"]*/max_out_len = 1500,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            "sed -i 's/batch_size=[^\"]*/batch_size=512,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            r"""sed -i '/generation_kwargs = dict(/,/),/c\        generation_kwargs = dict(\n            temperature = 0,\n            ignore_eos = True,\n        ),'  ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"""
        )
        run_command("mkdir ./benchmark/ais_bench/datasets/gsm8k")
        run_command(
            "\cp /root/.cache/modelscope/hub/datasets/vllm-ascend/GSM8K-in3500-bs2800/test.jsonl ./benchmark/ais_bench/datasets/gsm8k/"
        )
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
        print("tpot is " + str(tpot))
        print("output_token_throughput is " + str(output_token_throughput))
        # self.assertGreaterEqual(
        #     float(output_token_throughput),
        #     1109,
        # )
        # self.assertLessEqual(
        #     float(tpot),
        #     55,
        # )


if __name__ == "__main__":
    unittest.main()
