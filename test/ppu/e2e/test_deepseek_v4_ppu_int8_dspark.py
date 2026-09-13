import json
import os
import subprocess
import sys
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# 1. 从环境变量获取模型路径（优先读取 MODEL，兼容 MODEL_PATH）
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8"
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000  # 增大超时时间，防止 8 卡加载慢导致超时

# 从环境变量获取本地数据集路径
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepSeekV4EvalScope(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # 2. 完全替换为用户指定的启动参数
        other_args = [
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--cuda-graph-max-bs-decode",
            "512",
            "--disable-custom-all-reduce",
            "--disable-prefill-cuda-graph",
            "--disable-shared-experts-fusion",
            "--served-model-name",
            "DeepSeek-V4-Flash-0731",  # 指定服务模型名称
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--enable-metrics",
            "--enable-cache-report",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "DSPARK",  # 开启 DSPARK 投机
            "--speculative-num-steps",
            "1",
            "--speculative-num-draft-tokens",
            "6",
        ]

        print(f"[Server] 开始启动 SGLang 服务，加载模型: {MODEL_PATH}")
        # 自动拉起 SGLang 服务
        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        # 测试结束，清理 Server 进程树
        print("[Server] 评测结束，正在关闭 SGLang 服务...")
        kill_process_tree(cls.process.pid)

    def test_gsm8k_with_evalscope(self):
        """核心测试方法：使用 EvalScope 评测 GSM8K 准确率（与 CLI 参数完全对齐）"""

        dataset_args = {"gsm8k": {"filters": {"remove_until": "</think>"}}}

        dataset_args["gsm8k"]["local_path"] = GSM8K_DATA_PATH
        print(f"[EvalScope] 检测到本地数据集路径: {GSM8K_DATA_PATH}")

        generation_config = {
            "max_tokens": 384000,
            "temperature": 1,
            "top_p": 1,
            "n": 1,
            "extra_body": {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"thinking": True},
            },
        }

        eval_cmd = [
            "evalscope",
            "eval",
            "--model",
            "DeepSeek-V4-Flash-0731",
            "--api-url",
            f"{BASE_URL}/v1",
            "--api-key",
            "EMPTY",
            "--eval-type",
            "openai_api",
            "--datasets",
            "gsm8k",
            "--dataset-args",
            json.dumps(dataset_args),
            "--generation-config",
            json.dumps(generation_config),
            "--eval-batch-size",
            "64",
            "--work-dir",
            "outputs",
            "--timeout",
            "60000",
            "--stream",
        ]

        print(f"[EvalScope] 执行评测命令: {' '.join(eval_cmd)}")

        # 4. 执行评测
        result = subprocess.run(
            eval_cmd, env=os.environ, stdout=sys.stdout, stderr=sys.stderr, text=True
        )

        # 5. 断言评测过程正常结束（退出码为 0）
        self.assertEqual(
            result.returncode, 0, f"EvalScope 评测异常退出，退出码: {result.returncode}"
        )


if __name__ == "__main__":
    unittest.main()
