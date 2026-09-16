import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1200, stage="nightly", runner_config="4-gpu-b200")

NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

NEMOTRON_3_SUPER_NVFP4_ARGS = [
    "--tp-size",
    "4",
    "--trust-remote-code",
    "--reasoning-parser",
    "nemotron_3",
    "--tool-call-parser",
    "qwen3_coder",
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 17}',
]

DP_ATTENTION_EP_ARGS = [
    "--dp-size",
    "4",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--ep-size",
    "4",
    "--moe-a2a-backend",
    "flashinfer",
    "--moe-runner-backend",
    "flashinfer_cutedsl",
    "--mamba-full-memory-ratio",
    "5.0",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--attention-backend",
    "trtllm_mha",
    "--max-running-requests",
    "1024",
    "--mem-fraction-static",
    "0.93",
    "--max-prefill-tokens",
    "8192",
]

MTP_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--max-running-requests",
    "200",
    "--mem-fraction-static",
    "0.75",
]


def _run_gsm8k(test_case):
    args = SimpleNamespace(
        model=test_case.model,
        eval_name="gsm8k",
        num_shots=5,
        num_examples=200,
        max_tokens=16000,
        num_threads=200,
        repeat=1,
        temperature=1.0,
        top_p=0.95,
        base_url=test_case.base_url,
        host="http://127.0.0.1",
        port=int(test_case.base_url.split(":")[-1]),
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    test_case.assertGreaterEqual(metrics["score"], 0.96)


class TestNvidiaNemotron3SuperNVFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NEMOTRON_3_SUPER_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(0):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=NEMOTRON_3_SUPER_NVFP4_ARGS,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self)


class TestNvidiaNemotron3SuperNVFP4MTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NEMOTRON_3_SUPER_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(0):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=NEMOTRON_3_SUPER_NVFP4_ARGS + MTP_ARGS,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self)


class TestNvidiaNemotron3SuperNVFP4DPAttentionEP(CustomTestCase):
    """DP attention + EP with the FlashInfer one-sided A2A and CuteDSL MoE runner."""

    @classmethod
    def setUpClass(cls):
        cls.model = NEMOTRON_3_SUPER_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_ENABLE_ASYNC_ASSERT.override(0),
            envs.SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(4096),
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(1024 * 1024 * 1024),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=NEMOTRON_3_SUPER_NVFP4_ARGS + DP_ATTENTION_EP_ARGS,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self)


if __name__ == "__main__":
    unittest.main()
