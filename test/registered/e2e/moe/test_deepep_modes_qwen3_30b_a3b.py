"""DeepEP normal / low-latency / auto dispatch on Qwen3-30B-A3B (TP=EP=4), any device.

The same file runs on NVIDIA CUDA (deep_ep) and Intel XPU (deep_ep_xpu from
applications.ai.gpu.deepsymm, intranode PCIe P2P). Every class shares one server
configuration and differs only in --deepep-mode, plus a TP=4 baseline without
DeepEP, so results compare across modes and across devices.

Each class scores GSM8K with sgl-eval and runs a serving benchmark. Both print a
DEEPEP_MODES_RESULT {json} line (device, mode, metrics) for cross-comparison:

    python test/registered/e2e/moe/test_deepep_modes_qwen3_30b_a3b.py 2>&1 \
        | grep DEEPEP_MODES_RESULT

XPU: ZE_AFFINITY_MASK=0,1,2,3 keeps all four ranks on one socket.
"""

import json
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.benchmark.serving import run_benchmark
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kits.perf_bench_kit import check_perf, reported
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    get_benchmark_args,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=1800, stage="nightly", runner_config="4-gpu-h100")
# applications.ai.gpu.deepsymm is not yet in the public XPU CI image.
register_xpu_ci(
    est_time=1800,
    suite="nightly-xpu-kernel-main-4-gpu",
    nightly=True,
    disabled="applications.ai.gpu.deepsymm (deep_ep_xpu) not installed in XPU CI image",
)

MODEL = "Qwen/Qwen3-30B-A3B"
DEVICE = auto_config_device()

# Low-latency dispatch caps tokens per rank, and in low_latency mode prefill goes
# through it too. An a2a MoE layer sees the batch scattered over the TP ranks, so a
# 1024-token chunk is 256 per rank; keep headroom above that.
NUM_MAX_DISPATCH_TOKENS_PER_RANK = 512
CHUNKED_PREFILL_SIZE = 1024

COMMON_ARGS = [
    "--trust-remote-code",
    "--dtype",
    "bfloat16",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
    "--max-total-tokens",
    "65536",
    "--mem-fraction-static",
    "0.8",
    "--disable-radix-cache",
    # Non-thinking mode keeps GSM8K short enough for one eval per mode.
    "--default-chat-template-kwargs",
    '{"enable_thinking": false}',
]
DEVICE_ARGS = {
    "xpu": [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--disable-overlap-schedule",
    ],
    "cuda": [],
}
# BF16 experts behind DeepEP: DeepGEMM on CUDA, sgl-kernel-xpu on XPU (the default).
DEEPEP_DEVICE_ARGS = {
    "xpu": [],
    "cuda": ["--moe-runner-backend", "deep_gemm"],
}


def _device_name() -> str:
    module = getattr(torch, DEVICE, None)
    if module is not None and module.is_available():
        return module.get_device_name(0)
    return DEVICE


def _report(test_case, kind: str, metrics: dict) -> None:
    result = {
        "device": DEVICE,
        "device_name": _device_name(),
        "model": MODEL,
        "deepep_mode": test_case.deepep_mode or "none",
        "kind": kind,
        **metrics,
    }
    print(f"DEEPEP_MODES_RESULT {json.dumps(result)}", flush=True)


class _DeepEPModeMixin:
    # None runs the TP=4 baseline without DeepEP.
    deepep_mode = None

    gsm8k_num_examples = 200
    gsm8k_score_threshold = 0.85

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = COMMON_ARGS + DEVICE_ARGS[DEVICE]
        if cls.deepep_mode is not None:
            other_args += [
                "--ep-size",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                cls.deepep_mode,
                "--deepep-dispatcher-output-dtype",
                "bf16",
            ] + DEEPEP_DEVICE_ARGS[DEVICE]
        env = {
            **os.environ,
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": str(
                NUM_MAX_DISPATCH_TOKENS_PER_RANK
            ),
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        metrics = run_sgl_eval(
            SimpleNamespace(
                eval_name="gsm8k",
                base_url=self.base_url,
                model=self.model,
                num_examples=self.gsm8k_num_examples,
                num_threads=32,
                max_tokens=2048,
                # Qwen3 recommended non-thinking sampling.
                temperature=0.7,
                top_p=0.8,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        _report(
            self,
            "accuracy",
            {
                "gsm8k_score": metrics["score"],
                "gsm8k_num_examples": self.gsm8k_num_examples,
                "eval_output_throughput": metrics["output_throughput"],
            },
        )
        self.assertGreaterEqual(metrics["score"], self.gsm8k_score_threshold)

    def test_bench_serving(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=64,
            random_input_len=1024,
            random_output_len=256,
            max_concurrency=16,
            device=DEVICE,
        )
        res = run_benchmark(args)
        self.assertEqual(res["completed"], args.num_prompts)
        metrics = {
            key: res[key]
            for key in (
                "output_throughput",
                "input_throughput",
                "median_ttft_ms",
                "median_tpot_ms",
                "median_itl_ms",
                "median_e2e_latency_ms",
            )
        }
        _report(self, "bench_serving", metrics)
        check_perf(
            self,
            reported("output_throughput", metrics["output_throughput"], unit="tok/s"),
            reported("median_ttft_ms", metrics["median_ttft_ms"], unit="ms"),
            reported("median_tpot_ms", metrics["median_tpot_ms"], unit="ms"),
            suffix=f"_{self.deepep_mode or 'baseline'}_{DEVICE}",
        )


class TestBaselineTP4(_DeepEPModeMixin, CustomTestCase):
    deepep_mode = None


class TestDeepEPNormal(_DeepEPModeMixin, CustomTestCase):
    deepep_mode = "normal"


class TestDeepEPLowLatency(_DeepEPModeMixin, CustomTestCase):
    deepep_mode = "low_latency"


class TestDeepEPAuto(_DeepEPModeMixin, CustomTestCase):
    deepep_mode = "auto"


if __name__ == "__main__":
    unittest.main(verbosity=3)
