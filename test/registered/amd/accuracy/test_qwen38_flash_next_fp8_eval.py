"""AMD Qwen3.8-Flash-Next-FP8 graph-mode GSM8K accuracy test.

The released FP8 checkpoint uses TP1 on gfx950 and TP2+EP2 on gfx942, where
the 192 GiB device capacity cannot hold the target, MTP draft, and graph/KV
state on one GPU. These nightly tests pin the checkpoint revision and exercise
the same AITER decode-graph path with EAGLE speculation on both architectures.
Direct AITER paged QSA remains disabled here so the core model correctness gate
does not depend on an unreleased AITER API.

Registries:
  nightly-amd-2-gpu-mi30x-qwen38-flash-next-fp8 suite
  nightly-amd-1-gpu-mi35x-qwen38-flash-next-fp8 suite
"""

import base64
import io
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    write_github_step_summary,
)

register_amd_ci(
    est_time=1800,
    suite="nightly-amd-2-gpu-mi30x-qwen38-flash-next-fp8",
    nightly=True,
)
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-1-gpu-mi35x-qwen38-flash-next-fp8",
    nightly=True,
)

# The checkpoint declares model_type "qwen4_exp", which only resolves once the
# model-support PR has landed. Without this guard the nightly turns red every
# night on a missing dependency rather than on an accuracy regression.
try:
    from sglang.srt.configs import Qwen4ExpConfig  # noqa: F401

    QWEN4_EXP_SUPPORTED = True
except ImportError:
    QWEN4_EXP_SUPPORTED = False

MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
MODEL_PATH = os.environ.get("QWEN38_FLASH_NEXT_FP8_MODEL_PATH", MODEL_ID)
MODEL_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
ACCURACY_THRESHOLD = 0.94
SERVER_LAUNCH_TIMEOUT = 1800
NUM_REQUESTED_EXAMPLES = 1319
NUM_SHOTS = 5
NUM_EVALUATED_EXAMPLES = NUM_REQUESTED_EXAMPLES - NUM_SHOTS
IMAGE_PATH = Path(__file__).resolve().parents[4] / "examples/assets/example_image.png"
TP_SIZE = int(os.environ.get("QWEN38_FLASH_NEXT_FP8_TP_SIZE", "1"))
EP_SIZE = int(os.environ.get("QWEN38_FLASH_NEXT_FP8_EP_SIZE", "1"))
MEM_FRACTION_STATIC = os.environ.get(
    "QWEN38_FLASH_NEXT_FP8_MEM_FRACTION_STATIC", "0.95"
)

if TP_SIZE % EP_SIZE != 0:
    raise ValueError(
        f"Qwen3.8 nightly requires TP ({TP_SIZE}) divisible by EP ({EP_SIZE})"
    )

SERVER_ARGS = [
    "--revision",
    MODEL_REVISION,
    "--served-model-name",
    MODEL_ID,
    "--tp-size",
    str(TP_SIZE),
    "--ep-size",
    str(EP_SIZE),
    "--attention-backend",
    "aiter",
    "--moe-runner-backend",
    "aiter",
    "--kv-cache-dtype",
    "auto",
    "--chunked-prefill-size",
    "16384",
    "--watchdog-timeout",
    "1200",
    "--mem-fraction-static",
    MEM_FRACTION_STATIC,
    "--max-running-requests",
    "4",
    "--cuda-graph-max-bs-decode",
    "4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

SERVER_ENV = {
    "SGLANG_USE_AITER": "0",
}


@unittest.skipUnless(
    QWEN4_EXP_SUPPORTED,
    "Qwen3.8-Flash-Next model support is not in this build "
    "(sglang.srt.configs.Qwen4ExpConfig is missing)",
)
class TestQwen38FlashNextFP8AMD(CustomTestCase):
    """Gate Qwen3.8-Flash-Next FP8 accuracy on the graph fallback path."""

    def _assert_multimodal_generation(self):
        image_data = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "What color are the taxis? Answer with one word.",
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        self.assertIn("yellow", content.lower())

    def test_gsm8k_accuracy(self):
        server_stdout = io.StringIO()
        server_stderr = io.StringIO()
        process = popen_launch_server(
            MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=SERVER_ENV,
            return_stdout_stderr=(server_stdout, server_stderr),
        )

        try:
            self._assert_multimodal_generation()
            requests.post(
                DEFAULT_URL_FOR_TEST + "/flush_cache",
                params={"timeout": 30},
                timeout=45,
            ).raise_for_status()
            args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                model=MODEL_ID,
                eval_name="gsm8k",
                num_examples=NUM_REQUESTED_EXAMPLES,
                num_threads=512,
                num_shots=NUM_SHOTS,
                max_tokens=4096,
                chat_template_kwargs={"enable_thinking": False},
            )
            metrics = run_eval(args)
            score = metrics["score"]
            latency = metrics.get("latency", 0.0)
            output_throughput = metrics.get("output_throughput", 0.0)
            status = "PASS" if score >= ACCURACY_THRESHOLD else "FAIL"

            self.assertIsNone(
                process.poll(), "Qwen3.8-Flash-Next server exited during evaluation"
            )
            server_logs = server_stdout.getvalue() + server_stderr.getvalue()
            decode_lines = [
                line for line in server_logs.splitlines() if "Decode batch" in line
            ]
            self.assertTrue(
                decode_lines,
                "Qwen3.8-Flash-Next completed without decode batch evidence",
            )
            eager_decode_lines = [
                line for line in decode_lines if "cuda graph: True" not in line
            ]
            self.assertFalse(
                eager_decode_lines,
                "Qwen3.8-Flash-Next used eager decode instead of graph replay: "
                + "\n".join(eager_decode_lines[:5]),
            )
            self.assertNotRegex(
                server_logs,
                r"Parameter .*not found in params_dict",
                "Qwen3.8-Flash-Next skipped checkpoint parameters during loading",
            )

            if is_in_ci():
                summary = "### Qwen3.8-Flash-Next-FP8 GSM8K (AMD)\n\n"
                summary += (
                    "| Model revision | TP | EP | Graph | Responses | Accuracy | "
                    "Threshold | Latency | Output throughput | Status |\n"
                )
                summary += "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
                summary += (
                    f"| `{MODEL_REVISION}` | {TP_SIZE} | {EP_SIZE} | yes | "
                    f"{NUM_EVALUATED_EXAMPLES} | "
                    f"{score:.3f} | {ACCURACY_THRESHOLD:.2f} | {latency:.1f}s | "
                    f"{output_throughput:.1f} tok/s | {status} |\n"
                )
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                score,
                ACCURACY_THRESHOLD,
                f"Qwen3.8-Flash-Next-FP8 accuracy {score:.3f} below "
                f"threshold {ACCURACY_THRESHOLD:.2f}",
            )
        finally:
            terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
