import unittest
from types import SimpleNamespace
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=96, stage="extra-a", runner_config="2-gpu-large")

MODEL_NAME = "31B"
TARGET_PATH = "google/gemma-4-31B-it"
DRAFT_PATH = "z-lab/gemma-4-31B-it-DFlash"
TENSOR_PARALLEL_SIZE = 2

DRAFT_ATTENTION_BACKEND = "flashinfer"
SPECULATIVE_NUM_DRAFT_TOKENS = 16
GSM8K_NUM_EXAMPLES = 200
GSM8K_NUM_THREADS = 128
SERVER_LAUNCH_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3

# Match the existing Gemma4 31B MTP accuracy floor.
GSM8K_SCORE_THRESHOLD = 0.75
ACCEPT_LENGTH_THRESHOLD = 5.4


def get_server_info(base_url: str) -> dict:
    response = requests.get(base_url + "/server_info", timeout=10)
    response.raise_for_status()
    return response.json()


def get_avg_spec_accept_length(base_url: str) -> Optional[float]:
    try:
        info = get_server_info(base_url)
    except Exception:
        return None
    internal_states = info.get("internal_states") or []
    if not internal_states:
        return None
    value = internal_states[0].get("avg_spec_accept_length")
    if value is None:
        return None
    return float(value)


class TestGemma4DFlash31B(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _common_server_args(cls) -> list[str]:
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.55",
            "--max-running-requests",
            "16",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "32768",
            "--skip-server-warmup",
        ]
        if TENSOR_PARALLEL_SIZE > 1:
            args += ["--tp-size", str(TENSOR_PARALLEL_SIZE)]
        return args

    @classmethod
    def _server_args(cls) -> list[str]:
        return [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            DRAFT_PATH,
            "--speculative-num-draft-tokens",
            str(SPECULATIVE_NUM_DRAFT_TOKENS),
            "--speculative-draft-attention-backend",
            DRAFT_ATTENTION_BACKEND,
        ] + cls._common_server_args()

    @classmethod
    def _gsm8k_args(cls) -> SimpleNamespace:
        return SimpleNamespace(
            base_url=cls.base_url,
            model=TARGET_PATH,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=GSM8K_NUM_THREADS,
            num_shots=5,
        )

    @staticmethod
    def _stop_process(process) -> None:
        try:
            kill_process_tree(process.pid)
        except Exception:
            pass

    def test_gsm8k_dflash(self) -> None:
        process = None
        try:
            process = popen_launch_server(
                TARGET_PATH,
                self.base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=self._server_args(),
            )
            requests.get(self.base_url + "/flush_cache", timeout=30)

            server_info = get_server_info(self.base_url)
            self.assertEqual(
                server_info.get("speculative_algorithm"),
                "DFLASH",
                f"{MODEL_NAME}: server did not start with DFLASH",
            )
            self.assertEqual(
                server_info.get("speculative_draft_attention_backend"),
                DRAFT_ATTENTION_BACKEND,
                f"{MODEL_NAME}: unexpected DFLASH draft attention backend",
            )
            self.assertEqual(
                server_info.get("speculative_num_draft_tokens"),
                SPECULATIVE_NUM_DRAFT_TOKENS,
                f"{MODEL_NAME}: unexpected DFLASH block size",
            )
            self.assertFalse(
                bool(server_info.get("disable_cuda_graph")),
                f"{MODEL_NAME}: CUDA graph is disabled",
            )

            metrics = run_eval(self._gsm8k_args())
            dflash_score = float(metrics["score"])
            avg_accept = get_avg_spec_accept_length(self.base_url)
        finally:
            if process is not None:
                self._stop_process(process)

        print(
            f"[Gemma4 {MODEL_NAME} DFlash] "
            f"score={dflash_score:.4f} threshold={GSM8K_SCORE_THRESHOLD:.4f} "
            f"avg_spec_accept_length={avg_accept}"
        )
        if is_in_ci():
            write_github_step_summary(
                f"### Gemma4 {MODEL_NAME} DFlash\n"
                f"score={dflash_score:.4f}\n"
                f"threshold={GSM8K_SCORE_THRESHOLD:.4f}\n"
                f"avg_spec_accept_length={avg_accept}\n"
            )

        self.assertGreaterEqual(dflash_score, GSM8K_SCORE_THRESHOLD)
        self.assertIsNotNone(avg_accept)
        self.assertGreaterEqual(
            avg_accept,
            ACCEPT_LENGTH_THRESHOLD,
            f"{MODEL_NAME}: accept length too low",
        )


if __name__ == "__main__":
    unittest.main()
