import unittest
from types import SimpleNamespace
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=128, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=450, suite="stage-b-test-1-gpu-large-amd")


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


class TestFrozenKVMTP(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _common_server_args(cls) -> list[str]:
        args = [
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
        return args

    @classmethod
    def _server_args(cls, topk: int) -> list[str]:
        return [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-draft-model-path",
            "google/gemma-4-E4B-it-assistant",
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            str(topk),
            "--speculative-num-draft-tokens",
            str({1: 6, 3: 12}[topk]),
        ] + cls._common_server_args()

    @classmethod
    def _gsm8k_args(cls) -> SimpleNamespace:
        return SimpleNamespace(
            base_url=cls.base_url,
            model="google/gemma-4-E4B-it",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )

    @staticmethod
    def _stop_process(process) -> None:
        try:
            kill_process_tree(process.pid)
        except Exception:
            pass

    def _run_gsm8k_mtp(self, topk: int) -> None:
        process = None
        try:
            process = popen_launch_server(
                "google/gemma-4-E4B-it",
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
                other_args=self._server_args(topk),
            )
            requests.get(self.base_url + "/flush_cache", timeout=30)

            server_info = get_server_info(self.base_url)
            self.assertEqual(
                server_info.get("speculative_eagle_topk"),
                topk,
                f"E4B: server did not start with topk={topk}",
            )
            self.assertFalse(
                bool(server_info.get("disable_cuda_graph")),
                f"E4B/topk{topk}: CUDA graph is disabled",
            )

            metrics = run_eval(self._gsm8k_args())
            mtp_score = float(metrics["score"])
            avg_accept = get_avg_spec_accept_length(self.base_url)
        finally:
            if process is not None:
                self._stop_process(process)

        print(
            f"[Frozen-KV MTP E4B topk={topk}] "
            f"score={mtp_score:.4f} threshold={0.65:.4f} "
            f"avg_spec_accept_length={avg_accept}"
        )
        if is_in_ci():
            write_github_step_summary(
                f"### Frozen-KV MTP E4B topk={topk}\n"
                f"score={mtp_score:.4f}\n"
                f"threshold={0.65:.4f}\n"
                f"avg_spec_accept_length={avg_accept}\n"
            )

        self.assertGreaterEqual(mtp_score, 0.65)
        self.assertIsNotNone(avg_accept)
        self.assertGreaterEqual(
            avg_accept,
            1.5,
            f"E4B/topk{topk}: accept length too low",
        )

    def test_gsm8k_mtp(self) -> None:
        for topk in (1, 3):
            with self.subTest(topk=topk):
                self._run_gsm8k_mtp(topk)


if __name__ == "__main__":
    unittest.main()
