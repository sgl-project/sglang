"""Manual Gemma 4 MTP GSM8K validation.

Implements the validation plan in
``/workspace/gemma4_downloads/frozen_kv_mtp_design.md`` §10.2:

* config A (no MTP, baseline): launched once in :meth:`setUpClass` to
  record the calibrated target score.
* config C (MTP + CUDA graph, ``topk=1``): default cuda graph capture (target +
  ``FrozenKVMTPCudaGraphRunner`` for the recurrent draft loop), including
  normal padding to the nearest captured batch size.
* config D (MTP + ``topk=5``): tree-shaped verify path with the Frozen-KV
  draft loop running eager.

Pass criterion: GSM8K score in MTP configs is within
``gsm8k_score_drop_tolerance`` of A, and ``avg_spec_accept_length`` is above
``accept_length_threshold``.

Useful env vars (all optional):

* ``SGLANG_GEMMA4_MTP_SERVER_CUDA_VISIBLE_DEVICES``  — pin the server
  to a specific GPU set (default: env-inherited).
* ``SGLANG_GEMMA4_MTP_SERVER_MAX_RUNNING_REQUESTS``
* ``SGLANG_GEMMA4_MTP_SERVER_MAX_TOTAL_TOKENS``
* ``SGLANG_GEMMA4_MTP_GSM8K_EXAMPLES``
* ``SGLANG_GEMMA4_MTP_GSM8K_THREADS``
"""

from __future__ import annotations

import gc
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

SERVER_CUDA_VISIBLE_DEVICES_ENV = "SGLANG_GEMMA4_MTP_SERVER_CUDA_VISIBLE_DEVICES"
MAX_RUNNING_REQUESTS_ENV = "SGLANG_GEMMA4_MTP_SERVER_MAX_RUNNING_REQUESTS"
MAX_TOTAL_TOKENS_ENV = "SGLANG_GEMMA4_MTP_SERVER_MAX_TOTAL_TOKENS"
GSM8K_EXAMPLES_ENV = "SGLANG_GEMMA4_MTP_GSM8K_EXAMPLES"
GSM8K_THREADS_ENV = "SGLANG_GEMMA4_MTP_GSM8K_THREADS"


@dataclass(frozen=True)
class ModelPair:
    name: str
    target_path: str
    assistant_path: str
    gsm8k_score_drop_tolerance: float = 0.02
    accept_length_threshold: Optional[float] = None
    gsm8k_num_examples: int = 200
    gsm8k_num_threads: int = 128
    server_cuda_visible_devices: Optional[str] = None
    tensor_parallel_size: int = 1


def ensure_checkpoint(path: str, label: str) -> None:
    """Allow HF repo IDs (no local check) but validate local paths if present."""
    p = Path(path)
    if not p.exists():
        # Treat as HF Hub repo id.
        return
    required = ["model.safetensors", "config.json"]
    missing = [r for r in required if not (p / r).exists() and not list(p.glob(r))]
    if missing:
        raise FileNotFoundError(
            f"{label} checkpoint is incomplete at {path}. "
            "Expected config.json and model.safetensors."
        )


def get_avg_spec_accept_length(base_url: str) -> Optional[float]:
    try:
        info = requests.get(base_url + "/server_info", timeout=10).json()
    except Exception:
        return None
    internal_states = info.get("internal_states") or []
    if not internal_states:
        return None
    val = internal_states[0].get("avg_spec_accept_length")
    if val is None:
        return None
    return float(val)


def get_server_info(base_url: str) -> Dict:
    response = requests.get(base_url + "/server_info", timeout=10)
    response.raise_for_status()
    return response.json()


class Gemma4MTPGSM8KMixin:
    """GSM8K harness for no-MTP, topk=1 MTP, and topk=5 MTP.

    ``setUpClass`` launches the target-only baseline (config A) and records
    the score. The MTP tests launch separate servers, run GSM8K, assert the
    score did not drop, and tear down.
    """

    model_pair: ModelPair

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @classmethod
    def _server_env(cls) -> Optional[Dict[str, str]]:
        env: Dict[str, str] = dict(os.environ)
        env["SGLANG_ENABLE_SPEC_V2"] = "0"
        cuda_visible = (
            os.environ.get(SERVER_CUDA_VISIBLE_DEVICES_ENV)
            or cls.model_pair.server_cuda_visible_devices
        )
        if cuda_visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        return env

    @classmethod
    def _gsm8k_num_threads(cls) -> int:
        return int(os.getenv(GSM8K_THREADS_ENV, str(cls.model_pair.gsm8k_num_threads)))

    @classmethod
    def _gsm8k_num_examples(cls) -> int:
        return int(
            os.getenv(GSM8K_EXAMPLES_ENV, str(cls.model_pair.gsm8k_num_examples))
        )

    @classmethod
    def _max_running_requests(cls) -> int:
        return int(os.getenv(MAX_RUNNING_REQUESTS_ENV, "16"))

    @classmethod
    def _max_total_tokens(cls) -> int:
        return int(os.getenv(MAX_TOTAL_TOKENS_ENV, "32768"))

    @classmethod
    def _common_server_args(cls) -> List[str]:
        args = [
            "--attention-backend",
            "triton",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.55",
            "--max-running-requests",
            str(cls._max_running_requests()),
            "--context-length",
            "2048",
            "--max-total-tokens",
            str(cls._max_total_tokens()),
            "--skip-server-warmup",
        ]
        if cls.model_pair.tensor_parallel_size > 1:
            args += ["--tp-size", str(cls.model_pair.tensor_parallel_size)]
        return args

    @classmethod
    def _mtp_server_args(cls, topk: int = 1, num_draft_tokens: int = 6) -> List[str]:
        # NEXTN is resolved to Frozen-KV MTP for Gemma4 assistant drafts.
        # For topk=1, do not override --cuda-graph-bs; the test should
        # exercise the default capture set and padding-to-captured-batch-size
        # behavior. For topk>1, the target still uses CUDA graph where
        # available, while the Frozen-KV MTP draft loop currently runs eager.
        return [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-draft-model-path",
            cls.model_pair.assistant_path,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            str(topk),
            "--speculative-num-draft-tokens",
            str(num_draft_tokens),
        ] + cls._common_server_args()

    @classmethod
    def _gsm8k_args(cls, base_url: str) -> SimpleNamespace:
        return SimpleNamespace(
            base_url=base_url,
            model=cls.model_pair.target_path,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=cls._gsm8k_num_examples(),
            num_threads=cls._gsm8k_num_threads(),
        )

    @staticmethod
    def _stop_process(process) -> None:
        try:
            kill_process_tree(process.pid)
        except Exception:
            pass
        try:
            process.wait(timeout=30)
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # setUp / baseline
    # ------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ensure_checkpoint(cls.model_pair.target_path, cls.model_pair.name + " target")
        ensure_checkpoint(
            cls.model_pair.assistant_path, cls.model_pair.name + " assistant"
        )

        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"

        # Config A: target-only baseline.
        target_process = popen_launch_server(
            cls.model_pair.target_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=cls._server_env(),
            other_args=cls._common_server_args(),
        )
        try:
            requests.get(cls.base_url + "/flush_cache", timeout=30)
            metrics = run_eval(cls._gsm8k_args(cls.base_url))
            cls.target_score = float(metrics["score"])
        finally:
            cls._stop_process(target_process)

        print(f"[{cls.model_pair.name}] baseline (no MTP): {cls.target_score:.4f}")

    # ------------------------------------------------------------------
    # MTP runs
    # ------------------------------------------------------------------
    def _run_mtp_gsm8k(self, label: str, topk: int, num_draft_tokens: int) -> None:
        process = popen_launch_server(
            self.model_pair.target_path,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=self._server_env(),
            other_args=self._mtp_server_args(
                topk=topk, num_draft_tokens=num_draft_tokens
            ),
        )
        try:
            requests.get(self.base_url + "/flush_cache", timeout=30)
            server_info = get_server_info(self.base_url)
            self.assertEqual(
                server_info.get("speculative_eagle_topk"),
                topk,
                f"[{self.model_pair.name}/{label}]: server did not start with "
                f"speculative_eagle_topk={topk}",
            )
            disable_cuda_graph = bool(server_info.get("disable_cuda_graph"))
            if disable_cuda_graph:
                self.fail(
                    f"[{self.model_pair.name}/{label}]: CUDA graph was requested "
                    "but the server started with disable_cuda_graph=True"
                )
            metrics = run_eval(self._gsm8k_args(self.base_url))
            mtp_score = float(metrics["score"])
            avg_accept = get_avg_spec_accept_length(self.base_url) or 0.0
        finally:
            self._stop_process(process)

        print(
            f"[{self.model_pair.name}/{label}] target_score={self.target_score:.4f}"
            f" mtp_score={mtp_score:.4f}"
            f" avg_spec_accept_length={avg_accept:.4f}"
        )

        tol = self.model_pair.gsm8k_score_drop_tolerance
        if mtp_score + tol < self.target_score:
            self.fail(
                f"[{self.model_pair.name}/{label}]: GSM8K score {mtp_score:.4f}"
                f" dropped below baseline {self.target_score:.4f} - tol {tol:.4f}"
            )

        threshold = self.model_pair.accept_length_threshold
        if threshold is not None:
            if avg_accept is None:
                self.fail(
                    f"[{self.model_pair.name}/{label}]: avg_spec_accept_length missing"
                )
            if avg_accept < threshold:
                self.fail(
                    f"[{self.model_pair.name}/{label}]: avg_spec_accept_length"
                    f" {avg_accept:.4f} below threshold {threshold:.4f}"
                )

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------
    def test_gsm8k_no_mtp(self) -> None:
        """Acknowledge the calibrated baseline run from ``setUpClass``."""
        print(f"[{self.model_pair.name}/no-mtp] baseline score={self.target_score:.4f}")

    def test_gsm8k_mtp_cuda_graph(self) -> None:
        """Config C: MTP enabled with the FrozenKVMTPCudaGraphRunner."""
        self._run_mtp_gsm8k(
            label="mtp-cuda-graph",
            topk=1,
            num_draft_tokens=6,
        )

    def test_gsm8k_mtp_topk5(self) -> None:
        """Top-k fan-out: tree-shaped verify with eager Frozen-KV draft loop."""
        self._run_mtp_gsm8k(
            label="mtp-topk5-eager-draft",
            topk=5,
            num_draft_tokens=16,
        )


class TestGemma4E2BMTPGSM8K(Gemma4MTPGSM8KMixin, CustomTestCase):
    model_pair = ModelPair(
        name="E2B",
        target_path="google/gemma-4-E2B-it",
        assistant_path="gg-hf-am/gemma-4-E2B-it-assistant",
        # Loose threshold while we calibrate the algorithm; the GSM8K
        # score check is the primary correctness gate.
        accept_length_threshold=0.0,
    )


class TestGemma4E4BMTPGSM8K(Gemma4MTPGSM8KMixin, CustomTestCase):
    model_pair = ModelPair(
        name="E4B",
        target_path="google/gemma-4-E4B-it",
        assistant_path="gg-hf-am/gemma-4-E4B-it-assistant",
        accept_length_threshold=None,
    )


class TestGemma4MTP31BGSM8K(Gemma4MTPGSM8KMixin, CustomTestCase):
    model_pair = ModelPair(
        name="31B",
        target_path="google/gemma-4-31B-it",
        assistant_path="gg-hf-am/gemma-4-31B-it-assistant",
        accept_length_threshold=None,
        server_cuda_visible_devices="0,1",
        tensor_parallel_size=2,
    )


class TestGemma4MTP26BA4BGSM8K(Gemma4MTPGSM8KMixin, CustomTestCase):
    model_pair = ModelPair(
        name="26B-A4B",
        target_path="google/gemma-4-26B-A4B-it",
        assistant_path="gg-hf-am/gemma-4-26B-A4B-it-assistant",
        accept_length_threshold=None,
        server_cuda_visible_devices="0,1",
        tensor_parallel_size=2,
    )


if __name__ == "__main__":
    unittest.main()
