"""
E2E Tests: NPU MoE Full RL Training Loop
=========================================
Simulates real RLHF full training steps: pause → release_memory → resume_memory → weight_update → resume → verify.

Covered cross paths:
  - pause(in_place) / pause(retract) two modes
  - release_memory / resume_memory (weights + kv_cache)
  - update_weights_from_disk (same model no-op)
  - decode consistency verification after resume

Model: Qwen/Qwen3-30B-A3B (BF16, Unquantized MoE)
Hardware: NPU

Note:
  - the NPU-customized torch_memory_saver is required.
    pip install torch_memory_saver-xxx.whl
"""

import logging
import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_WEIGHTS_PATH as QWEN3_30B_A3B,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=300,
    suite="full-1-npu-a3",
    nightly=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _get_decode_signature(base_url):
    """Deterministic decode signature for verifying consistency before and after weight updates."""
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            "return_logprob": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    ret = resp.json()
    logprobs = ret["meta_info"]["output_token_logprobs"]
    assert logprobs, "empty output_token_logprobs"
    return {
        "text": ret["text"],
        "token_ids": [int(x[1]) for x in logprobs],
        "logprobs": [float(x[0]) for x in logprobs],
    }


def _assert_same(a, b, *, atol=1e-4, msg=""):
    assert a["text"] == b["text"], f"{msg}text mismatch: {a['text']!r} != {b['text']!r}"
    assert a["token_ids"] == b["token_ids"], f"{msg}token_ids mismatch"
    for idx, (la, lb) in enumerate(zip(a["logprobs"], b["logprobs"])):
        assert abs(la - lb) <= atol, (
            f"{msg}logprob diff at idx={idx}: {la} vs {lb} "
            f"(delta={abs(la - lb):.6f}, tol={atol})"
        )


class TestNPUMoEFullRLLoop(CustomTestCase):
    """E2E test for MoE model full RL training loop.

    Simulates RLHF scenario:
      rollout → pause → release_memory → (training occupies memory)
      → resume_memory → weight_update → resume → continue generation under new weights

    [Test Category] RL Sleep Mode + Full Loop
    [Test Target] POST /pause_generation, POST /continue_generation,
                  POST /release_memory_occupation, POST /resume_memory_occupation,
                  POST /update_weights_from_disk, POST /v1/chat/completions
    """

    REQUEST_TIMEOUT = 600

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B
        cls.base_url = DEFAULT_URL_FOR_TEST
        server_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.95",
            "--disable-cuda-graph",
            "--max-running-requests",
            "8",
            "--tp-size",
            "1",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env={**os.environ, "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        try:
            import subprocess as sp

            sp.run(["pkill", "-f", "multiprocessing.forkserver"], capture_output=True)
        except Exception:
            pass

    # ── helpers ──────────────────────────────────────────────────

    def _pause(self, mode="in_place"):
        return requests.post(
            f"{self.base_url}/pause_generation",
            json={"mode": mode},
            timeout=30,
        )

    def _continue(self):
        return requests.post(
            f"{self.base_url}/continue_generation",
            json={},
            timeout=30,
        )

    def _release_memory(self, tags):
        return requests.post(
            f"{self.base_url}/release_memory_occupation",
            json={"tags": tags},
            timeout=60,
        )

    def _resume_memory(self, tags):
        return requests.post(
            f"{self.base_url}/resume_memory_occupation",
            json={"tags": tags},
            timeout=60,
        )

    def _update_weights(self):
        return requests.post(
            f"{self.base_url}/update_weights_from_disk",
            json={
                "model_path": self.model,
                "flush_cache": True,
            },
            timeout=600,
        )

    def test_npu_moe_rl_loop_in_place(self):
        """Full RL loop (in_place).

        pause(in_place) → release_memory(weights+kv_cache) → resume_memory → weight_update → resume → verify.
        """
        logger.info("[RL loop in_place] baseline decode...")
        baseline = _get_decode_signature(self.base_url)

        logger.info("[RL loop in_place] pause(in_place)")
        self._pause("in_place").raise_for_status()
        try:
            time.sleep(1)
            self._release_memory(["weights", "kv_cache"]).raise_for_status()
            time.sleep(2)
            self._resume_memory(["weights", "kv_cache"]).raise_for_status()
        finally:
            self._continue().raise_for_status()

        logger.info("[RL loop in_place] update_weights_from_disk (no-op)")
        resp = self._update_weights()
        self.assertEqual(
            resp.status_code, 200, f"Weight update failed: {resp.text[:500]}"
        )
        self.assertTrue(
            resp.json().get("success"), f"Weight update not successful: {resp.json()}"
        )

        logger.info("[RL loop in_place] verify decode")
        updated = _get_decode_signature(self.base_url)
        _assert_same(
            baseline, updated, msg="RL loop in_place: weight update broke decode. "
        )

    def test_npu_moe_rl_loop_retract(self):
        """Full RL loop (retract).

        pause(retract) → release_memory(weights+kv_cache) → resume_memory → weight_update → resume → verify.
        """
        logger.info("[RL loop retract] baseline decode...")
        baseline = _get_decode_signature(self.base_url)

        logger.info("[RL loop retract] pause(retract)")
        self._pause("retract").raise_for_status()
        try:
            time.sleep(1)
            self._release_memory(["weights", "kv_cache"]).raise_for_status()
            time.sleep(2)
            self._resume_memory(["weights", "kv_cache"]).raise_for_status()
        finally:
            self._continue().raise_for_status()

        logger.info("[RL loop retract] update_weights_from_disk (no-op)")
        resp = self._update_weights()
        self.assertEqual(
            resp.status_code, 200, f"Weight update failed: {resp.text[:500]}"
        )
        self.assertTrue(
            resp.json().get("success"), f"Weight update not successful: {resp.json()}"
        )

        logger.info("[RL loop retract] verify decode")
        updated = _get_decode_signature(self.base_url)
        _assert_same(
            baseline, updated, msg="RL loop retract: weight update broke decode. "
        )

    def test_npu_moe_rl_loop_multi_cycle(self):
        """Multi-cycle RL loop — alternating in_place + retract, each cycle independently verified.

        Verifies: decode remains consistent with baseline after two cycles of
        pause/release/resume_mem/update, with no accumulated state errors.
        """
        logger.info("[RL loop multi_cycle] baseline decode...")
        baseline = _get_decode_signature(self.base_url)

        for cycle, mode in enumerate(["in_place", "retract"]):
            logger.info("[RL loop multi_cycle] cycle %d — pause(%s)", cycle, mode)
            self._pause(mode).raise_for_status()
            try:
                time.sleep(1)
                self._release_memory(["weights", "kv_cache"]).raise_for_status()
                time.sleep(1)
                self._resume_memory(["weights", "kv_cache"]).raise_for_status()
            finally:
                self._continue().raise_for_status()

            logger.info(
                "[RL loop multi_cycle] cycle %d update_weights_from_disk (no-op)", cycle
            )
            resp = self._update_weights()
            self.assertTrue(
                resp.json().get("success"),
                f"Cycle {cycle} weight update failed: {resp.json()}",
            )

            updated = _get_decode_signature(self.base_url)
            _assert_same(
                baseline, updated, msg=f"Multi-cycle cycle {cycle}: decode diverged. "
            )


if __name__ == "__main__":
    unittest.main()
