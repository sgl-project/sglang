"""Adaptive speculative decoding under data-parallel attention (multi-GPU).

Regression guard for the DP-attention consensus path: when --enable-dp-attention
is set, each DP rank must activate the *same* adaptive tier every decode step.
A divergent tier means a divergent num_draft_tokens -> divergent
global_dp_buffer_len -> mismatched NCCL collective shapes -> hang. The consensus
(min over per-rank desired tiers, carried on the existing MLP-sync all_gather)
keeps every rank in lockstep, so a tier switch that changes the draft-token width
mid-flight must still complete rather than deadlock.

Topology: --tp 2 --dp 2 --enable-dp-attention -> 2 GPUs, 2 DP-attention groups.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=360, stage="extra-a", runner_config="2-gpu-large-amd")

# A batch big enough that, split across 2 DP ranks by the auto load balancer,
# every rank still lands at BS >= 8 (the high-load tier's routing key).
HIGH_LOAD_BATCH = 24


class TestAdaptiveSpeculativeDPAttentionServer(CustomTestCase):
    """Adaptive tier switching stays lockstep across DP ranks (no deadlock).

    Config routes BS<8 -> steps=3 (draft width 4) and BS>=8 -> steps=1 (draft
    width 2), so a rising/falling load forces a *shape-changing* tier switch.
    Under --enable-dp-attention this only completes if the two ranks agree on
    the tier; otherwise the verify collective deadlocks.
    """

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    COUNT_PROMPT = "Count from 1 to 400, separated by commas. Output only the numbers."

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [3], "warmup_batches": 0},
                    "8": {"candidate_steps": [1], "warmup_batches": 0},
                },
                f,
            )
            cls.adaptive_config_path = f.name

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--tp",
                    "2",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--max-running-requests",
                    "64",
                    "--skip-server-warmup",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _steps(self) -> int:
        r = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["internal_states"][0]["speculative_num_steps"]

    def _generate_single(self) -> dict:
        one = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}
        r = requests.post(
            self.base_url + "/generate",
            json={"text": self.COUNT_PROMPT, "sampling_params": one},
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["meta_info"]

    def test_dp_tier_switch_no_deadlock(self):
        """BS=1 -> steps=3 -> shape-changing switch to steps=1 under load -> back.

        Each phase completing (not timing out) is the deadlock check: a broken
        consensus lets the ranks diverge on num_draft_tokens and the verify
        collective hangs. The accept-rate assertions additionally prove drafting
        actually runs under DP, not just that the server didn't crash.
        """
        # Phase 1: BS=1 -> steps=3, drafting active on the serving rank.
        m1 = self._generate_single()
        self.assertEqual(self._steps(), 3, "expected steps=3 at BS=1")
        self.assertGreater(
            m1["spec_accept_rate"], 0.8, f"not drafting at steps=3 under DP: {m1}"
        )

        # Phase 2: a 24-way batch splits ~12 per DP rank (>= 8) -> steps=1. Both
        # ranks must switch to the narrower draft width together, mid-flight.
        full = {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}
        r = requests.post(
            self.base_url + "/generate",
            json={
                "text": [self.COUNT_PROMPT] * HIGH_LOAD_BATCH,
                "sampling_params": [full] * HIGH_LOAD_BATCH,
            },
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._steps(), 1, "BS>=8 did not switch to steps=1 under DP")

        # Phase 3: BS=1 -> steps=3 again, drafting restored (shape widened back).
        m3 = self._generate_single()
        self.assertEqual(self._steps(), 3, "did not reopen to steps=3 under DP")
        self.assertGreater(
            m3["spec_accept_rate"], 0.8, f"drafting not restored under DP: {m3}"
        )

    def test_gsm8k_accuracy_under_dp(self):
        """Correctness is preserved through the adaptive+DP path (no regression),
        and the lifetime accept-length is reported for the PR evidence trail."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"GSM8K under adaptive+dp-attention: {metrics}")
        self.assertGreater(metrics["score"], 0.20)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_accept_len = server_info["internal_states"][0]["avg_spec_accept_length"]
        print(f"avg_spec_accept_length (dp)={avg_accept_len:.4f}")


if __name__ == "__main__":
    unittest.main()
