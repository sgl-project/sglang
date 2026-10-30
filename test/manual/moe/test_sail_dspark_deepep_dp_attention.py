"""DeepEP concurrent dispatch and idle-rank handshake on SAIL TP8/EP8/attn-DP8."""

import concurrent.futures as cf
import os
import unittest

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_sail
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Not registered in CI: it needs a local checkpoint with an MoE draft and 8 GPUs,
# so it stays manual.
MODEL_PATH = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash-0731")
BASE_URL = "http://127.0.0.1:8999"
MAX_RUNNING_REQUESTS = 64
GAMMA = 5
# The LL per-rank capacity must cover graph batch * verify width: 64 * 6 = 384.
DISPATCH_TOKENS_PER_RANK = 512
MIN_MEAN_ACCEPT_LENGTH = 2.0


class TestSailDsparkDeepEPDpAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest("Requires a SAIL device.")
        if torch.cuda.device_count() < 8:
            raise unittest.SkipTest("Requires 8 SAIL devices.")
        with (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(
                DISPATCH_TOKENS_PER_RANK
            ),
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.override(True),
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False),
        ):
            cls.process = popen_launch_server(
                MODEL_PATH,
                BASE_URL,
                timeout=60000,
                other_args=[
                    "--trust-remote-code",
                    "--served-model-name",
                    "DeepSeek-V4-Flash",
                    "--tp-size",
                    "8",
                    "--ep-size",
                    "8",
                    "--attn-dp-size",
                    "8",
                    "--enable-dp-lm-head",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "low_latency",
                    "--moe-dense-tp-size",
                    "1",
                    "--disable-shared-experts-fusion",
                    "--attention-backend",
                    "dsv4",
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--mem-fraction-static",
                    "0.7",
                    "--max-running-requests",
                    str(MAX_RUNNING_REQUESTS),
                    "--cuda-graph-max-bs-decode",
                    str(MAX_RUNNING_REQUESTS),
                    # attention-DP splits the global 4096 into 512 per rank,
                    # which still aligns with page 256.
                    "--chunked-prefill-size",
                    "4096",
                    "--page-size",
                    "256",
                    "--disable-radix-cache",
                    "--disable-custom-all-reduce",
                    "--schedule-conservativeness",
                    "0.3",
                    "--num-continuous-decode-steps",
                    "1",
                    "--speculative-algorithm",
                    "DSPARK",
                    "--speculative-dspark-block-size",
                    str(GAMMA),
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    str(GAMMA + 1),
                    "--reasoning-parser",
                    "deepseek-v4",
                    "--tool-call-parser",
                    "deepseekv4",
                    "--enable-cache-report",
                    "--decode-log-interval",
                    "1",
                    "--watchdog-timeout",
                    "60000",
                    "--soft-watchdog-timeout",
                    "60000",
                    "--dist-timeout",
                    "60000",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "process") and cls.process:
                terminate_and_kill_process_tree(cls.process)
        finally:
            super().tearDownClass()

    def _generate(self, text, max_new_tokens):
        response = requests.post(
            BASE_URL + "/generate",
            json={
                "text": text,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
            timeout=900,
        )
        response.raise_for_status()
        return response.json()

    def test_resolved_deepep_topology(self):
        response = requests.get(BASE_URL + "/server_info", timeout=60)
        response.raise_for_status()
        info = response.json()
        for field, expected in (
            ("speculative_algorithm", "DSPARK"),
            ("moe_a2a_backend", "deepep"),
            ("tp_size", 8),
            ("dp_size", 1),
            ("attn_dp_size", 8),
            ("ep_size", 8),
            ("enable_dp_lm_head", True),
            ("speculative_num_draft_tokens", GAMMA + 1),
        ):
            self.assertEqual(info[field], expected, field)

    def test_concurrent_requests_span_every_dp_rank(self):
        """All 64 requests must finish, reach every rank, answer correctly and keep
        the draft accepted."""
        prompt = (
            "Q: A store has {} widgets. It sells 37 and then receives a shipment "
            "of 55. How many widgets does it have now? Think step by step.\nA:"
        )

        def one(i):
            output = self._generate(prompt.format(100 + i), 128)
            return output, 100 + i - 37 + 55

        with cf.ThreadPoolExecutor(max_workers=MAX_RUNNING_REQUESTS) as executor:
            results = list(executor.map(one, range(MAX_RUNNING_REQUESTS)))
        self.assertEqual(len(results), MAX_RUNNING_REQUESTS)
        metadata = [output["meta_info"] for output, _ in results]
        self.assertEqual(sum(mi.get("num_retractions") or 0 for mi in metadata), 0)
        self.assertEqual({mi.get("dp_rank") for mi in metadata}, set(range(8)))
        for output, expected in results:
            self.assertIn(str(expected), output["text"])
        accept = [
            mi["spec_accept_length"] for mi in metadata if mi.get("spec_accept_length")
        ]
        self.assertTrue(accept, "No spec_accept_length; the draft may not have run.")
        self.assertGreater(sum(accept) / len(accept), MIN_MEAN_ACCEPT_LENGTH)

    def test_idle_dp_ranks_join_handshake(self):
        """A single request leaves 7 idle ranks; they must join the LL dispatch so the
        collective does not deadlock."""
        self.assertIn("Paris", self._generate("The capital of France is", 32)["text"])
        for _ in range(3):
            output = self._generate("Q: What is 17 multiplied by 23?\nA:", 64)
            self.assertIn("391", output["text"])


if __name__ == "__main__":
    unittest.main()
