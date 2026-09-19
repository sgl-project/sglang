import concurrent.futures as cf
import json
import os
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, popen_launch_server

MODEL_PATH = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash-0731")
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000

# DSpark pushes bs * (gamma + 1) rows per rank into DeepEP low-latency dispatch,
# which has no Python-side overflow guard. gamma=5 and cuda-graph-max-bs-decode
# is 64, so 64 * 6 = 384; 512 leaves headroom under the hard 1024 ceiling.
DISPATCH_TOKENS_PER_RANK = "512"
MAX_RUNNING_REQUESTS = 64
GAMMA = 5

# A draft whose weights loaded incorrectly collapses to ~1.0. Observed mean is
# ~4.9 at full concurrency, so 2.0 separates healthy from broken with margin.
MIN_MEAN_ACCEPT_LENGTH = 2.0


class TestDeepSeekV4DsparkDeepEpDpAttention(CustomTestCase):
    """DSpark draft + DeepEP all-to-all + dp attention on 8x PPU (ep8/dp8)."""

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--served-model-name",
            "DeepSeek-V4-Flash",
            "--tp-size",
            "8",
            "--ep-size",
            "8",
            "--enable-dp-attention",
            "--dp-size",
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
            # dp attention divides this by dp_size; 4096 -> 512, which stays
            # divisible by the page_size the dsv4 backend forces (256).
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
        ]

        env = dict(os.environ)
        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = DISPATCH_TOKENS_PER_RANK
        # Use a fixed verify width to avoid potential all-to-all dispatch errors.
        env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
        # Warm up M in 1..1024 instead of 1..16384 (~5h20m -> ~6min). Covers
        # this config: prefill maxes at 512 and decode at 64 * 6 = 384.
        env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
        env.setdefault("SGLANG_DG_CACHE_DIR", "/tmp/dg_cache0")

        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _server_info(self):
        with urllib.request.urlopen(f"{BASE_URL}/server_info", timeout=60) as r:
            return json.loads(r.read())

    def _generate(self, text, max_new_tokens):
        body = json.dumps(
            {
                "text": text,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def test_server_resolves_to_dspark_deepep_dp8(self):
        info = self._server_info()
        self.assertEqual(info["speculative_algorithm"], "DSPARK")
        self.assertEqual(info["moe_a2a_backend"], "deepep")
        self.assertEqual(info["tp_size"], 8)
        self.assertEqual(info["dp_size"], 8)
        self.assertEqual(info["ep_size"], 8)
        self.assertTrue(info["enable_dp_attention"])
        self.assertTrue(info["enable_dp_lm_head"])
        self.assertEqual(info["speculative_num_draft_tokens"], GAMMA + 1)

    def test_concurrent_requests_span_every_dp_rank(self):
        """64 concurrent requests must all complete, land on all 8 DP ranks,
        stay arithmetically correct, and keep the draft accepting."""
        prompt = (
            "Q: A store has {} widgets. It sells 37 and then receives a shipment "
            "of 55. How many widgets does it have now? Think step by step.\nA:"
        )

        def one(i):
            out = self._generate(prompt.format(100 + i), 128)
            mi = out["meta_info"]
            return {
                "dp_rank": mi.get("dp_rank"),
                "accept_length": mi.get("spec_accept_length"),
                "retractions": mi.get("num_retractions"),
                "finish": (mi.get("finish_reason") or {}).get("type"),
                "text": out.get("text", ""),
                "expected": 100 + i - 37 + 55,
            }

        with cf.ThreadPoolExecutor(max_workers=MAX_RUNNING_REQUESTS) as ex:
            results = list(ex.map(one, range(MAX_RUNNING_REQUESTS)))

        self.assertEqual(len(results), MAX_RUNNING_REQUESTS)
        self.assertEqual(sum(r["retractions"] or 0 for r in results), 0)

        ranks = {r["dp_rank"] for r in results}
        self.assertEqual(
            ranks,
            set(range(8)),
            f"requests did not reach every DP rank: {sorted(ranks)}",
        )

        wrong = [r for r in results if str(r["expected"]) not in r["text"]]
        if wrong:
            self.fail(
                f"{len(wrong)} wrong answers, e.g. expected {wrong[0]['expected']}: "
                f"{wrong[0]['text'][:200]!r}"
            )

        accept = [r["accept_length"] for r in results if r["accept_length"]]
        self.assertTrue(accept, "no spec_accept_length reported; draft not running?")
        mean_accept = sum(accept) / len(accept)
        self.assertGreater(
            mean_accept,
            MIN_MEAN_ACCEPT_LENGTH,
            f"mean accept length {mean_accept:.2f} suggests a broken draft",
        )

    def test_idle_dp_ranks_still_join_the_handshake(self):
        """A single request leaves 7 DP ranks idle. They must still participate
        in the DeepEP low-latency dispatch or the collective deadlocks."""
        first = self._generate("The capital of France is", 32)["text"]
        self.assertIn("Paris", first)

        # Repeat so the idle ranks exercise the handshake across several steps.
        for _ in range(3):
            out = self._generate("Q: What is 17 multiplied by 23?\nA:", 64)
            self.assertIn("391", out["text"])


if __name__ == "__main__":
    unittest.main()
