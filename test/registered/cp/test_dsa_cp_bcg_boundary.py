"""Boundary matrix for DSA interleave-CP breakable prefill graphs (GlmMoeDsa).

Runs the same prompts through a server with breakable prefill graphs on and
one with prefill graphs disabled (eager CP) for live chunk lengths that sit on
a capture bucket, one below it, and lengths not divisible by the CP size, and
checks (i) greedy tokens / prompt logprobs agree, (ii) the captured path was
taken for lengths under the capture cap, and (iii) KV rows of a neighbouring
cached request are untouched (radix-cache canary: a request whose KV slots are
allocated next to the probe is replayed from cache afterwards and must
reproduce its logprobs).
"""

import json
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=3600, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = os.environ.get("SGLANG_TEST_CP_BCG_MODEL", "Inferact/GLM-5.3-NVFP4")
SERVER_LAUNCH_TIMEOUT = 1800
# Max |delta logprob| between graphs on/off over the prompt tokens (teacher
# forced, so no divergence effect) and the greedy continuation.
LOGPROB_TOL = 0.1
# Bucket-exact, bucket-1 and not divisible by the CP size.
LENGTHS = (100, 127, 128, 333, 511, 512, 1001)
MAX_NEW_TOKENS = 16
CANARY_LENGTH = 512
# Default breakable prefill buckets reach well beyond every length above.
CAPTURE_CAP = 4096
RESULTS_PATH = os.environ.get("SGLANG_TEST_CP_BCG_RESULTS")


def _common_args(attn_cp_size: int):
    return [
        "--trust-remote-code",
        "--quantization",
        "modelopt_fp4",
        "--tp",
        "4",
        "--ep-size",
        str(attn_cp_size),
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--attn-cp-size",
        str(attn_cp_size),
        "--dsa-prefill-backend",
        "trtllm",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        "64",
        "--context-length",
        "65536",
        "--chunked-prefill-size",
        "32768",
        "--max-prefill-tokens",
        "32768",
        "--max-running-requests",
        "16",
        "--mem-fraction-static",
        "0.80",
        "--enable-metrics",
        "--random-seed",
        "42",
    ]


class TestDsaCpBcgBoundaryCP4(CustomTestCase):
    attn_cp_size = 4

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.external = os.environ.get("SGLANG_TEST_CP_BCG_URL_ON") is not None

    @classmethod
    def tearDownClass(cls):
        cls._stop()

    @classmethod
    def _stop(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None

    @classmethod
    def _launch(cls, prefill_graphs: bool):
        cls._stop()
        if cls.external:
            key = (
                "SGLANG_TEST_CP_BCG_URL_ON"
                if prefill_graphs
                else "SGLANG_TEST_CP_BCG_URL_OFF"
            )
            cls.base_url = os.environ[key]
            return
        backend = "breakable" if prefill_graphs else "disabled"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=_common_args(cls.attn_cp_size)
            + ["--cuda-graph-backend-prefill", backend],
        )

    # ---- helpers -----------------------------------------------------------

    def _flush(self):
        requests.post(
            self.base_url + "/flush_cache", params={"timeout": 30}, timeout=60
        ).raise_for_status()

    def _prompt(self, length: int, seed: int = 0):
        # Token ids well inside the vocabulary; seed shifts the content so the
        # canary and the probe never share a prefix.
        base = 1000 + 7 * seed
        return [base + (i * 31 + seed * 17) % 5000 for i in range(length)]

    def _generate(self, input_ids, max_new_tokens: int):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _prefill_graph_passes(self):
        response = requests.get(self.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith("sglang:cuda_graph_passes_total{")
            and 'mode="prefill_cuda_graph"' in line
        )

    @staticmethod
    def _logprobs(output):
        meta = output["meta_info"]
        prompt = [
            lp for lp, _tok, _txt in meta["input_token_logprobs"] if lp is not None
        ]
        gen = [lp for lp, _tok, _txt in meta["output_token_logprobs"]]
        return prompt, gen

    @staticmethod
    def _max_delta(a, b):
        return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)

    def _run_matrix(self, prefill_graphs: bool):
        """One pass over LENGTHS on the current server; returns per-length rows."""
        rows = {}
        canary_ids = self._prompt(CANARY_LENGTH, seed=99)
        for length in LENGTHS:
            self._flush()
            # Canary first: its KV slots are allocated right before the probe's.
            canary_ref = self._generate(canary_ids, MAX_NEW_TOKENS)
            self.assertEqual(canary_ref["meta_info"]["cached_tokens"], 0)
            before = self._prefill_graph_passes()
            probe = self._generate(self._prompt(length), MAX_NEW_TOKENS)
            graph_passes = self._prefill_graph_passes() - before
            self.assertEqual(probe["meta_info"]["prompt_tokens"], length)
            if prefill_graphs and length <= CAPTURE_CAP:
                self.assertGreater(
                    graph_passes, 0, f"length={length}: eager fallback, no graph pass"
                )
            elif not prefill_graphs:
                self.assertEqual(graph_passes, 0, f"length={length}")
            # Replay the canary from the radix cache: every cached KV row is
            # read again, so a pad-row write into its slots changes the result.
            canary_hit = self._generate(canary_ids, MAX_NEW_TOKENS)
            self.assertGreater(canary_hit["meta_info"]["cached_tokens"], 0)
            self.assertEqual(
                canary_hit["output_ids"],
                canary_ref["output_ids"],
                f"length={length}: canary KV rows changed under the probe",
            )
            ref_gen = self._logprobs(canary_ref)[1]
            hit_gen = self._logprobs(canary_hit)[1]
            canary_delta = self._max_delta(ref_gen, hit_gen)
            self.assertLessEqual(canary_delta, LOGPROB_TOL, f"length={length}")
            prompt_lp, gen_lp = self._logprobs(probe)
            rows[length] = {
                "text": probe["text"],
                "output_ids": probe["output_ids"],
                "prompt_logprobs": prompt_lp,
                "gen_logprobs": gen_lp,
                "graph_passes": graph_passes,
                "canary_rows": canary_hit["meta_info"]["cached_tokens"],
                "canary_max_delta": canary_delta,
            }
            print(
                f"cp={self.attn_cp_size} graphs={'on' if prefill_graphs else 'off'} "
                f"length={length} graph_passes={graph_passes} "
                f"canary_rows={rows[length]['canary_rows']} "
                f"canary_max_delta={canary_delta:.4g}"
            )
        return rows

    def test_graphs_on_vs_eager_cp(self):
        self._launch(prefill_graphs=True)
        on = self._run_matrix(prefill_graphs=True)
        self._launch(prefill_graphs=False)
        off = self._run_matrix(prefill_graphs=False)
        self._stop()

        summary = {}
        for length in LENGTHS:
            a, b = on[length], off[length]
            prompt_delta = self._max_delta(a["prompt_logprobs"], b["prompt_logprobs"])
            n = min(len(a["gen_logprobs"]), len(b["gen_logprobs"]))
            gen_delta = self._max_delta(a["gen_logprobs"][:n], b["gen_logprobs"][:n])
            same_tokens = sum(
                x == y for x, y in zip(a["output_ids"], b["output_ids"], strict=False)
            )
            summary[length] = {
                "cp": self.attn_cp_size,
                "graph_passes_on": a["graph_passes"],
                "graph_passes_off": b["graph_passes"],
                "max_prompt_logprob_delta": prompt_delta,
                "max_gen_logprob_delta": gen_delta,
                "same_output_tokens": f"{same_tokens}/{len(a['output_ids'])}",
                "canary_rows_checked": a["canary_rows"],
                "canary_max_delta_on": a["canary_max_delta"],
                "canary_max_delta_off": b["canary_max_delta"],
            }
            print(f"summary length={length}: {summary[length]}")
            self.assertLessEqual(prompt_delta, LOGPROB_TOL, f"length={length}")
            self.assertEqual(
                a["output_ids"][0], b["output_ids"][0], f"length={length}: first token"
            )
        if RESULTS_PATH:
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(summary) + "\n")


class TestDsaCpBcgBoundaryCP2(TestDsaCpBcgBoundaryCP4):
    attn_cp_size = 2


if __name__ == "__main__":
    unittest.main()
