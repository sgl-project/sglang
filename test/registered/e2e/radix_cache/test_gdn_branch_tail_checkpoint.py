"""Opt-in GDN branch capture preserves final-tail reuse through real Decode."""

import json
import math
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")


class TestGDNBranchTailCheckpoint(CustomTestCase):
    model = os.environ.get(
        "QWEN35_GDN_TEST_MODEL", DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST
    )
    tp = int(os.environ.get("GDN_BRANCH_TAIL_TEST_TP", "1"))

    def _request(self, body):
        body.setdefault("sampling_params", {}).update(temperature=0, ignore_eos=True)
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate", json=body, timeout=180
        )
        response.raise_for_status()
        return response.json()

    def _generate(self, ids, salt, length=1):
        result = self._request(
            {
                "input_ids": ids,
                "cache_salt": salt,
                "sampling_params": {"max_new_tokens": length},
            }
        )
        self.assertEqual(result["meta_info"]["prompt_tokens"], len(ids))
        self.assertEqual(len(result["output_ids"]), length)
        return result

    def _flush(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/flush_cache", params={"timeout": 30}, timeout=60
        )
        response.raise_for_status()

    def _score(self, ids, salt):
        # The first returned position has no score; include the preceding token.
        result = self._request(
            {
                "input_ids": ids,
                "cache_salt": salt,
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 6048,
            }
        )
        pairs = [
            (value, token)
            for value, token, _ in result["meta_info"]["input_token_logprobs"]
            if value is not None
        ]
        self.assertEqual([token for _, token in pairs], ids[6049:])
        self.assertTrue(all(math.isfinite(value) for value, _ in pairs))
        return result, [value for value, _ in pairs]

    def _arm(self, enabled, seed, producer, fork, suffix):
        args = [
            "--tp-size",
            str(self.tp),
            "--language-only",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "triton",
            "--linear-attn-prefill-backend",
            "triton",
            "--linear-attn-decode-backend",
            "triton",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--page-size",
            "64",
            "--hybrid-cache-checkpoints-level",
            "2",
            "--mamba-max-states-per-path",
            "1",
            "--max-mamba-cache-size",
            "64",
            "--max-total-tokens",
            "32768",
            "--context-length",
            "8192",
            "--chunked-prefill-size",
            "4096",
            "--max-prefill-tokens",
            "4096",
            "--max-running-requests",
            "4",
            "--disable-overlap-schedule",
            "--disable-decode-cuda-graph",
            "--disable-prefill-cuda-graph",
            "--disable-custom-all-reduce",
            "--mem-fraction-static",
            "0.7",
        ]
        values = {}
        with tempfile.TemporaryFile(mode="w+") as log:
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=args,
                env={"SGLANG_EXPERIMENTAL_GDN_BRANCH_AND_TAIL": str(int(enabled))},
                return_stdout_stderr=(log, log),
            )
            try:
                self._flush()
                for length in (1, 32, 128):
                    salt = f"branch-tail-{length}"
                    self._generate(seed, salt)
                    log.seek(0, 2)
                    start = log.tell()
                    produced = self._generate(producer, salt, length)
                    if length == 1:
                        log.seek(start)
                        batches = re.findall(
                            r"Prefill batch, #new-seq: 1, #new-token: (\d+), #cached-token: (\d+)",
                            log.read(),
                        )
                        self.assertEqual(batches, [("4096", "0"), ("1953", "0")])
                    continued = self._generate(producer + suffix, salt)
                    if length in (1, 32):
                        self.assertEqual(
                            continued["meta_info"]["cached_tokens"],
                            6016 if enabled else 5952,
                        )
                    # Rebuild the initial tree for the independent branch probe.
                    self._flush()
                    self._generate(seed, salt)
                    self._generate(producer, salt, length)
                    forked = self._generate(fork, salt)
                    self.assertEqual(forked["meta_info"]["cached_tokens"], 5952)
                    values[str(length)] = {
                        "output_ids": produced["output_ids"],
                        "continuation_cached": continued["meta_info"]["cached_tokens"],
                        "fork_cached": forked["meta_info"]["cached_tokens"],
                    }
                    self._flush()

                salt = "branch-tail-score"
                self._generate(seed, salt)
                self._generate(producer, salt)
                scored, hit_scores = self._score(producer + suffix, salt)
                _, cold_scores = self._score(producer + suffix, "score-cold-1")
                _, cold_control = self._score(producer + suffix, "score-cold-2")
                self.assertEqual(len(hit_scores), len(cold_scores))
                values["scores"] = {
                    "cached": scored["meta_info"]["cached_tokens"],
                    "hit": hit_scores,
                    "cold": cold_scores,
                    "cold_repeat": cold_control,
                }
                self._flush()
            finally:
                terminate_and_kill_process_tree(process)
        return values

    def test_branch_and_tail_reuse_with_multistep_decode(self):
        rng = random.Random(427)
        common = [rng.randrange(1000, 2000) for _ in range(5952)]
        seed = common + [rng.randrange(1000, 2000) for _ in range(255)]
        producer = common + [rng.randrange(1000, 2000) for _ in range(97)]
        fork = common + [rng.randrange(1000, 2000) for _ in range(12)]
        suffix = [rng.randrange(1000, 2000) for _ in range(64)]
        self.assertEqual(len({seed[5952], producer[5952], fork[5952]}), 3)
        control = self._arm(False, seed, producer, fork, suffix)
        enabled = self._arm(True, seed, producer, fork, suffix)
        for length in ("1", "32", "128"):
            self.assertEqual(
                control[length]["output_ids"], enabled[length]["output_ids"]
            )
        for length in ("1", "32"):
            self.assertEqual(control[length]["continuation_cached"], 5952)
            self.assertEqual(enabled[length]["continuation_cached"], 6016)
        # With 128 outputs native decode tracking publishes a later state; the
        # deliberately input-only continuation does not share that output suffix.
        self.assertEqual(
            control["128"]["continuation_cached"], enabled["128"]["continuation_cached"]
        )
        control_delta = [
            abs(a - b)
            for a, b in zip(control["scores"]["hit"], control["scores"]["cold"])
        ]
        enabled_delta = [
            abs(a - b)
            for a, b in zip(enabled["scores"]["hit"], enabled["scores"]["cold"])
        ]
        cold_repeat_delta = [
            abs(a - b)
            for a, b in zip(control["scores"]["cold"], control["scores"]["cold_repeat"])
        ]
        report = {
            "control": control,
            "enabled": enabled,
            "logprob_delta": {
                "native_restore": control_delta,
                "dual_restore": enabled_delta,
                "cold_repeat": cold_repeat_delta,
            },
        }
        # Same BF16 kernel and state dtype: compare against native restoration
        # noise, with an absolute margin for a different recomputation length.
        native_mean = sum(control_delta) / len(control_delta)
        dual_mean = sum(enabled_delta) / len(enabled_delta)
        self.assertLessEqual(dual_mean, 2 * native_mean + 0.02, report["logprob_delta"])
        if path := os.environ.get("GDN_BRANCH_TAIL_RESULT_PATH"):
            Path(path).write_text(json.dumps(report, indent=2))
        print(json.dumps(report["logprob_delta"]), flush=True)


if __name__ == "__main__":
    unittest.main()
