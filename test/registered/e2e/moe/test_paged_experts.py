"""Paged experts match the unpaged model, with decode captured (prompts one at a time) and host-planned (one batch)."""

import statistics
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-large")

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "The ocean's waves crash against the shore in an endless rhythm,",
]
# Decode graphs only at bs 1, so a batch of prompts decodes without one; prefill graphs only at
# the sizes the prompts need.
SERVER_ARGS = [
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-max-bs-decode",
    "1",
    "--cuda-graph-bs-prefill",
    "64",
    "128",
    "--skip-server-warmup",
    # Small enough that the unpaged and paged servers share one GPU.
    "--max-total-tokens",
    "4096",
    "--mem-fraction-static",
    "0.6",
    "--disable-shared-experts-fusion",
    "--disable-radix-cache",
    "--enable-deterministic-inference",
    "--attention-backend",
    "triton",
]
PAGED_ARGS = ["--enable-paged-experts", "--paged-experts-num-resident", "8"]
_HOST, _PORT = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
PAGED_URL = f"{_HOST}:{find_available_port(int(_PORT) + 100)}"


class _PagedMatchesUnpaged:
    model: str
    #: False for MoE kernels that are not batch-invariant: compare the median logprob difference
    #: per prompt, decode ones up to the first greedy token the servers differ on.
    exact = True
    median_abs_tolerance = 0.0

    @classmethod
    def setUpClass(cls):
        # Both servers boot and score at the same time.
        with ThreadPoolExecutor(2) as pool:
            reference = pool.submit(
                _serve_and_score, cls.model, DEFAULT_URL_FOR_TEST, []
            )
            paged = pool.submit(_serve_and_score, cls.model, PAGED_URL, PAGED_ARGS)
            cls.reference, cls.paged = reference.result(), paged.result()

    def test_captured_decode_matches(self):
        self._assert_match(self.reference["single"], self.paged["single"])

    def test_host_planned_decode_matches(self):
        self._assert_match(self.reference["batch"], self.paged["batch"])

    def _assert_match(self, reference, paged):
        decode_compared = decode_total = 0
        for prompt, ref, got in zip(PROMPTS, reference, paged):
            with self.subTest(prompt=prompt[:30]):
                if self.exact:
                    self.assertEqual(got, ref)
                    continue
                diffs = [abs(a - b) for a, b in zip(got[0], ref[0]) if a is not None]
                for (ref_tok, ref_lp), (tok, lp) in zip(ref[1], got[1]):
                    if tok != ref_tok:
                        break
                    diffs.append(abs(lp - ref_lp))
                    decode_compared += 1
                decode_total += len(ref[1])
                self.assertLessEqual(
                    statistics.median(diffs), self.median_abs_tolerance
                )
        if not self.exact:
            self.assertGreaterEqual(decode_compared, decode_total // 2)


class TestPagedExpertsBf16(_PagedMatchesUnpaged, CustomTestCase):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE


class TestPagedExpertsFp8Block(_PagedMatchesUnpaged, CustomTestCase):
    """Block-FP8 experts (128x128 scales) on the triton runner."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE


class TestPagedExpertsGptq(_PagedMatchesUnpaged, CustomTestCase):
    """GPTQ int4 experts on the Marlin runner, whose reduction depends on the batch shape: the
    unpaged server alone differs by up to 4 nats at a near-tied position between one prompt and
    a batch, with medians below 0.05."""

    model = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    exact = False
    median_abs_tolerance = 0.1


def _serve_and_score(model, base_url, other_args):
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS + other_args,
        # Compile DeepGEMM kernels lazily for the few shapes the prompts use.
        env={"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"},
    )
    try:
        return {
            "single": [_generate(base_url, [p])[0] for p in PROMPTS],
            "batch": _generate(base_url, PROMPTS),
        }
    finally:
        terminate_and_kill_process_tree(process)


def _generate(base_url, prompts):
    """Prompt logprobs and greedy decode (token, logprob) pairs for ``prompts``, sent as one
    batched request."""
    metas = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    ).json()
    return [
        (
            [lp for lp, _, _ in m["meta_info"]["input_token_logprobs"]],
            [(tok, lp) for lp, tok, _ in m["meta_info"]["output_token_logprobs"]],
        )
        for m in metas
    ]


if __name__ == "__main__":
    unittest.main(verbosity=3)
