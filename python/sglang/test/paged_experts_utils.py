"""Shared checks for the paged-experts e2e tests: paged vs unpaged logprobs.

With few resident experts, a prompt routes to more experts than fit at once and is served in
several waves; decode then evicts and pages experts every step. Both servers run with
--enable-deterministic-inference, so wherever the MoE kernel is batch-invariant, prompt and
greedy decode logprobs must equal the unpaged server's bit for bit: any difference is then a
paging bug, not kernel noise.
"""

import requests

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "The ocean's waves crash against the shore in an endless rhythm,",
]
COMMON_ARGS = [
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--disable-shared-experts-fusion",
    "--disable-radix-cache",
    "--enable-deterministic-inference",
    "--attention-backend",
    "triton",
]
PAGED_ARGS = ["--enable-paged-experts", "--paged-experts-num-resident", "8"]
# CUDA graphs on: with K=8 and top-8 routing, decode batch size 1 is captured and every decode
# step decides and pages on the GPU inside the graph; prefill graphs run the paged MoE layers
# as eager breaks.
GRAPH_ARGS = [
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-max-bs-decode",
    "4",
]


class PagedMatchesUnpagedBase(CustomTestCase):
    """Subclass with ``model`` set. Import this module, not the class, so unittest does not
    collect the base class itself."""

    model: str
    #: False for MoE kernels that are not batch-invariant: compare prompt logprobs within a
    #: tolerance, and skip decode, where a near-tie may flip a greedy token.
    exact = True
    max_abs_tolerance = 0.0
    mean_abs_tolerance = 0.0

    @classmethod
    def setUpClass(cls):
        cls.reference = _serve_and_score(model=cls.model, other_args=[])

    def test_logprobs_match(self):
        paged = _serve_and_score(model=self.model, other_args=PAGED_ARGS)
        self._assert_match(reference=self.reference, paged=paged)

    def test_captured_decode_logprobs_match(self):
        reference = _serve_and_score(model=self.model, other_args=GRAPH_ARGS)
        paged = _serve_and_score(model=self.model, other_args=GRAPH_ARGS + PAGED_ARGS)
        self._assert_match(reference=reference, paged=paged)

    def _assert_match(self, reference, paged):
        for prompt, ref, got in zip(PROMPTS, reference, paged):
            with self.subTest(prompt=prompt[:30]):
                if self.exact:
                    self.assertEqual(got, ref)
                    continue
                diffs = [abs(a - b) for a, b in zip(got[0], ref[0]) if a is not None]
                self.assertLessEqual(max(diffs), self.max_abs_tolerance)
                self.assertLessEqual(sum(diffs) / len(diffs), self.mean_abs_tolerance)


def _serve_and_score(model, other_args):
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=COMMON_ARGS + other_args,
    )
    try:
        return _score(base_url=DEFAULT_URL_FOR_TEST)
    finally:
        terminate_and_kill_process_tree(process)


def _score(base_url):
    """Prompt logprobs plus greedy decode (token, logprob) pairs, one request at a time."""
    results = []
    for prompt in PROMPTS:
        meta = requests.post(
            base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        ).json()["meta_info"]
        results.append(
            (
                [lp for lp, _, _ in meta["input_token_logprobs"]],
                [(tok, lp) for lp, tok, _ in meta["output_token_logprobs"]],
            )
        )
    return results
