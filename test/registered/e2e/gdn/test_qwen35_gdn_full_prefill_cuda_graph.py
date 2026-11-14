"""Full prefill CUDA graphs on the hybrid Qwen3.5 GDN architecture.

Breakable prefill graphs run the GDN extend eagerly between captured segments;
full graphs capture it on static per-bucket tables. Both replay the same bucket
shapes, so every probe must match bit for bit: padded multi-request batches, and
extends that resume from a radix-cached state.
"""

import random
import re
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3.5-0.8B"
SERVER_ARGS = [
    # Full graphs need the Triton GDN extend (SM90/SM100 default to FlashInfer).
    "--linear-attn-prefill-backend",
    "triton",
    # With FA3 (the SM90 default) full and breakable graphs differ on dense models too.
    "--attention-backend",
    "flashinfer",
    "--mem-fraction-static",
    "0.5",
    "--cuda-graph-max-bs-decode",
    "4",
    "--enable-metrics",
    "--random-seed",
    "42",
]
# Radix-cached GDN states sit on 64-token boundaries: 576 of these 600 hit.
PREFIX_LEN = 600


def _prompt(rng, n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


def _prefill_passes():
    """Prefill forwards so far, replayed on a graph and run eagerly."""
    metrics = requests.get(DEFAULT_URL_FOR_TEST + "/metrics", timeout=30).text
    passes = {}
    for mode in ("prefill_cuda_graph", "prefill_none"):
        matches = re.findall(
            rf'^sglang:cuda_graph_passes_total\{{[^}}]*mode="{mode}"[^}}]*\}}'
            r"\s+([0-9.eE+-]+)$",
            metrics,
            re.MULTILINE,
        )
        passes[mode] = sum(map(float, matches), 0.0)
    return passes


def _generate(prompts, *, logprob_start_len):
    before = _prefill_passes()
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
        },
        timeout=300,
    )
    response.raise_for_status()
    after = _prefill_passes()
    outs = [
        {
            "output_ids": out["output_ids"],
            "input_logprobs": [x[0] for x in out["meta_info"]["input_token_logprobs"]],
            "output_logprobs": [
                x[0] for x in out["meta_info"]["output_token_logprobs"]
            ],
            "cached_tokens": out["meta_info"]["cached_tokens"],
        }
        for out in response.json()
    ]
    assert len(outs) == len(prompts), (len(outs), len(prompts))
    passes = {mode: after[mode] - before[mode] for mode in after}
    return {"outputs": outs, "prefill_passes": passes}


def _probes():
    rng = random.Random(0)
    probes = {}
    requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=60).raise_for_status()
    # One padded forward over several request slots, scoring every prompt token.
    probes["batch"] = _generate(
        [_prompt(rng, n) for n in (5, 60, 200, 31)], logprob_start_len=0
    )
    probes["exact_bucket"] = _generate([_prompt(rng, 16)], logprob_start_len=0)
    # Prompt logprobs from token 0 would disable the prefix match below.
    prefix = _prompt(rng, PREFIX_LEN)
    probes["prefix_seed"] = _generate([prefix], logprob_start_len=-1)
    probes["prefix_extend"] = _generate(
        [prefix + _prompt(rng, n) for n in (3, 70, 300)],
        logprob_start_len=PREFIX_LEN,
    )
    return probes


def _run(prefill_backend):
    # Breakable graphs then run the GDN extend eagerly; full graphs ignore the switch.
    with envs.SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND.override(False):
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS + ["--cuda-graph-backend-prefill", prefill_backend],
        )
    try:
        return _probes()
    finally:
        terminate_and_kill_process_tree(process)


class TestQwen35GDNFullPrefillCudaGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.breakable = _run("breakable")
        cls.full = _run("full")

    def test_every_prefill_replays_a_graph(self):
        for run in (self.breakable, self.full):
            for name, probe in run.items():
                with self.subTest(probe=name):
                    passes = probe["prefill_passes"]
                    self.assertGreater(passes["prefill_cuda_graph"], 0)
                    self.assertEqual(passes["prefill_none"], 0)
        for out in self.full["prefix_extend"]["outputs"]:
            self.assertGreater(out["cached_tokens"], 0)

    def test_full_graphs_match_breakable_graphs(self):
        """Full prefill graphs crashed at capture on hybrid GDN models; they must
        score every token exactly as breakable graphs do."""
        for name, probe in self.breakable.items():
            for i, (expected, actual) in enumerate(
                zip(probe["outputs"], self.full[name]["outputs"], strict=True)
            ):
                with self.subTest(probe=name, request=i):
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
