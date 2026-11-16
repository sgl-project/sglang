"""Kimi-K3.1 (first eight layers, real weights) on the unified pool matches the
static pools in the same mode (CUDA graph, eager): a short case within the
indexer top-k and a long case over twice the top-k, each a chunked prefill with
decode, then a prefix-reusing request. Within the top-k, logprobs agree per
token. Past it, sparse selection is not bitwise deterministic (static runs
differ by about 0.03 on average, up to about 0.4), so only the mean difference
is bounded, and greedy tokens are compared only when every context is within
the top-k.
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=2600, stage="weekly", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3.1-Pro"
# The checkpoint's first eight layers, numbered from 1 as its config numbers
# them: two MLA+DSA layers among KDA layers. Everything else is its own config.
FULL_LAYERS = [4, 8]
# Loading reads every checkpoint shard, though only eight layers are kept.
SERVER_LAUNCH_TIMEOUT = 3600
KDA_LAYERS = [1, 2, 3, 5, 6, 7]
BASE_ARGS = [
    "--trust-remote-code",
    "--skip-tokenizer-init",
    "--tp-size",
    "2",
    "--random-seed",
    "1",
    "--chunked-prefill-size",
    "256",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.6",
]
# New tokens the second request of each case appends to the cached prefix.
CONTINUATION = [200 + i for i in range(64)]


def _prompt(length, offset):
    """A BOS then synthetic tokens; cases take disjoint ranges so neither
    reuses the other's prefix."""
    return [1] + [offset + (i * 37) % 5000 for i in range(length - 1)]


def _text_config(model_path):
    from sglang.srt.hf_transformers.config import get_config

    return get_config(model_path, trust_remote_code=True).text_config


def _model_override(text):
    """The checkpoint's config with only the kept layers."""
    linear = dict(text.linear_attn_config)
    linear.update(full_attn_layers=FULL_LAYERS, kda_layers=KDA_LAYERS)
    dsa = dict(text.dsa_config)
    dsa.update(dsa_layers=FULL_LAYERS)
    return {
        "text_config": {
            "num_hidden_layers": len(FULL_LAYERS) + len(KDA_LAYERS),
            "linear_attn_config": linear,
            "dsa_config": dsa,
        }
    }


class TestKimiK31UnifiedMemoryParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        text = _text_config(MODEL_PATH)
        cls.index_topk = text.dsa_config["index_topk"]
        cls.prompts = {
            "short": _prompt(701, offset=100),
            # 4,097 tokens for a top-k of 2,048.
            "long": _prompt(2 * cls.index_topk + 1, offset=5100),
        }
        assert len(cls.prompts["short"]) + len(CONTINUATION) <= cls.index_topk
        cls.base_args = BASE_ARGS + [
            "--json-model-override-args",
            json.dumps(_model_override(text)),
        ]

    def _generate(self, input_ids, logprob_start_len=0):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
            },
            timeout=300,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        # Each logprob with the length of the context it was predicted from:
        # the prompt's position `logprob_start_len + i` sees that many tokens.
        prompt = [
            (logprob_start_len + i, lp)
            for i, (lp, _, _) in enumerate(meta["input_token_logprobs"])
            if lp is not None
        ]
        output = [
            (len(input_ids) + i, lp, tid)
            for i, (lp, tid, _) in enumerate(meta["output_token_logprobs"])
        ]
        return prompt, output, meta.get("cached_tokens", 0)

    def _run(self, extra_args):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=self.base_args + extra_args,
        )
        try:
            results = {}
            for case, prefix in self.prompts.items():
                first = self._generate(prefix)
                # The same prefix again, then new tokens: a cache hit whose
                # new tokens read the cached latent rows and indexer keys.
                # Prompt logprobs only past the prefix: asking for them from
                # the start would recompute the prefix instead of reusing it.
                second = self._generate(
                    prefix + CONTINUATION, logprob_start_len=len(prefix)
                )
                results[case] = first, second
            return results
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def _assert_close(self, unified, static, what):
        unified_input, unified_output, _ = unified
        static_input, static_output, _ = static
        pairs = [
            (context, u, s, f"prompt token at {context}")
            for (context, u), (_, s) in zip(unified_input, static_input)
        ]
        if all(context <= self.index_topk for context, _, _ in static_output):
            # Every context within the top-k: greedy decoding is deterministic.
            self.assertEqual(
                [tid for _, _, tid in unified_output],
                [tid for _, _, tid in static_output],
                f"{what}: greedy tokens diverged",
            )
            pairs += [
                (context, u, s, f"output token at {context}")
                for (context, u, _), (_, s, _) in zip(unified_output, static_output)
            ]
        self.assertEqual(len(unified_input), len(static_input), what)
        sparse = []
        for context, u, s, where in pairs:
            if context <= self.index_topk:
                # Every token attended: the same rows, read the same way.
                self.assertAlmostEqual(u, s, delta=0.01, msg=f"{what}: {where}")
            else:
                sparse.append(abs(u - s))
        if sparse:
            mean = sum(sparse) / len(sparse)
            self.assertLessEqual(
                mean, 0.1, f"{what}: mean difference past the top-k {mean:.4f}"
            )

    def test_unified_matches_static(self):
        for name, extra in (
            ("cuda graph", []),
            ("eager", ["--disable-cuda-graph"]),
        ):
            with self.subTest(mode=name):
                static = self._run(extra)
                unified = self._run(["--enable-unified-memory", *extra])
                for case in self.prompts:
                    with self.subTest(case=case):
                        self._check_case(unified[case], static[case], name, case)

    def _check_case(self, unified, static, mode, case):
        cached = unified[1][2]
        self.assertGreater(cached, 0, "the second request hit no prefix")
        if case == "long":
            # The extend's context, cached keys included, exceeds the top-k,
            # so the indexer's selection decides what attention reads.
            self.assertGreater(cached, self.index_topk)
        what = f"{mode}, {case}"
        self._assert_close(unified[0], static[0], f"{what}, first request")
        self._assert_close(unified[1], static[1], f"{what}, prefix hit")


if __name__ == "__main__":
    unittest.main()
