"""Multi-adapter EAGLE decoding: adapt the target, never the shared draft.

Guard startup, adapter effects in uniform/mixed batches, graph and eager
target-verify (proved by counters), and accept length above one.

Cross-configuration text equality is not an oracle: reduction-order changes
can flip greedy tokens across batch shapes or restarts. Losslessness is
checked separately by test/manual/lora/run_spec_lora_matrix.py after measuring
that noise floor.
"""

import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=95, stage="base-b", runner_config="1-gpu-large")

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Write a one-sentence story about a brave detective on Mars.",
]
# Ranks 8 and 64 expose mistakes in per-request lora_ranks indexing.
ADAPTERS = [
    ("fact", "algoprog/fact-generation-llama-3.1-8b-instruct-lora"),
    ("guard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
]
# Row kinds of a mixed batch: base first, then every adapter.
ROUTES = [None] + [name for name, _ in ADAPTERS]
SAMPLING = {"temperature": 0, "max_new_tokens": 32}
# Captures the 9-request mixed batch below but not the 24-request one. The
# default limit on an 80 GB GPU (48 with EAGLE) captures both.
DECODE_GRAPH_MAX_BS = 12
WIDE_BATCH = 24


def _decode_passes(base_url, *, require_samples=False):
    """Count graph/eager target-verify forwards; require samples after decoding."""
    response = requests.get(base_url + "/metrics", timeout=30)
    if response.status_code != 200:
        raise AssertionError(
            f"/metrics returned HTTP {response.status_code}: {response.text[:200]}"
        )
    samples = {
        mode: [
            float(value)
            for value in re.findall(
                r'^sglang:cuda_graph_passes_total\{[^}]*mode="'
                + mode
                + r'"[^}]*\}\s+([0-9.eE+-]+)$',
                response.text,
                re.MULTILINE,
            )
        ]
        for mode in ("decode_cuda_graph", "decode_none")
    }
    if require_samples and not any(samples.values()):
        raise AssertionError(
            "/metrics has no decode samples of sglang:cuda_graph_passes_total; "
            "the server needs --enable-metrics"
        )
    return {mode: sum(values) for mode, values in samples.items()}


class TestEagle3MultiLoRA(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                # Canonical EAGLE3 sglang config, as in
                # test/registered/core/test_basic_sanity_eagle3.py: the draft
                # checkpoint is fp16, and bf16 + flashinfer cutlass RMSNorm
                # hits a dtype mismatch on the draft's input_layernorm.
                "--dtype=float16",
                "--attention-backend=triton",
                "--speculative-algorithm=EAGLE3",
                f"--speculative-draft-model-path={DEFAULT_DRAFT_MODEL_EAGLE3}",
                "--speculative-num-steps=5",
                "--speculative-eagle-topk=8",
                "--speculative-num-draft-tokens=32",
                "--enable-lora",
                "--lora-backend=triton",
                "--max-lora-rank=64",
                # +1: the base model occupies a memory-pool slot too, and the
                # batches below co-batch base requests with adapter requests.
                f"--max-loras-per-batch={len(ADAPTERS) + 1}",
                "--mem-fraction-static=0.7",
                f"--cuda-graph-max-bs-decode={DECODE_GRAPH_MAX_BS}",
                "--enable-metrics",
                "--lora-paths",
            ]
            + [f"{name}={path}" for name, path in ADAPTERS],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self, texts, lora_paths):
        response = requests.post(
            self.base_url + "/generate",
            json={"text": texts, "lora_path": lora_paths, "sampling_params": SAMPLING},
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(texts))
        for item in results:
            self.assertTrue(item["text"].strip(), f"empty output in {results}")
        return [item["text"] for item in results]

    def _generate_mixed(self, prompts):
        """Serve every prompt under every route in one batch; outputs by route."""
        texts = self._generate(
            [p for p in prompts for _ in ROUTES],
            [r for _ in prompts for r in ROUTES],
        )
        return {route: texts[i :: len(ROUTES)] for i, route in enumerate(ROUTES)}

    def _assert_adapters_applied(self, by_route, where):
        # The base rows come from the same batch, so a silently dropped
        # adapter would reproduce them exactly.
        for name, _ in ADAPTERS:
            self.assertNotEqual(
                by_route[name],
                by_route[None],
                f"adapter {name} matched the base model on every prompt of "
                f"{where}; LoRA was likely not applied during target-verify",
            )

    def _decode_passes_since(self, before):
        after = _decode_passes(self.base_url, require_samples=True)
        return {mode: after[mode] - before[mode] for mode in after}

    def test_adapters_are_applied_under_speculation(self):
        before = _decode_passes(self.base_url)
        base = self._generate(PROMPTS, [None] * len(PROMPTS))
        for name, _ in ADAPTERS:
            adapted = self._generate(PROMPTS, [name] * len(PROMPTS))
            self.assertNotEqual(
                adapted,
                base,
                f"adapter {name} matched the base model on every prompt; LoRA "
                "was likely not applied during target-verify",
            )
        self._assert_adapters_applied(self._generate_mixed(PROMPTS), "a mixed batch")
        passes = self._decode_passes_since(before)
        self.assertGreater(
            passes["decode_cuda_graph"], 0, f"no target-verify graph replay: {passes}"
        )
        self.assertEqual(
            passes["decode_none"],
            0,
            f"a batch within the capture limit ran eager target-verify: {passes}",
        )

        before = _decode_passes(self.base_url)
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(WIDE_BATCH // len(ROUTES))]
        self._assert_adapters_applied(
            self._generate_mixed(prompts), f"a {WIDE_BATCH}-request eager batch"
        )
        passes = self._decode_passes_since(before)
        self.assertGreater(
            passes["decode_none"],
            0,
            f"the {WIDE_BATCH}-request batch never ran eager target-verify: {passes}",
        )

    def test_speculation_is_active(self):
        self._generate(PROMPTS, [ADAPTERS[0][0]] * len(PROMPTS))
        info = requests.get(self.base_url + "/get_server_info").json()
        accept_length = info["internal_states"][0]["avg_spec_accept_length"]
        self.assertGreater(
            accept_length,
            1.0,
            f"no drafts accepted with LoRA enabled: {accept_length}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
