"""Phase 2 NPU tests: Ascend attention path consistency (default vs FIA vs FA3).

The Ascend attention backend has three selectable execution paths:

* default - ``torch_npu._npu_flash_attention_qlens`` / native SDPA,
* FA3 - ``flash_attn_npu_v3`` when ``ASCEND_USE_FA=1`` (requires the
  ``flash_attn_npu_v3`` package),
* FIA - torch_npu fused-infer-attention when ``ASCEND_USE_FIA=1``.

On A3, each case launches one server per available path with identical inputs
and compares greedy token IDs and logprobs. FA3 participates when its extension
is available. On A5, the default prefill operator cannot run on Ascend 950DT,
so the comparisons skip and separate FIA cases validate generation outputs.
"""

import math
import os
import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_rl_utils import (
    _is_ascend_950,
    assert_logprob_signature_equal,
    generate,
    logprob_signature,
    npu_rl_env,
    npu_server,
    post,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1500, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=1500, suite="nightly-test-1-npu-a3", nightly=True)

MAX_NEW_TOKENS = 8

PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
)
# ~1200 tokens: forces multi-chunk prefill when --chunked-prefill-size is 256.
LONG_PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 120)

SAMPLING = {
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "ignore_eos": True,
}

BASE_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--mem-fraction-static",
    "0.5",
    "--context-length",
    "2048",
)

SPEC_ARGS = (
    "--dtype",
    "float16",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--cuda-graph-max-bs-decode",
    "4",
    "--mem-fraction-static",
    "0.7",
    "--cuda-graph-backend-prefill=disabled",
)

HAVE_QWEN3_06B = os.path.isfile(os.path.join(QWEN3_0_6B_WEIGHTS_PATH, "config.json"))
HAVE_QWEN3_8B = os.path.isfile(os.path.join(QWEN3_8B_WEIGHTS_PATH, "config.json"))
HAVE_EAGLE3_DRAFT = os.path.isfile(
    os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, "config.json")
)
IS_A5 = _is_ascend_950()
A5_COMPARISON_SKIP = "A5 uses FIA-only coverage; default prefill is unsupported"
FIA_ENV = {"ASCEND_USE_FIA": "1", "ASCEND_USE_FA": "0"}


def _fa3_available():
    try:
        import flash_attn_npu_v3  # noqa: F401
    except Exception:
        return False
    return True


def _available_backends():
    """Return [(name, env-overrides)] for every usable attention path."""
    backends = [
        ("default", {"ASCEND_USE_FIA": "0", "ASCEND_USE_FA": "0"}),
        ("fia", FIA_ENV),
    ]
    if _fa3_available():
        backends.append(("fa3", {"ASCEND_USE_FIA": "0", "ASCEND_USE_FA": "1"}))
    return backends


def _pairs(results):
    names = list(results)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            yield left, right


def _collect_single(base_url):
    return tuple(
        tuple(generate(base_url, prompt, max_new_tokens=MAX_NEW_TOKENS)["output_ids"])
        for prompt in PROMPTS
    )


def _collect_mixed(base_url):
    """One batch of requests with very different prompt lengths."""
    body = post(
        base_url,
        "/generate",
        {
            "text": [PROMPTS[0], LONG_PROMPT[:80], PROMPTS[2], PROMPTS[3]],
            "sampling_params": dict(SAMPLING),
        },
        timeout=300,
    ).json()
    outputs = body if isinstance(body, list) else body.get("data", [body])
    return tuple(tuple(item["output_ids"]) for item in outputs)


def _collect_logprob(base_url):
    return logprob_signature(base_url, PROMPTS, max_new_tokens=MAX_NEW_TOKENS)


def _collect_chunked(base_url):
    body = generate(base_url, LONG_PROMPT, max_new_tokens=MAX_NEW_TOKENS, timeout=300)
    return tuple(body["output_ids"])


def _run_backend(model_path, extra_args, collectors, env_extra):
    with npu_server(
        model_path,
        extra_args=extra_args,
        env={**npu_rl_env(), **env_extra},
    ) as base_url:
        return {key: collector(base_url) for key, collector in collectors.items()}


def _run_per_backend(extra_args, collectors, *, model_path=QWEN3_0_6B_WEIGHTS_PATH):
    """Collect once per available backend, one server at a time."""
    return {
        name: _run_backend(model_path, extra_args, collectors, env_extra)
        for name, env_extra in _available_backends()
    }


@unittest.skipIf(IS_A5, A5_COMPARISON_SKIP)
@unittest.skipUnless(HAVE_QWEN3_06B, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
class TestNpuAscendAttentionConsistency(CustomTestCase):
    """Single / mixed-batch / logprob outputs must agree across attention paths."""

    @classmethod
    def setUpClass(cls):
        cls.results = _run_per_backend(
            BASE_ARGS,
            {
                "single": _collect_single,
                "mixed": _collect_mixed,
                "logprob": _collect_logprob,
            },
        )

    def test_01_single_prompt_consistency(self):
        for left, right in _pairs(self.results):
            self.assertEqual(
                self.results[left]["single"],
                self.results[right]["single"],
                f"single prompt: {left} vs {right} differ",
            )

    def test_02_mixed_batch_consistency(self):
        for left, right in _pairs(self.results):
            self.assertEqual(
                self.results[left]["mixed"],
                self.results[right]["mixed"],
                f"mixed batch: {left} vs {right} differ",
            )

    def test_03_logprob_consistency(self):
        for left, right in _pairs(self.results):
            assert_logprob_signature_equal(
                self.results[left]["logprob"],
                self.results[right]["logprob"],
                f"logprob: {left} vs {right}",
            )


@unittest.skipIf(IS_A5, A5_COMPARISON_SKIP)
@unittest.skipUnless(HAVE_QWEN3_06B, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
class TestNpuAscendAttentionChunkedPrefillConsistency(CustomTestCase):
    """Multi-chunk prefill must produce the same tokens on every path."""

    @classmethod
    def setUpClass(cls):
        cls.results = _run_per_backend(
            (*BASE_ARGS, "--chunked-prefill-size", "256"),
            {"chunked": _collect_chunked},
        )

    def test_01_chunked_prefill_consistency(self):
        for left, right in _pairs(self.results):
            self.assertEqual(
                self.results[left]["chunked"],
                self.results[right]["chunked"],
                f"chunked prefill: {left} vs {right} differ",
            )


@unittest.skipIf(IS_A5, A5_COMPARISON_SKIP)
@unittest.skipUnless(
    HAVE_EAGLE3_DRAFT, f"draft model not found: {QWEN3_8B_EAGLE3_WEIGHTS_PATH}"
)
@unittest.skipUnless(HAVE_QWEN3_8B, f"model not found: {QWEN3_8B_WEIGHTS_PATH}")
class TestNpuAscendAttentionSpecConsistency(CustomTestCase):
    """Speculative decoding (EAGLE3) must select the same tokens on every path."""

    @classmethod
    def setUpClass(cls):
        cls.results = _run_per_backend(
            SPEC_ARGS, {"spec": _collect_single}, model_path=QWEN3_8B_WEIGHTS_PATH
        )

    def test_01_speculative_decoding_consistency(self):
        for left, right in _pairs(self.results):
            self.assertEqual(
                self.results[left]["spec"],
                self.results[right]["spec"],
                f"speculative: {left} vs {right} differ",
            )


@unittest.skipUnless(IS_A5, "FIA-only coverage is for A5")
class TestNpuAscendAttentionFia(CustomTestCase):
    """Validate FIA generation without a cross-backend reference on A5."""

    def _assert_token_rows(self, rows):
        self.assertEqual(len(rows), len(PROMPTS))
        for token_ids in rows:
            self.assertEqual(len(token_ids), MAX_NEW_TOKENS)

    @unittest.skipUnless(HAVE_QWEN3_06B, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
    def test_single_mixed_and_logprobs(self):
        results = _run_backend(
            QWEN3_0_6B_WEIGHTS_PATH,
            BASE_ARGS,
            {
                "single": _collect_single,
                "mixed": _collect_mixed,
                "logprob": _collect_logprob,
            },
            FIA_ENV,
        )
        self._assert_token_rows(results["single"])
        self._assert_token_rows(results["mixed"])
        self.assertEqual(len(results["logprob"]), len(PROMPTS))
        for token_ids, logprobs in results["logprob"]:
            self.assertEqual(len(token_ids), MAX_NEW_TOKENS)
            self.assertEqual(len(logprobs), len(token_ids))
            self.assertTrue(all(math.isfinite(value) for value in logprobs), logprobs)

    @unittest.skipUnless(HAVE_QWEN3_06B, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
    def test_chunked_prefill(self):
        results = _run_backend(
            QWEN3_0_6B_WEIGHTS_PATH,
            (*BASE_ARGS, "--chunked-prefill-size", "256"),
            {"chunked": _collect_chunked},
            FIA_ENV,
        )
        self.assertEqual(len(results["chunked"]), MAX_NEW_TOKENS)

    @unittest.skipUnless(
        HAVE_EAGLE3_DRAFT, f"draft model not found: {QWEN3_8B_EAGLE3_WEIGHTS_PATH}"
    )
    @unittest.skipUnless(HAVE_QWEN3_8B, f"model not found: {QWEN3_8B_WEIGHTS_PATH}")
    def test_speculative_decoding(self):
        results = _run_backend(
            QWEN3_8B_WEIGHTS_PATH, SPEC_ARGS, {"spec": _collect_single}, FIA_ENV
        )
        self._assert_token_rows(results["spec"])


if __name__ == "__main__":
    unittest.main()
