"""Phase 2 NPU tests: Ascend attention path consistency (default vs FIA vs FA3).

The Ascend attention backend has three selectable execution paths:

* default - ``torch_npu._npu_flash_attention_qlens`` / native SDPA,
* FA3 - ``flash_attn_npu_v3`` when ``ASCEND_USE_FA=1`` (requires the
  ``flash_attn_npu_v3`` package),
* FIA - torch_npu fused-infer-attention when ``ASCEND_USE_FIA=1``.

RL training alternates rollout and training steps on the same models, and A5
products pin FIA while A3 machines default elsewhere, so greedy token ids and
logprobs must agree across these paths. Each case launches one server per
available path with identical inputs and compares the responses.

The FA3 side is capability-gated: when ``flash_attn_npu_v3`` is not installed
FA3 simply joins no comparison, and default-versus-FIA still runs, so the suite
never fails for a missing optional package.
"""

import os
import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_rl_utils import (
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


def _fa3_available():
    try:
        import flash_attn_npu_v3  # noqa: F401
    except Exception:
        return False
    return True


def _available_backends():
    """Return [(name, env-overrides)] for every usable attention path."""
    backends = [("default", {}), ("fia", {"ASCEND_USE_FIA": "1"})]
    if _fa3_available():
        backends.append(("fa3", {"ASCEND_USE_FA": "1"}))
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


def _run_per_backend(extra_args, collectors):
    """Collect once per available backend, one server at a time."""
    results = {}
    for name, env_extra in _available_backends():
        with npu_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            extra_args=extra_args,
            env={**npu_rl_env(), **env_extra},
        ) as base_url:
            results[name] = {
                key: collector(base_url) for key, collector in collectors.items()
            }
    return results


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


@unittest.skipUnless(
    HAVE_EAGLE3_DRAFT, f"draft model not found: {QWEN3_8B_EAGLE3_WEIGHTS_PATH}"
)
@unittest.skipUnless(HAVE_QWEN3_8B, f"model not found: {QWEN3_8B_WEIGHTS_PATH}")
class TestNpuAscendAttentionSpecConsistency(CustomTestCase):
    """Speculative decoding (EAGLE3) must select the same tokens on every path."""

    @classmethod
    def setUpClass(cls):
        cls.results = _run_per_backend(SPEC_ARGS, {"spec": _collect_single})

    def test_01_speculative_decoding_consistency(self):
        for left, right in _pairs(self.results):
            self.assertEqual(
                self.results[left]["spec"],
                self.results[right]["spec"],
                f"speculative: {left} vs {right} differ",
            )


if __name__ == "__main__":
    unittest.main()
