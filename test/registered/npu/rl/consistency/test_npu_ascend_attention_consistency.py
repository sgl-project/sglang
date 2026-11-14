"""NPU tests: Ascend attention path consistency (FA3 vs FIA).

Phase 2 of https://github.com/sgl-project/sglang/issues/48822. The Ascend
attention backend has three selectable execution paths:

* default - ``torch_npu._npu_flash_attention_qlens`` / native SDPA,
* FA3 - ``flash_attn_npu_v3`` when ``ASCEND_USE_FA=1`` (requires the
  ``flash_attn_npu_v3`` package),
* FIA - torch_npu fused-infer-attention when ``ASCEND_USE_FIA=1``.

RL training alternates rollout and training steps on the same models, and A5
products pin FIA while A3 machines default elsewhere, so greedy token ids and
logprobs must agree across these paths. Every case below launches one server
per available path with identical inputs and compares the responses.

The FA3 side is capability-gated: when ``flash_attn_npu_v3`` is not installed
the FA3 comparisons are skipped with a reason instead of failing, while the
default-vs-FIA comparison still runs (per the RFC's skip rules).

[Test Category] RL consistency
[Test Target] ASCEND_USE_FA / ASCEND_USE_FIA / attention path parity
"""

import os
import shutil
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1500, suite="nightly-1-npu-a3", nightly=True)
register_npu_ci(est_time=1500, suite="base-b-test-1-npu-a3")

LOGPROB_TOLERANCE = 5e-2
MAX_NEW_TOKENS = 8

PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
)
# ~1200 tokens: forces multi-chunk prefill when --chunked-prefill-size is 256.
LONG_PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 120)

NPU_SERVER_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": (
        "expandable_segments:True,pin_memory_expandable_segments:True"
    ),
}


def _fa3_available():
    try:
        import flash_attn_npu_v3  # noqa: F401
    except Exception:
        return False
    return True


def _available_backends():
    """Return [(name, env-overrides)] for every usable attention path."""
    backends = [
        ("default", {}),
        ("fia", {"ASCEND_USE_FIA": "1"}),
    ]
    if _fa3_available():
        backends.append(("fa3", {"ASCEND_USE_FA": "1"}))
    return backends


def _require_two_backends(testcase_or_class):
    backends = _available_backends()
    if len(backends) < 2:
        raise unittest.SkipTest(
            "attention consistency needs at least two paths: "
            "flash_attn_npu_v3 is not installed so FA3 is unavailable"
        )
    return backends


def _post_generate(base_url, payload, timeout=180):
    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise AssertionError(f"generate returned error: {body['error']}")
    return body


def _collect_single(base_url):
    result = {}
    for index, prompt in enumerate(PROMPTS):
        body = _post_generate(
            base_url,
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
            },
        )
        result[index] = tuple(body["output_ids"])
    return result


def _collect_mixed(base_url):
    """One batch of requests with very different prompt lengths."""
    body = _post_generate(
        base_url,
        {
            "text": [PROMPTS[0], LONG_PROMPT[:80], PROMPTS[2], PROMPTS[3]],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        },
    )
    outputs = body if isinstance(body, list) else body.get("data", [body])
    return {i: tuple(item["output_ids"]) for i, item in enumerate(outputs)}


def _collect_logprob(base_url):
    result = {}
    for index, prompt in enumerate(PROMPTS):
        body = _post_generate(
            base_url,
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
        )
        meta = body.get("meta_info", {})
        logprobs = tuple(
            item[0] for item in (meta.get("output_token_logprobs") or [])
        )
        if not logprobs:
            raise AssertionError(f"no output_token_logprobs returned for {prompt!r}")
        result[index] = (tuple(body["output_ids"]), logprobs)
    return result


def _collect_chunked(base_url):
    body = _post_generate(
        base_url,
        {
            "text": LONG_PROMPT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        },
        timeout=300,
    )
    return {"long": tuple(body["output_ids"])}


def _launch_collect_kill(model, other_args, collectors, env_extra):
    """Launch one server, run the collectors, always kill the server."""
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        env={**os.environ, **NPU_SERVER_ENV, **env_extra},
        other_args=other_args,
    )
    try:
        wait_for_healthy(DEFAULT_URL_FOR_TEST)
        return {
            name: collector(DEFAULT_URL_FOR_TEST)
            for name, collector in collectors.items()
        }
    finally:
        kill_process_tree(process.pid)
        # The next backend's server launches on the same device immediately;
        # wait for the driver to finish teardown first.
        _wait_for_npu_idle()


def wait_for_healthy(base_url, timeout=600):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(base_url + "/health", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"server not healthy within {timeout}s")


def _wait_for_npu_idle(timeout=180):
    """Wait until the previous server handed its NPU back to the driver.

    SIGKILL'd servers leave device teardown racing the next launch, which
    surfaces as ``SetDevice ... error code 507033: Failed to start the
    device``. Poll the first device's HBM usage rate instead of sleeping for
    a guessed duration (the RFC forbids fixed sleeps).
    """
    import re
    import subprocess
    import time

    if not shutil.which("npu-smi"):
        time.sleep(20)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["npu-smi", "info", "-t", "usages", "-i", "0", "-c", "0"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            match = re.search(r"HBM Usage Rate\(%\)\s*:\s*(\d+)", out)
            if match and int(match.group(1)) <= 5:
                return
        except Exception:
            pass
        time.sleep(2)


def _assert_signatures_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: outputs differ\n  {actual}\nvs\n  {expected}")


def _assert_logprobs_within(actual, expected, label):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: logprob lengths differ")
    for position, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > LOGPROB_TOLERANCE:
            raise AssertionError(
                f"{label}: logprob differs at token {position}: {a} vs {e} "
                f"(tolerance {LOGPROB_TOLERANCE})"
            )


BASE_ARGS = (
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.5",
    "--context-length",
    "2048",
)


@unittest.skipUnless(
    os.path.isfile(os.path.join(QWEN3_0_6B_WEIGHTS_PATH, "config.json")),
    f"Qwen3-0.6B model config not found at {QWEN3_0_6B_WEIGHTS_PATH}",
)
class TestNpuAscendAttentionConsistency(CustomTestCase):
    """Single / mixed-batch / logprob outputs must agree across attention paths."""

    @classmethod
    def setUpClass(cls):
        backends = _require_two_backends(cls)
        collectors = {
            "single": _collect_single,
            "mixed": _collect_mixed,
            "logprob": _collect_logprob,
        }
        cls.results = {}
        for name, env_extra in backends:
            cls.results[name] = _launch_collect_kill(
                QWEN3_0_6B_WEIGHTS_PATH,
                list(BASE_ARGS),
                collectors,
                env_extra,
            )

    def _compare(self, pattern, checker=None):
        backends = list(self.results)
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                left, right = backends[i], backends[j]
                label = f"{pattern}: {left} vs {right}"
                if pattern == "logprob":
                    left_data = self.results[left][pattern]
                    right_data = self.results[right][pattern]
                    for index in left_data:
                        self.assertEqual(
                            left_data[index][0], right_data[index][0], f"{label}: ids"
                        )
                        _assert_logprobs_within(
                            left_data[index][1], right_data[index][1], label
                        )
                else:
                    _assert_signatures_equal(
                        self.results[left][pattern],
                        self.results[right][pattern],
                        label,
                    )

    def test_01_single_prompt_consistency(self):
        self._compare("single")

    def test_02_mixed_batch_consistency(self):
        self._compare("mixed")

    def test_03_logprob_consistency(self):
        self._compare("logprob")


@unittest.skipUnless(
    os.path.isfile(os.path.join(QWEN3_0_6B_WEIGHTS_PATH, "config.json")),
    f"Qwen3-0.6B model config not found at {QWEN3_0_6B_WEIGHTS_PATH}",
)
class TestNpuAscendAttentionChunkedPrefillConsistency(CustomTestCase):
    """Multi-chunk prefill must produce the same tokens on every path."""

    @classmethod
    def setUpClass(cls):
        backends = _require_two_backends(cls)
        collectors = {"chunked": _collect_chunked}
        cls.results = {}
        for name, env_extra in backends:
            cls.results[name] = _launch_collect_kill(
                QWEN3_0_6B_WEIGHTS_PATH,
                list(BASE_ARGS) + ["--chunked-prefill-size", "256"],
                collectors,
                env_extra,
            )

    def test_01_chunked_prefill_consistency(self):
        backends = list(self.results)
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                left, right = backends[i], backends[j]
                _assert_signatures_equal(
                    self.results[left]["chunked"],
                    self.results[right]["chunked"],
                    f"chunked prefill: {left} vs {right}",
                )


@unittest.skipUnless(
    os.path.isfile(os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, "config.json")),
    f"Qwen3-8B EAGLE3 draft model not found at {QWEN3_8B_EAGLE3_WEIGHTS_PATH}",
)
@unittest.skipUnless(
    os.path.isfile(os.path.join(QWEN3_8B_WEIGHTS_PATH, "config.json")),
    f"Qwen3-8B model config not found at {QWEN3_8B_WEIGHTS_PATH}",
)
class TestNpuAscendAttentionSpecConsistency(CustomTestCase):
    """Speculative decoding (EAGLE3) must select the same tokens on every path.

    Follows the launch recipe of the NPU EAGLE3 sanity suite (float16 target,
    1 draft step, topk 1, 2 draft tokens) so the comparison exercises the
    production spec path of each attention backend.
    """

    SPEC_ARGS = [
        "--dtype",
        "float16",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--trust-remote-code",
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
    ]

    @classmethod
    def setUpClass(cls):
        backends = _require_two_backends(cls)
        cls.results = {}
        for name, env_extra in backends:
            cls.results[name] = _launch_collect_kill(
                QWEN3_8B_WEIGHTS_PATH,
                list(cls.SPEC_ARGS),
                {"spec": _collect_single},
                env_extra,
            )

    def test_01_speculative_decoding_consistency(self):
        backends = list(self.results)
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                left, right = backends[i], backends[j]
                _assert_signatures_equal(
                    self.results[left]["spec"],
                    self.results[right]["spec"],
                    f"speculative: {left} vs {right}",
                )


if __name__ == "__main__":
    unittest.main()
