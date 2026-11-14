"""NPU tests: deterministic inference for RL rollouts (BS1 vs BSN bitwise).

Phase 2 of https://github.com/sgl-project/sglang/issues/48822. RL rollouts
feed generated tokens and logprobs back into the trainer; a rollout engine
that returns different logprobs for the same prompt depending on how many
other requests shared the batch silently poisons the training signal. With
``--enable-deterministic-inference`` the NPU backend swaps in batch-invariant
kernels (``torch.ops.batch_invariant_ops.npu_*``), so the same request must
produce bitwise-identical token ids and logprobs at batch size 1, at batch
size N and after reordering the batch.

The suite is capability-gated: the batch-invariant extension package is a
native build and may be absent from a container, in which case both classes
skip with an actionable reason instead of failing.

[Test Category] RL consistency
[Test Target] --enable-deterministic-inference, --rl-on-policy-target
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="nightly-1-npu-a3", nightly=True)
register_npu_ci(est_time=500, suite="base-b-test-1-npu-a3")

MAX_NEW_TOKENS = 8

PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
)

NPU_SERVER_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": (
        "expandable_segments:True,pin_memory_expandable_segments:True"
    ),
}


def _batch_invariant_ops_available():
    """Whether the native batch_invariant_ops extension is importable."""
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401  # the ops are NPU-only
        import batch_invariant_ops  # noqa: F401

        torch.ops.batch_invariant_ops.npu_mm_batch_invariant
        torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
    except Exception:
        return False
    return True


def _flush_cache(base_url):
    response = requests.post(base_url + "/flush_cache", timeout=60)
    response.raise_for_status()


def _post_generate(base_url, payload, timeout=180):
    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise AssertionError(f"generate returned error: {body['error']}")
    return body


def _signature_of(body):
    """(token ids, logprobs) of one /generate response entry."""
    meta = body.get("meta_info", {})
    logprobs = tuple(item[0] for item in (meta.get("output_token_logprobs") or []))
    if not logprobs:
        raise AssertionError("no output_token_logprobs returned")
    return (tuple(body["output_ids"]), logprobs)


def _request_signatures(base_url, prompts):
    payload = {
        "text": list(prompts),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "return_logprob": True,
    }
    body = _post_generate(base_url, payload)
    if isinstance(body, list):
        return [_signature_of(item) for item in body]
    outputs = body.get("data", [body])
    return [_signature_of(item) for item in outputs]


def _collect_determinism_trials(base_url, prompts):
    """BS=1 reference, a BS=N batch, and a reordered BS=N batch."""
    trials = {}

    _flush_cache(base_url)
    trials["bs1"] = [
        _request_signatures(base_url, [prompt])[0] for prompt in prompts
    ]

    _flush_cache(base_url)
    trials["bsn"] = _request_signatures(base_url, prompts)

    _flush_cache(base_url)
    shuffled = list(prompts)
    shuffled.reverse()
    trials["bsn_reversed"] = [
        None,
    ] * len(shuffled)
    results = _request_signatures(base_url, shuffled)
    for position, prompt in enumerate(shuffled):
        trials["bsn_reversed"][prompts.index(prompt)] = results[position]

    return trials


def _assert_bitwise_determinism(trials, label):
    for trial_name, signatures in trials.items():
        for index, signature in enumerate(signatures):
            self_reference = trials["bs1"][index]
            if signature != self_reference:
                raise AssertionError(
                    f"{label}: {trial_name} diverged from bs1 for prompt {index}\n"
                    f"  bs1:     {self_reference}\n"
                    f"  {trial_name}: {signature}"
                )


DETERMINISTIC_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.5",
    "--context-length",
    "1024",
    "--enable-deterministic-inference",
]


def _deterministic_skip_reason():
    if not _batch_invariant_ops_available():
        return (
            "batch_invariant_ops native extension is not installed; "
            "build and install the torch extension package first "
            "(torch.ops.batch_invariant_ops.npu_*)"
        )
    if not os.path.isfile(os.path.join(QWEN3_0_6B_WEIGHTS_PATH, "config.json")):
        return f"Qwen3-0.6B model config not found at {QWEN3_0_6B_WEIGHTS_PATH}"
    return None


class _DeterministicServerTestCase(CustomTestCase):
    extra_server_args: list = []
    requires_on_policy_target = False

    @classmethod
    def setUpClass(cls):
        reason = _deterministic_skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, **NPU_SERVER_ENV},
            other_args=DETERMINISTIC_ARGS + list(cls.extra_server_args),
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.trials = _collect_determinism_trials(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestNpuDeterministicInference(_DeterministicServerTestCase):
    """Same prompt, same tokens and logprobs at BS1, BSN and reversed order."""

    def test_01_bs1_matches_bsn_bitwise(self):
        _assert_bitwise_determinism(self.trials, "deterministic inference")


class TestNpuDeterministicInferenceOnPolicyTarget(_DeterministicServerTestCase):
    """--rl-on-policy-target fsdp keeps inference deterministic.

    The on-policy target selects the numerical contract the rollout must
    match (an FSDP trainer); enabling it must not weaken batch invariance,
    and the server must actually accept and register the flag.
    """

    extra_server_args = ["--rl-on-policy-target", "fsdp"]

    def test_01_bs1_matches_bsn_bitwise(self):
        _assert_bitwise_determinism(self.trials, "rl_on_policy_target=fsdp")

    def test_02_server_registered_the_target(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        if response.status_code == 404:
            # Older builds expose the same payload under /get_server_info.
            response = requests.get(
                self.base_url + "/get_server_info", timeout=30
            )
        response.raise_for_status()
        info = response.json()
        args = info.get("server_args", info)
        self.assertEqual(
            args.get("rl_on_policy_target"),
            "fsdp",
            "server did not register --rl-on-policy-target fsdp",
        )


if __name__ == "__main__":
    unittest.main()
