"""Phase 2 NPU tests: deterministic inference for RL rollouts (BS1 vs BSN bitwise).

RL rollouts feed generated tokens and logprobs back into the trainer; a rollout
engine that returns different logprobs for the same prompt depending on how many
other requests shared the batch silently poisons the training signal. With
``--enable-deterministic-inference`` the NPU backend swaps in batch-invariant
kernels (``torch.ops.batch_invariant_ops.npu_*``), so the same request must
produce bitwise-identical token ids and logprobs at batch size 1, at batch size
N, and after reordering the batch.

The suite is capability-gated: the batch-invariant extension is a native build
that may be absent from a container, in which case every case skips with an
actionable reason instead of failing.
"""

import os
import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    flush_cache,
    get,
    npu_rl_env,
    npu_server,
    post,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-1-npu-a3", nightly=True)

MAX_NEW_TOKENS = 8
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
)
SAMPLING = {
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "ignore_eos": True,
}

DETERMINISTIC_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--mem-fraction-static",
    "0.5",
    "--context-length",
    "1024",
    "--enable-deterministic-inference",
)

HAVE_MODEL = os.path.isfile(os.path.join(QWEN3_0_6B_WEIGHTS_PATH, "config.json"))


def _batch_invariant_ops_available():
    """Whether the native batch_invariant_ops extension is importable."""
    try:
        import batch_invariant_ops  # noqa: F401
        import torch
        import torch_npu  # noqa: F401

        torch.ops.batch_invariant_ops.npu_mm_batch_invariant
        torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
    except Exception:
        return False
    return True


HAVE_BATCH_INVARIANT_OPS = _batch_invariant_ops_available()

SKIP_REASON = (
    "batch_invariant_ops native extension is not installed; build and install "
    "the torch extension package plus the matching CANN vendor op package first"
)


def _signature_of(body):
    """(token ids, logprobs) of one /generate response entry.

    Logprobs are compared without rounding: batch invariance is a bitwise
    property, so a rounded comparison would hide the drift this suite exists to
    catch.
    """
    meta = body.get("meta_info", {})
    logprobs = tuple(item[0] for item in (meta.get("output_token_logprobs") or ()))
    if not logprobs:
        raise AssertionError("no output_token_logprobs returned")
    return (tuple(body["output_ids"]), logprobs)


def _request_signatures(base_url, prompts):
    body = post(
        base_url,
        "/generate",
        {
            "text": list(prompts),
            "sampling_params": dict(SAMPLING),
            "return_logprob": True,
        },
        timeout=300,
    ).json()
    if isinstance(body, list):
        return [_signature_of(item) for item in body]
    return [_signature_of(item) for item in body.get("data", [body])]


def _collect_trials(base_url):
    """BS=1 reference, a BS=N batch, and a reordered BS=N batch."""
    flush_cache(base_url)
    bs1 = [_request_signatures(base_url, [prompt])[0] for prompt in PROMPTS]

    flush_cache(base_url)
    bsn = _request_signatures(base_url, PROMPTS)

    flush_cache(base_url)
    shuffled = list(reversed(PROMPTS))
    reversed_results = _request_signatures(base_url, shuffled)
    bsn_reversed = [None] * len(shuffled)
    for position, prompt in enumerate(shuffled):
        bsn_reversed[PROMPTS.index(prompt)] = reversed_results[position]

    return {"bs1": bs1, "bsn": bsn, "bsn_reversed": bsn_reversed}


def _assert_bitwise_determinism(testcase, trials, label):
    for trial_name, signatures in trials.items():
        for index, signature in enumerate(signatures):
            testcase.assertEqual(
                signature,
                trials["bs1"][index],
                f"{label}: {trial_name} diverged from bs1 for prompt {index}\n"
                f"  bs1:     {trials['bs1'][index]}\n"
                f"  {trial_name}: {signature}",
            )


class _DeterministicServerCase:
    extra_server_args = ()

    @classmethod
    def setUpClass(cls):
        env = npu_rl_env()
        cls._server = npu_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            extra_args=(*DETERMINISTIC_ARGS, *cls.extra_server_args),
            env=env,
        )
        cls.base_url = cls._server.__enter__()
        cls.trials = _collect_trials(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)


@unittest.skipUnless(HAVE_MODEL, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
@unittest.skipUnless(HAVE_BATCH_INVARIANT_OPS, SKIP_REASON)
class TestNpuDeterministicInference(_DeterministicServerCase, CustomTestCase):
    """Same prompt, same tokens and logprobs at BS1, BSN and reversed order."""

    def test_01_bs1_matches_bsn_bitwise(self):
        _assert_bitwise_determinism(self, self.trials, "deterministic inference")


@unittest.skipUnless(HAVE_MODEL, f"model not found: {QWEN3_0_6B_WEIGHTS_PATH}")
@unittest.skipUnless(HAVE_BATCH_INVARIANT_OPS, SKIP_REASON)
class TestNpuDeterministicInferenceOnPolicyTarget(
    _DeterministicServerCase, CustomTestCase
):
    """--rl-on-policy-target fsdp keeps inference deterministic.

    The on-policy target selects the numerical contract the rollout must match
    (an FSDP trainer); enabling it must not weaken batch invariance, and the
    server must actually accept and register the flag.
    """

    extra_server_args = ("--rl-on-policy-target", "fsdp")

    def test_01_bs1_matches_bsn_bitwise(self):
        _assert_bitwise_determinism(self, self.trials, "rl_on_policy_target=fsdp")

    def test_02_server_registered_the_target(self):
        try:
            info = get(self.base_url, "/server_info").json()
        except RuntimeError:
            # Older builds expose the same payload under /get_server_info.
            info = get(self.base_url, "/get_server_info").json()
        args = info.get("server_args", info)
        self.assertEqual(
            args.get("rl_on_policy_target"),
            "fsdp",
            "server did not register --rl-on-policy-target fsdp",
        )


if __name__ == "__main__":
    unittest.main()
