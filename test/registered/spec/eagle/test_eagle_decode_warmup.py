"""--enable-decode-warmup: an engine that ran the warm-up rounds before ready
serves the same tokens and acceptance as one that did not (greedy, so the
comparison does not depend on cross-process sampling noise)."""

import unittest
from contextlib import ExitStack

import pytest
import torch

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    CustomTestCase,
)

register_cuda_ci(est_time=420, stage="nightly", runner_config="1-gpu-small")

pytestmark = pytest.mark.gpu

_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea.",
    "List three prime numbers greater than 100:",
    "def fibonacci(n):",
]
_SAMPLING = {"temperature": 0.0, "max_new_tokens": 48}


def _serve(*, warmup: bool, extra=()):
    engine = sgl.Engine(
        model_path=DEFAULT_TARGET_MODEL_EAGLE3,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE3,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        mem_fraction_static=0.7,
        max_running_requests=8,
        dtype="bfloat16",
        random_seed=42,
        enable_decode_warmup=warmup,
        decode_warmup_batch_sizes=[4, 2, 1],
        decode_warmup_prompt_len=256,
        **dict(extra),
    )
    try:
        outputs = engine.generate(_PROMPTS, _SAMPLING)
    finally:
        engine.shutdown()
    return [(o["output_ids"], o["meta_info"].get("spec_verify_ct")) for o in outputs]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestEagleDecodeWarmup(CustomTestCase):
    def test_warmup_leaves_served_tokens_and_acceptance_unchanged(self):
        off = _serve(warmup=False)
        on = _serve(warmup=True)
        self.assertEqual(on, off)

    def test_warmup_with_post_verify_late_join(self):
        """The rounds replay post-verify graphs; with the late-joined state commit
        the pending commit must be joined before the dummies' rows are freed."""
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True)
            )
            stack.enter_context(envs.SGLANG_ENABLE_POST_VERIFY_LATE_JOIN.override(True))
            off = _serve(warmup=False)
            on = _serve(warmup=True)
        self.assertEqual(on, off)

    def test_warmup_with_preferred_sampling_params_starts(self):
        """The dummies take --preferred-sampling-params; a sampled set must not
        leave RNG or pool state behind that changes greedy serving."""
        off = _serve(warmup=False)
        on = _serve(
            warmup=True,
            extra={"preferred_sampling_params": '{"temperature": 0.7, "top_p": 0.9}'},
        )
        self.assertEqual(on, off)


if __name__ == "__main__":
    unittest.main()
