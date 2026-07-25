"""Regression test for the SWA admission livelock (PrefillAdder over-charging a
full sliding window of decode headroom for cached-prefix resumes).

With a hybrid-SWA model, an SWA pool sized near two sliding windows, LPM
scheduling, and requests that resume on a radix-cached prefix >= one window, the
buggy scheduler spins forever re-rejecting a feasible request (idle GPU, 100%
scheduler CPU). The fix caps decode headroom at what the request actually adds
to its window, so admission succeeds and phase 2 completes in seconds.

Uses the small ``thinkingmachines/Inkling-NVFP4`` ``test`` revision (8-layer,
sliding_window_size=512) on a single GPU.
"""

import os
import random
import threading
import unittest

import sglang
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-large")

_MODEL_PATH = os.environ.get(
    "INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling-NVFP4"
)
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")

WINDOW = 512  # Inkling test-branch sliding_window_size
NEW_TOKENS = 256
# A healthy scheduler finishes phase 2 in seconds; a livelock spins until the
# process is killed. The bound is generous so it never flakes on a healthy run.
PHASE2_TIMEOUT_S = 240


class TestSWAAdmissionLivelock(CustomTestCase):
    def test_cached_prefix_resume_does_not_livelock(self):
        engine_kwargs = dict(
            model_path=_MODEL_PATH,
            tp_size=1,
            trust_remote_code=True,
            quantization="modelopt_fp4",
            attention_backend="fa4",
            fp4_gemm_runner_backend="marlin",
            moe_runner_backend="marlin",
            mamba_radix_cache_strategy="extra_buffer",
            # SWA pool ~= 2 sliding windows: the livelock boundary.
            max_total_tokens=1280,
            swa_full_tokens_ratio=0.81,
            context_length=1408,
            chunked_prefill_size=512,
            max_running_requests=16,
            schedule_conservativeness=0.05,
            schedule_policy="lpm",  # parks the cached-prefix resume at the queue head
            mem_fraction_static=0.5,
            disable_cuda_graph=True,
            log_level="warning",
        )
        if _MODEL_REVISION:
            engine_kwargs["revision"] = _MODEL_REVISION

        engine = sglang.Engine(**engine_kwargs)
        try:
            rng = random.Random(0)
            len_rng = random.Random(1)
            cut_rng = random.Random(2)
            # Prompts >= one full sliding window so every phase-2 resume matches
            # (and admission-locks) >= one window of cache; capped so prompt +
            # NEW_TOKENS fits the SWA pool for a solo request.
            prompt_lens = sorted(
                len_rng.randint(WINDOW, WINDOW + 128) for _ in range(4)
            )
            base = [rng.randrange(199_998) for _ in range(max(prompt_lens))]
            prompts = [base[:n] for n in prompt_lens]

            def gen(input_ids, max_news):
                return engine.generate(
                    input_ids=input_ids,
                    sampling_params=[
                        {"temperature": 0.0, "max_new_tokens": n, "ignore_eos": True}
                        for n in max_news
                    ],
                )

            # Phase 1: baselines populate the radix cache.
            outs = gen(prompts, [NEW_TOKENS] * len(prompts))
            baselines = [o["output_ids"] for o in outs]

            # Phase 2: resumes replay the prompt + a random cut of the baseline
            # output, decoding out to the baseline length. Uncut/short-cut ones
            # resume on >= one window of cached prefix (the livelock trigger).
            reqs = [
                (p + b[:cut], NEW_TOKENS - cut)
                for p, b in zip(prompts, baselines)
                for cut in (cut_rng.randint(0, NEW_TOKENS - 64) for _ in range(4))
            ]
            random.Random(0).shuffle(reqs)

            done = threading.Event()
            errors = []

            def run_phase2():
                try:
                    gen([r[0] for r in reqs], [r[1] for r in reqs])
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                finally:
                    done.set()

            threading.Thread(target=run_phase2, daemon=True).start()
            completed = done.wait(PHASE2_TIMEOUT_S)
            self.assertTrue(
                completed,
                f"SWA admission livelock: phase 2 did not complete in "
                f"{PHASE2_TIMEOUT_S}s (scheduler spinning, GPU idle).",
            )
            self.assertFalse(errors, f"phase 2 raised: {errors}")
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
