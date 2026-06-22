"""
End-to-end test for the int8 mamba checkpoint pool on a real GDN-hybrid model.

Launches Qwen3-Next-80B-A3B (a gated-delta-net / linear-attention hybrid) with
``--enable-int8-mamba-checkpoint`` and checks two things while the int8 dequant
path is exercised:

  * KLDivergenceMixin — on a prefix/decode cache HIT the generated logprobs are
    compared (KL) against a full recompute. This is the *sensitive* precision
    guard: it directly bounds how far the int8-reused state moves the output
    distribution from the exact-recompute distribution.
  * test_gsm8k — end-to-end task accuracy holds.

NOTE: the int8 checkpoint is only engaged when a cached prefix is reused FROM the
int8 pool, which requires ``--mamba-scheduler-strategy extra_buffer`` — the default
``no_buffer`` only snapshots the recurrent state at the full-sequence leaf, so a
fixed-prefix / divergent-question workload reuses ~0 mamba state and the int8 path
would never fire.

Usage:
    python3 -m unittest test_int8_mamba_checkpoint_e2e
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=192, stage="base-c", runner_config="4-gpu-h100")


class TestInt8MambaCheckpointE2E(KLDivergenceMixin, DefaultServerBase):
    """int8 mamba checkpoint pool on Qwen3-Next-80B-A3B (GDN-hybrid)."""

    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST

    # Cache-hit KL: int8 is a lossy codec, so its cache-hit divergence is
    # inherently larger than the bf16/fp8 reuse the other KL tests bound (~0.005),
    # and it grows with context length (a longer prefix = a fuller state = larger
    # absolute rounding error in the logits). Measured on a Qwen3.5-35B stand-in
    # over LongBench-V2 prompts: prefill ~0.044, decode ~0.024. Thresholds are set
    # to ~2x that, to cover model differences (80B) and the reuse path's
    # run-to-run noise while still catching a real int8 regression.
    kl_div_thres = 0.06
    kl_div_thres_prefill = 0.10
    kl_div_thres_decode = 0.06
    kl_div_max_samples = 16
    kl_div_prefill_max_new_tokens = 512
    kl_div_decode_max_new_tokens = 512

    gsm8k_threshold = 0.90
    num_gsm8k_questions = 100
    num_shots = 8
    parallel = 8

    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--enable-int8-mamba-checkpoint",
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]

    def test_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=self.num_shots,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=512,
            parallel=self.parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)


if __name__ == "__main__":
    unittest.main()
