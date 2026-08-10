"""Unit tests for LoRA + speculative decoding compatibility validation.

The matrix lives in ServerArgs._check_lora_speculative_compatibility, which
runs at engine-launch validation, after __post_init__ resolved speculative
aliases — so it matches resolved names only ("NEXTN" has collapsed to "EAGLE",
Gemma4 promotions to "FROZEN_KV_MTP").
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Settings that make LoRA + EAGLE-family speculation unsafe, each with the
# term its rejection must name.
UNSUPPORTED_SETTINGS = [
    ({"speculative_adaptive": True}, "speculative-adaptive"),
    ({"moe_runner_backend": "experimental_sgl_trtllm"}, "experimental_sgl_trtllm"),
    (
        {"speculative_moe_runner_backend": "experimental_sgl_trtllm"},
        "experimental_sgl_trtllm",
    ),
]


def _args(**kwargs) -> ServerArgs:
    # model_path="dummy" early-returns __post_init__, so fields land raw and
    # _check_lora_speculative_compatibility can be exercised directly.
    kwargs.setdefault("lora_backend", "triton")
    return ServerArgs(model_path="dummy", **kwargs)


class TestLoRASpeculativeCompatibility(CustomTestCase):
    def test_ngram_and_no_spec_pass_with_any_backend(self):
        """NGRAM support predates the EAGLE path and has no backend
        restriction; a diff that tightened the early return would break
        existing NGRAM+LoRA deployments."""
        for algo in [None, "NGRAM"]:
            for backend in ["csgmv", "triton"]:
                _args(
                    speculative_algorithm=algo, lora_backend=backend
                )._check_lora_speculative_compatibility()

    def test_unsupported_settings_are_rejected(self):
        for overrides, expected in UNSUPPORTED_SETTINGS:
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, expected):
                    _args(
                        speculative_algorithm="EAGLE", **overrides
                    )._check_lora_speculative_compatibility()

    def test_unsupported_algorithms_are_rejected(self):
        """FROZEN_KV_MTP matters here: EAGLE/NEXTN with a Gemma4 assistant
        draft is silently promoted to it during resolution, so the error has
        to name the promotion or the user sees an algorithm they never typed."""
        for algo in ["STANDALONE", "DFLASH", "DSPARK"]:
            with self.subTest(algo=algo):
                with self.assertRaises(ValueError):
                    _args(
                        speculative_algorithm=algo
                    )._check_lora_speculative_compatibility()
        with self.assertRaisesRegex(ValueError, "promoted"):
            _args(
                speculative_algorithm="FROZEN_KV_MTP"
            )._check_lora_speculative_compatibility()

    def test_overlap_plan_stream_env_is_rejected(self):
        """LoRA prep runs inside ForwardBatch.init_new, which EAGLE calls
        under the plan stream; that stream has no ordering against in-flight
        forwards, so LoRA's buffer writes could race a reader."""
        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(True):
            with self.assertRaisesRegex(ValueError, "PLAN_STREAM"):
                _args(
                    speculative_algorithm="EAGLE"
                )._check_lora_speculative_compatibility()

    def test_multi_adapter_eagle_passes_full_validation(self):
        """Goes through the public check_lora_server_args, so it also guards
        the wiring: the old NGRAM-only gate raised here for EAGLE. A
        regression restoring it, adding a single-adapter cap, or restricting
        the kernel backend turns this red."""
        for algo in ["EAGLE", "EAGLE3"]:
            for backend in ["triton", "csgmv"]:
                with self.subTest(algo=algo, backend=backend):
                    args = _args(
                        enable_lora=True,
                        speculative_algorithm=algo,
                        lora_backend=backend,
                        lora_paths=["name-a=path-a", "name-b=path-b"],
                        max_loras_per_batch=2,
                    )
                    args.check_lora_server_args()
                    self.assertEqual(len(args.lora_paths), 2)


if __name__ == "__main__":
    unittest.main()
