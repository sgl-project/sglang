# SPDX-License-Identifier: Apache-2.0
"""The generic KV4 gate must preserve the validated SM120 GLM DSA route."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import kv_cache_hook
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlmNvfp4Kv4Gate(unittest.TestCase):
    def check_gate(self, *, platform=None, model=None, uses_mla=True, **options):
        cfg = SimpleNamespace(
            **{
                "kv_cache_dtype": "nvfp4",
                "enable_unified_memory": False,
                "speculative_algorithm": None,
                "attention_backend": "dsa",
                "dsa_prefill_backend": "flashinfer_sparse_mla",
                "dsa_decode_backend": "flashinfer_sparse_mla",
                **options,
            }
        )
        gpu = SimpleNamespace(
            **{"is_cuda": True, "is_sm120": True, "is_sm100": False, **(platform or {})}
        )
        hf = SimpleNamespace(
            **{
                "architectures": ["GlmMoeDsaForCausalLM"],
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "learnable_sink": False,
                **(model or {}),
            }
        )
        with (
            patch.object(kv_cache_hook, "resolving_view", return_value=cfg),
            patch.object(kv_cache_hook, "resolved_view", return_value=cfg),
            patch.object(kv_cache_hook, "use_mla_backend", return_value=uses_mla),
            patch.object(
                kv_cache_hook,
                "attention_backends_of",
                return_value=(cfg.attention_backend, cfg.attention_backend),
            ),
            patch.object(kv_cache_hook, "get_platform", return_value=gpu),
            patch.object(
                kv_cache_hook,
                "model_config_of",
                return_value=SimpleNamespace(hf_config=hf),
            ),
        ):
            kv_cache_hook.handle_kv4_compatibility(cfg)

    def test_glm_nvfp4_sm120_fi_sparse_accepted(self):
        self.check_gate()

    def test_other_models_and_geometry_remain_rejected(self):
        for model in (
            {"architectures": ["DeepseekV32ForCausalLM"]},
            {"architectures": ["Glm4MoeForCausalLM"]},
            {"kv_lora_rank": 256},
            {"qk_rope_head_dim": 128},
            {"learnable_sink": True},
        ):
            with self.subTest(model=model), self.assertRaises(AssertionError):
                self.check_gate(model=model)

    def test_both_dsa_backends_must_be_fi_sparse(self):
        for field in ("dsa_prefill_backend", "dsa_decode_backend"):
            for backend in (None, "flashmla_sparse", "trtllm", "tilelang"):
                with (
                    self.subTest(field=field, backend=backend),
                    self.assertRaises(AssertionError),
                ):
                    self.check_gate(**{field: backend})

    def test_sm100_dsa_is_not_enabled(self):
        with self.assertRaises(AssertionError):
            self.check_gate(platform={"is_sm120": False, "is_sm100": True})

    def test_unsupported_platforms_remain_rejected(self):
        for platform in ({"is_cuda": False}, {"is_sm120": False, "is_sm100": False}):
            with self.subTest(platform=platform), self.assertRaises(RuntimeError):
                self.check_gate(platform=platform)

    def test_mxfp4_dsa_and_mha_dsa_remain_rejected(self):
        with self.assertRaises(AssertionError):
            self.check_gate(kv_cache_dtype="fp4_mx_block16")
        with self.assertRaises(AssertionError):
            self.check_gate(uses_mla=False)

    def test_existing_mla_backends_unchanged(self):
        for backend in ("flashinfer", "trtllm_mla"):
            with self.subTest(backend=backend):
                self.check_gate(attention_backend=backend)

    def test_non_kv4_still_bypasses_gate(self):
        for dtype in ("fp8_e4m3", "bfloat16"):
            with self.subTest(dtype=dtype):
                self.check_gate(kv_cache_dtype=dtype, platform={"is_cuda": False})


if __name__ == "__main__":
    unittest.main()
