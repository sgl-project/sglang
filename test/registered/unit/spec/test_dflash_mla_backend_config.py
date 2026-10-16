import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dflash_mla_backend import (
    dflash_mla_prefill_causal,
    resolve_dflash_mla_attention_backend,
    validate_dflash_mla_backend_config,
    validate_dflash_mla_runner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashMLABackendConfig(unittest.TestCase):
    def test_prefill_causality_preserves_legacy_default(self):
        for layer_types in (None, ["full_attention"]):
            for is_causal in (None, False, True):
                with self.subTest(layer_types=layer_types, is_causal=is_causal):
                    self.assertEqual(
                        dflash_mla_prefill_causal(
                            layer_types=layer_types, is_causal=is_causal
                        ),
                        layer_types is not None and is_causal is True,
                    )

    def _validate(self, backend="tokenspeed_mla", **overrides):
        config = dict(
            backend=backend,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_local_heads=32,
            model_dtype="torch.bfloat16",
            kv_cache_dtype="torch.float8_e4m3fn",
            kv_cache_dtype_str="fp8_e4m3",
            page_size=64,
            device="cuda:0",
            is_hip=False,
            is_blackwell=True,
            device_capability=(10, 3),
            has_sliding_attention=False,
        )
        config.update(overrides)
        validate_dflash_mla_backend_config(**config)

    def test_default_and_explicit_backend(self):
        self.assertEqual(resolve_dflash_mla_attention_backend(None), "triton")
        for backend in ("triton", "flashinfer", "tokenspeed_mla"):
            self.assertEqual(resolve_dflash_mla_attention_backend(backend), backend)
        for backend in ("fa3", "trtllm_mla", "flashinfer_mla", "unknown"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(
                    ValueError, "speculative-draft-attention-backend"
                ),
            ):
                resolve_dflash_mla_attention_backend(backend)

    def test_target_backend_is_unchanged(self):
        # The new validation must not limit any target-model MLA configuration.
        runner = SimpleNamespace(model_config=SimpleNamespace())
        validate_dflash_mla_runner(runner, "trtllm_mla")

    def test_tokenspeed_runtime_constraints(self):
        self._validate()
        cases = (
            ({"kv_cache_dtype": "torch.bfloat16"}, "draft-kv-cache-dtype"),
            ({"page_size": 1}, "page-size"),
            ({"is_blackwell": False}, "Blackwell"),
            ({"device_capability": (12, 0)}, "SM100/SM103"),
            ({"model_dtype": "torch.float16"}, "bfloat16 draft model"),
            ({"num_local_heads": 129}, "local heads"),
            ({"kv_lora_rank": 256}, "kv_lora_rank"),
            ({"qk_rope_head_dim": 32}, "qk_rope_head_dim"),
            ({"has_sliding_attention": True}, "sliding attention"),
            ({"is_hip": True}, "NVIDIA CUDA"),
            ({"dcp_size": 2}, "context parallelism"),
            ({"tp_size": 2, "attn_tp_size": 1}, "DP attention"),
            ({"device": "cpu"}, "CUDA or HIP"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._validate(**overrides)

    def test_flashinfer_constraints_are_distinct(self):
        self._validate(
            "flashinfer",
            page_size=1,
            is_blackwell=False,
            kv_cache_dtype="torch.bfloat16",
        )
        with self.assertRaisesRegex(ValueError, "sliding attention"):
            self._validate("flashinfer", has_sliding_attention=True)
        with self.assertRaisesRegex(ValueError, "kv_lora_rank"):
            self._validate("flashinfer", kv_lora_rank=128)
        for overrides, message in (
            ({"kv_cache_dtype": "torch.float8_e4m3fn"}, "draft KV dtype to match"),
            ({"model_dtype": "torch.float32"}, "model dtype"),
            ({"device_capability": (7, 5)}, "SM80"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._validate("flashinfer", **overrides)
        self._validate(
            "flashinfer", model_dtype="torch.float16", kv_cache_dtype="torch.float16"
        )

    def test_triton_supports_other_shapes_and_sliding_attention(self):
        self._validate(
            "triton",
            kv_lora_rank=256,
            qk_rope_head_dim=32,
            page_size=1,
            is_hip=True,
            is_blackwell=False,
            kv_cache_dtype="torch.bfloat16",
            has_sliding_attention=True,
        )
        with self.assertRaisesRegex(ValueError, "draft KV dtype"):
            self._validate("triton", kv_cache_dtype="torch.float4_e2m1fn_x2")
        for overrides, message in (
            ({"kv_cache_dtype_str": "mxfp8"}, "block-scaled"),
            ({"kv_lora_rank": 8}, "kv_lora_rank"),
            ({"kv_lora_rank": 1024}, "kv_lora_rank"),
            ({"qk_rope_head_dim": 3}, "qk_rope_head_dim"),
            ({"qk_rope_head_dim": 128}, "qk_rope_head_dim"),
            ({"num_local_heads": 0}, "local query head"),
            ({"page_size": 0}, "page_size"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._validate("triton", **overrides)


if __name__ == "__main__":
    unittest.main()
