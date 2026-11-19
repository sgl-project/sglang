"""DSA DCP resolves unset backends to the platform's DCP path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import (
    _deepseek_family_overrides,
    _dsa_dcp_overrides,
)
from sglang.srt.arg_groups.overrides import _check_dsa_backend_constraints
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PLATFORM = "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform"


def _cfg(**kw):
    base = dict(
        dcp_size=4,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        disaggregation_mode="null",
        speculative_eagle_topk=1,
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        dcp_replicate_q_proj=None,
        dcp_comm_backend=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _backends(overrides):
    return overrides.get("dsa_prefill_backend"), overrides.get("dsa_decode_backend")


_GLM_CONFIG = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], index_topk=2048)
_SPLIT_INDEX_K_FIELDS = ("dsa_prefill_backend", "dsa_decode_backend", "kv_cache_dtype")


def _glm_overrides(*, is_sm90=True, **fields):
    args = dict(
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        enable_prefill_cp=False,
        kv_cache_dtype="auto",
        dsa_topk_backend="sgl-kernel",
        speculative_algorithm=None,
        speculative_dsa_topk_backend="sgl-kernel",
    )
    args.update(fields)
    with override_platform(is_npu=False, is_xpu=False, is_hip=False, is_sm90=is_sm90):
        overrides = _deepseek_family_overrides(_cfg(**args), _GLM_CONFIG)
    return {k: v for k, v in overrides.items() if k in _SPLIT_INDEX_K_FIELDS}


class TestDsaDcpBackendDefaults(CustomTestCase):
    def test_hip_defaults_to_triton(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=True, is_npu=False)):
            self.assertEqual(
                _backends(_dsa_dcp_overrides(_cfg())), ("triton", "triton")
            )

    def test_cuda_defaults_to_trtllm(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=False, is_npu=False)):
            self.assertEqual(
                _backends(_dsa_dcp_overrides(_cfg())), ("trtllm", "trtllm")
            )

    def test_explicit_backend_is_kept(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=True, is_npu=False)):
            overrides = _dsa_dcp_overrides(_cfg(dsa_decode_backend="triton"))
        self.assertEqual(_backends(overrides), ("triton", None))

    def test_no_dcp_sets_nothing(self):
        self.assertEqual(_dsa_dcp_overrides(_cfg(dcp_size=1)), {})

    def test_npu_pd_dcp_uses_native_backend(self):
        platform = SimpleNamespace(is_hip=False, is_npu=True)
        with patch(_PLATFORM, return_value=platform):
            overrides = _dsa_dcp_overrides(
                _cfg(
                    disaggregation_mode="decode",
                    disaggregation_transfer_backend="ascend",
                )
            )
        self.assertNotIn("dsa_prefill_backend", overrides)
        self.assertNotIn("dsa_decode_backend", overrides)


class TestGlmSplitIndexKDcpStartup(CustomTestCase):
    def test_unset_fields_get_the_supported_values_only_under_dcp_on_sm90(self):
        self.assertEqual(
            _glm_overrides(),
            {
                "dsa_prefill_backend": "flashmla_sparse_q8",
                "dsa_decode_backend": "flashmla_kv",
                "kv_cache_dtype": "fp8_e4m3",
            },
        )
        self.assertEqual(
            _glm_overrides(
                is_sm90=False,
                kv_cache_dtype="bfloat16",
            ),
            {"dsa_prefill_backend": "trtllm", "dsa_decode_backend": "trtllm"},
        )
        self.assertEqual(
            _glm_overrides(
                dcp_size=1,
                dsa_decode_backend="fa3",
            ),
            {},
        )

    def test_explicit_supported_prefill_backend_is_kept(self):
        self.assertEqual(
            _glm_overrides(dsa_prefill_backend="flashmla_kv"),
            {
                "dsa_prefill_backend": "flashmla_kv",
                "dsa_decode_backend": "flashmla_kv",
                "kv_cache_dtype": "fp8_e4m3",
            },
        )

    def test_unsupported_explicit_values_are_rejected(self):
        for field, value in (
            ("dsa_prefill_backend", "flashmla_sparse"),
            ("dsa_decode_backend", "trtllm"),
            ("kv_cache_dtype", "bfloat16"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field.replace("_", "-")):
                    _glm_overrides(**{field: value})

    def test_settings_the_sharded_top_k_cannot_serve_are_rejected(self):
        """The ranks merge local top-k through the fused sgl-kernel v2 path and
        keep rank-local slots, so another backend or a rank-0 broadcast would
        attend the wrong KV rows."""
        for fields in (
            {"dsa_topk_backend": "flashinfer"},
            {"speculative_algorithm": "NEXTN", "speculative_dsa_topk_backend": "torch"},
        ):
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, "fused sgl-kernel v2"):
                    _glm_overrides(**fields)
        for algorithm in (None, "ngram"):
            _glm_overrides(
                speculative_algorithm=algorithm, speculative_dsa_topk_backend="torch"
            )
        with envs.SGLANG_DSA_FUSE_TOPK.override(False):
            with self.assertRaisesRegex(ValueError, "SGLANG_DSA_FUSE_TOPK"):
                _glm_overrides()
        with envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
            with self.assertRaisesRegex(ValueError, "SGLANG_DSA_TOPK_BROADCAST"):
                _glm_overrides()


class TestDsaDcpBackendConstraints(CustomTestCase):
    def test_hip_dcp_triton_allowed(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3", "triton", "triton", hip=True, dcp_size=4
        )

    def test_hip_dcp_rejects_other_backends(self):
        for prefill, decode in (
            ("trtllm", "trtllm"),
            ("tilelang", "tilelang"),
            ("triton", "tilelang"),
            ("aiter", "triton"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaisesRegex(ValueError, "no DCP path on ROCm"):
                    _check_dsa_backend_constraints(
                        "fp8_e4m3", prefill, decode, hip=True, dcp_size=4
                    )

    def test_hip_without_dcp_keeps_other_backends(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3", "tilelang", "tilelang", hip=True, dcp_size=1
        )


if __name__ == "__main__":
    unittest.main()
