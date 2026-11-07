"""DSA DCP resolves unset backends to the platform's DCP path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.srt.arg_groups.overrides import _check_dsa_backend_constraints
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


class TestDsaDcpBackendDefaults(CustomTestCase):
    def test_hip_defaults_to_triton(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=True)):
            self.assertEqual(
                _backends(_dsa_dcp_overrides(_cfg())), ("triton", "triton")
            )

    def test_cuda_defaults_to_trtllm(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=False)):
            self.assertEqual(
                _backends(_dsa_dcp_overrides(_cfg())), ("trtllm", "trtllm")
            )

    def test_explicit_backend_is_kept(self):
        with patch(_PLATFORM, return_value=SimpleNamespace(is_hip=True)):
            overrides = _dsa_dcp_overrides(_cfg(dsa_decode_backend="triton"))
        self.assertEqual(_backends(overrides), ("triton", None))

    def test_no_dcp_sets_nothing(self):
        self.assertEqual(_dsa_dcp_overrides(_cfg(dcp_size=1)), {})


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
