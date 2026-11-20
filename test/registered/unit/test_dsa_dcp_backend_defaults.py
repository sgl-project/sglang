"""DSA DCP resolves unset backends to the platform's DCP path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.srt.arg_groups.overrides import (
    _check_dsa_backend_constraints,
    _kimi_k31_dsa_backends,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PLATFORM = "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform"
_K31_PLATFORM = "sglang.srt.arg_groups.overrides.get_platform"


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


class TestKimiK31DsaBackendDefault(CustomTestCase):
    def _resolve(self, device_sm, **kw):
        platform = SimpleNamespace(
            is_hip=False, is_sm100=device_sm // 10 == 10, device_sm=device_sm
        )
        view = SimpleNamespace(
            enable_hisparse=False, dsa_prefill_backend=None, dsa_decode_backend=None
        )
        view.__dict__.update(kw)
        with patch(_K31_PLATFORM, return_value=platform):
            return _backends(_kimi_k31_dsa_backends(view))

    def test_sm100_family_defaults_to_prims_ts(self):
        for device_sm in (100, 103, 107):
            with self.subTest(device_sm=device_sm):
                self.assertEqual(self._resolve(device_sm), ("prims_ts", "prims_ts"))

    def test_other_cuda_keeps_flashmla_kv(self):
        self.assertEqual(self._resolve(90), ("flashmla_kv", "flashmla_kv"))

    def test_hisparse_keeps_flashmla_kv(self):
        """HiSparse admits only flashmla_kv; a prims_ts default would reject it."""
        self.assertEqual(
            self._resolve(103, enable_hisparse=True), ("flashmla_kv", "flashmla_kv")
        )


class TestDsaDcpBackendConstraints(CustomTestCase):
    def test_hip_dcp_triton_allowed(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3",
            "triton",
            "triton",
            hip=True,
            dcp_size=4,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
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
                        "fp8_e4m3",
                        prefill,
                        decode,
                        hip=True,
                        dcp_size=4,
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                    )

    def test_hip_without_dcp_keeps_other_backends(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3",
            "tilelang",
            "tilelang",
            hip=True,
            dcp_size=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )


if __name__ == "__main__":
    unittest.main()
