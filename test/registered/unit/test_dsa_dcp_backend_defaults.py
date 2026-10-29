"""DSA DCP resolves unset backends to the platform's DCP path, and the
resolved backends are checked against what the device can actually run."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.srt.arg_groups.overrides import _check_dsa_backend_constraints
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PLATFORM = "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform"
_CAPABILITY = "torch.cuda.get_device_capability"


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


class TestDsaTrtllmDeviceConstraint(CustomTestCase):
    """trtllm dispatches flashinfer's TRT-LLM-GEN MLA kernels. The gate accepts
    exactly the arches that pinned flashinfer builds them for, so a device
    flashinfer would refuse fails at launch instead of mid-forward. SGLang
    auto-selects trtllm for FP8 KV on every SM >= 10."""

    _ACCEPTED = ((10, 0), (10, 3), (10, 7))
    # flashinfer/mla/_core.py: the message that dispatch answers with.
    _REFUSAL = "requires SM100/SM103/SM107"

    def _check(self, capability, *, prefill="trtllm", decode="trtllm", kv="fp8_e4m3"):
        with patch(_CAPABILITY, return_value=capability):
            _check_dsa_backend_constraints(kv, prefill, decode, hip=False)

    def test_flashinfer_supported_arches_allowed(self):
        for capability in self._ACCEPTED:
            with self.subTest(capability=capability):
                self._check(capability)

    def test_consumer_and_workstation_blackwell_rejected(self):
        for capability in ((12, 0), (12, 1), (11, 0)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(ValueError, self._REFUSAL):
                    self._check(capability)

    def test_sm10_minors_outside_the_supported_set_rejected(self):
        # flashinfer gates on an exact arch triple, so `major == 10` alone would
        # accept devices it then refuses; keep the check tied to that triple.
        for capability in ((10, 1), (10, 2), (10, 4), (10, 6)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(ValueError, self._REFUSAL):
                    self._check(capability)

    def test_hopper_rejected(self):
        with self.assertRaisesRegex(ValueError, self._REFUSAL):
            self._check((9, 0))

    def test_one_sided_trtllm_rejected(self):
        with self.assertRaisesRegex(ValueError, self._REFUSAL):
            self._check((12, 0), prefill="flashmla_sparse", decode="trtllm")

    def test_bfloat16_kv_is_checked_too(self):
        with self.assertRaisesRegex(ValueError, self._REFUSAL):
            self._check((12, 0), kv="bfloat16")

    def test_other_backends_are_not_checked(self):
        for prefill, decode in (
            ("flashmla_sparse", "flashmla_kv"),
            ("tilelang", "tilelang"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                self._check((12, 0), prefill=prefill, decode=decode)


if __name__ == "__main__":
    unittest.main()
