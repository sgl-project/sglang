"""HiCache with DSA DCP is accepted only for a ROCm DSA NextN MTP draft."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hicache_hook import resolve_hicache_dcp_compatibility
from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.srt.configs.model_config import AttentionArch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_OVERRIDE_PLATFORM = "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform"
_DRAFT_IS_HIP = "sglang.srt.configs.model_config.is_hip"

_DSA = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], index_topk=2048)
_MLA_NOT_DSA = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])


def _override_cfg(**kw):
    base = dict(
        dcp_size=4,
        enable_hisparse=False,
        enable_hierarchical_cache=True,
        disaggregation_mode="null",
        speculative_eagle_topk=1,
        dsa_prefill_backend="triton",
        dsa_decode_backend="triton",
        dcp_replicate_q_proj=None,
        dcp_comm_backend=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _server_args(speculative_algorithm="EAGLE", hf_config=_DSA, **kw):
    base = dict(
        dcp_size=4,
        enable_hierarchical_cache=True,
        hicache_storage_backend=None,
        speculative_algorithm=speculative_algorithm,
        enable_lmcache=False,
        enable_hisparse=False,
        _model_config=SimpleNamespace(
            hf_config=hf_config, attention_arch=AttentionArch.MLA
        ),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestDsaDcpHiCacheOverride(CustomTestCase):
    def test_non_hip_rejects_hicache(self):
        with patch(_OVERRIDE_PLATFORM, return_value=SimpleNamespace(is_hip=False)):
            with self.assertRaisesRegex(ValueError, "HiCache only on ROCm"):
                _dsa_dcp_overrides(_override_cfg())

    def test_hip_accepts_hicache(self):
        with patch(_OVERRIDE_PLATFORM, return_value=SimpleNamespace(is_hip=True)):
            _dsa_dcp_overrides(_override_cfg())

    def test_hisparse_is_still_rejected_on_hip(self):
        with patch(_OVERRIDE_PLATFORM, return_value=SimpleNamespace(is_hip=True)):
            with self.assertRaisesRegex(ValueError, "HiSparse"):
                _dsa_dcp_overrides(_override_cfg(enable_hisparse=True))


class TestHiCacheDcpDraftGate(CustomTestCase):
    def test_hip_dsa_mtp_is_accepted(self):
        for algorithm in ("EAGLE", "NEXTN"):
            with self.subTest(algorithm=algorithm):
                with patch(_DRAFT_IS_HIP, return_value=True):
                    resolve_hicache_dcp_compatibility(_server_args(algorithm))

    def test_unsupported_drafts_are_rejected(self):
        cases = {
            "eagle3": dict(hip=True, speculative_algorithm="EAGLE3"),
            "eagle off rocm": dict(hip=False),
            "non-dsa model": dict(hip=True, hf_config=_MLA_NOT_DSA),
        }
        for name, case in cases.items():
            hip = case.pop("hip")
            with self.subTest(name):
                with patch(_DRAFT_IS_HIP, return_value=hip):
                    with self.assertRaisesRegex(
                        NotImplementedError, "only supports DSPARK"
                    ):
                        resolve_hicache_dcp_compatibility(_server_args(**case))

    def test_dspark_and_no_draft_skip_the_dsa_check(self):
        for algorithm in (None, "DSPARK"):
            with self.subTest(algorithm=algorithm):
                with patch(_DRAFT_IS_HIP, return_value=False):
                    resolve_hicache_dcp_compatibility(
                        _server_args(algorithm, hf_config=_MLA_NOT_DSA)
                    )


if __name__ == "__main__":
    unittest.main()
