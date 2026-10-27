import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _dsa_dcp_overrides
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def config(**kwargs):
    values = dict(
        dcp_size=4,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        disaggregation_mode="decode",
        speculative_eagle_topk=1,
        dsa_prefill_backend="triton",
        dsa_decode_backend="triton",
        dcp_replicate_q_proj=False,
    )
    return SimpleNamespace(**(values | kwargs))


class TestDsaPdDcpConfig(unittest.TestCase):
    def test_only_rocm_decode_is_allowed_for_pd_dcp(self):
        for hip in (False, True):
            for mode in ("null", "prefill", "decode"):
                with (
                    self.subTest(hip=hip, mode=mode),
                    patch(
                        "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform",
                        return_value=SimpleNamespace(is_hip=hip),
                    ),
                ):
                    cfg = config(disaggregation_mode=mode)
                    if mode == "null" or (hip and mode == "decode"):
                        self.assertEqual(_dsa_dcp_overrides(cfg), {})
                    else:
                        with self.assertRaisesRegex(ValueError, "DCP1 prefill"):
                            _dsa_dcp_overrides(cfg)

    def test_dcp1_prefill_is_unchanged(self):
        self.assertEqual(
            _dsa_dcp_overrides(config(dcp_size=1, disaggregation_mode="prefill")), {}
        )

    def test_other_dcp_constraints_are_preserved(self):
        with patch(
            "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform",
            return_value=SimpleNamespace(is_hip=True),
        ):
            for kwargs in (
                {"enable_hisparse": True},
                {"enable_hierarchical_cache": True},
                {"speculative_eagle_topk": 2},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    _dsa_dcp_overrides(config(**kwargs))


if __name__ == "__main__":
    unittest.main()
