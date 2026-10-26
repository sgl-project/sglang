import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HOOK = "sglang.srt.arg_groups.hisparse_hook"


def _cfg():
    return SimpleNamespace(
        enable_hisparse=True,
        disable_radix_cache=True,
        enable_mixed_chunk=False,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_adaptive=False,
    )


@patch("sglang.srt.configs.model_config.is_minimax_sparse", return_value=False)
@patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=False)
@patch("sglang.srt.configs.model_config.is_deepseek_v4", return_value=True)
class TestHiSparseHook(unittest.TestCase):
    def _validate(self, disaggregation_mode):
        server_args = SimpleNamespace(disaggregation_mode=disaggregation_mode)
        with (
            patch(f"{_HOOK}.resolving_view", return_value=_cfg()),
            patch(
                f"{_HOOK}.model_config_of",
                return_value=SimpleNamespace(hf_config=object()),
            ),
            patch(f"{_HOOK}.get_platform", return_value=SimpleNamespace(is_hip=True)),
        ):
            validate_hisparse(server_args)

    def test_dsv4_pd_prefill_is_rejected(self, *_mocks):
        with self.assertRaisesRegex(ValueError, "decode-only"):
            self._validate("prefill")

    def test_dsv4_rocm_decode_and_colocated_are_accepted(self, *_mocks):
        # The unified-KV path is no longer rejected at startup on ROCm.
        self._validate("decode")
        self._validate("null")


if __name__ == "__main__":
    unittest.main()
