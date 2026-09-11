import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_hook import (
    _configure_rocm_fp8_wo_a_gemm,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HOOK = "sglang.srt.arg_groups.model_hook"
_DTYPE_PROBE = "sglang.srt.model_loader.weight_utils.probe_safetensors_weight_dtype"


class TestRocmFp8WoAResolution(unittest.TestCase):
    def test_resolution_policy(self):
        model_config = SimpleNamespace(
            model_path="org/model",
            revision="requested",
            hf_config=SimpleNamespace(_commit_hash="resolved"),
        )
        cases = [
            # supported, explicit, checkpoint dtype, value, is_set, should probe
            (True, None, "F8_E4M3", True, False, True),
            (True, None, "BF16", False, True, True),
            (True, None, None, True, False, True),
            (True, True, None, True, True, False),
            (True, False, None, False, True, False),
            (False, True, None, False, True, False),
        ]
        with patch.dict(os.environ, {}, clear=False):
            for supported, explicit, dtype, value, is_set, should_probe in cases:
                with self.subTest(supported=supported, explicit=explicit, dtype=dtype):
                    flag = envs.SGLANG_OPT_FP8_WO_A_GEMM
                    flag.clear()
                    if explicit is not None:
                        flag.set(explicit)

                    with (
                        patch(
                            f"{_HOOK}._rocm_fp8_wo_a_supported",
                            return_value=supported,
                        ),
                        patch(_DTYPE_PROBE, return_value=dtype) as probe,
                    ):
                        _configure_rocm_fp8_wo_a_gemm(model_config, "/models")

                    self.assertEqual(flag.get(), value)
                    self.assertEqual(flag.is_set(), is_set)
                    if should_probe:
                        probe.assert_called_once_with(
                            "org/model",
                            ".wo_a.weight",
                            revision="resolved",
                            cache_dir="/models",
                        )
                    else:
                        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
