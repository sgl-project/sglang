import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompensatedMhcUpdateGuard(CustomTestCase):
    def test_direct_update_rejected_before_writes(self):
        with (
            envs.SGLANG_DSV41_COMPENSATED_MHC.override(True),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.default_weight_loader"
            ) as loader,
        ):
            ok, message = WeightUpdater.update_weights_from_tensor(
                SimpleNamespace(), [], load_format="direct"
            )
        self.assertFalse(ok)
        self.assertIn("SGLANG_DSV41_COMPENSATED_MHC", message)
        loader.assert_not_called()

    def test_opt_out_keeps_existing_update_support(self):
        with (
            envs.SGLANG_DSV41_COMPENSATED_MHC.override(False),
            patch(
                "sglang.kernels.ops.attention.dsv4.gemm.hpc_bf16xfp32_gemm_enabled",
                return_value=False,
            ),
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error())


if __name__ == "__main__":
    unittest.main()
