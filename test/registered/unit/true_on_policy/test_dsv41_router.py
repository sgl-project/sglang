"""TOP DeepSeek-V4.1 router: the log1p sqrtsoftplus gate is V4.1's program and V4's is not, nor the reverse."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.true_on_policy.dsv4 import validate_top_dsv4_topk

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _topk(log1p):
    return SimpleNamespace(
        topk_config=SimpleNamespace(
            scoring_func="sqrtsoftplus",
            correction_bias=object(),
            renormalize=True,
            use_grouped_topk=False,
            num_shared_topk_slots=0,
            custom_routing_function=None,
            apply_routed_scaling_factor_on_output=False,
            sqrtsoftplus_log1p=log1p,
            torch_native=False,
        )
    )


class TestDsv41Router(CustomTestCase):
    def _validate(self, log1p, v41):
        exec_ = SimpleNamespace(
            features=SimpleNamespace(enable_return_routed_experts=False)
        )
        with (
            patch(
                "sglang.srt.true_on_policy.config.is_true_on_policy_enabled",
                return_value=True,
            ),
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_),
        ):
            validate_top_dsv4_topk(
                _topk(log1p), is_deepseek_v4=True, is_deepseek_v41=v41
            )

    def test_v41_refuses_the_v4_softplus(self):
        with self.assertRaisesRegex(NotImplementedError, "sqrtsoftplus"):
            self._validate(log1p=False, v41=True)

    def test_v4_refuses_the_v41_softplus(self):
        with self.assertRaisesRegex(NotImplementedError, "sqrtsoftplus"):
            self._validate(log1p=True, v41=False)


if __name__ == "__main__":
    unittest.main()
