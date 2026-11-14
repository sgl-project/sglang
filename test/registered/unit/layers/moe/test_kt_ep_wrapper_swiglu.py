"""The gated activation's parameters reach kt-kernel's CPU experts (KTMoEWrapper)."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import kt_swiglu_kwargs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**kwargs):
    return MoeRunnerConfig(activation="silu", is_gated=True, **kwargs)


class TestKtSwigluKwargs(CustomTestCase):
    def test_plain_silu_passes_nothing(self):
        self.assertEqual(kt_swiglu_kwargs(_config()), {})
        self.assertEqual(kt_swiglu_kwargs(None), {})

    def test_clamped_swiglu_passes_the_limit(self):
        # DeepSeek-V4 / GLM-5: gate <= limit, |up| <= limit before silu(gate) * up
        self.assertEqual(
            kt_swiglu_kwargs(_config(swiglu_limit=10.0)), {"swiglu_limit": 10.0}
        )

    def test_swiglu_oai_passes_alpha_and_clamp(self):
        # MiniMax-M3: gate * sigmoid(alpha * gate) * (up + 1), clamped at 7
        cfg = _config(gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0)
        self.assertEqual(
            kt_swiglu_kwargs(cfg), {"swiglu_alpha": 1.702, "swiglu_limit": 7.0}
        )

    def test_swiglu_oai_without_a_clamp(self):
        self.assertEqual(
            kt_swiglu_kwargs(_config(gemm1_alpha=1.702)), {"swiglu_alpha": 1.702}
        )

    def test_other_beta_is_refused(self):
        with self.assertRaisesRegex(ValueError, "adds 1 to up"):
            kt_swiglu_kwargs(_config(gemm1_alpha=1.702, gemm1_beta=0.0))


class TestCreateWeightsForwardsTheActivation(CustomTestCase):
    def test_wrapper_gets_the_layer_activation(self):
        """create_weights hands the layer's moe_runner_config parameters to KTMoEWrapper."""
        captured = {}

        def fake_wrapper(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        method = kt_ep_wrapper.KTEPWrapperMethod.__new__(
            kt_ep_wrapper.KTEPWrapperMethod
        )
        method.tp_rank = 0
        method.num_gpu_experts = 1
        method.gpu_method = SimpleNamespace(create_weights=lambda **kw: None)
        method.kt_config = SimpleNamespace(
            layer_idx=3,
            cpuinfer_threads=1,
            threadpool_count=1,
            weight_path="unused",
            chunked_prefill_size=8,
            method="NVFP4",
            max_deferred_experts_per_token=0,
            num_layers=None,
        )
        layer = SimpleNamespace(
            top_k=4,
            intermediate_size_per_partition=32,
            moe_tp_size=1,
            moe_runner_config=_config(
                gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0
            ),
        )
        with mock.patch.object(
            kt_ep_wrapper, "KTMoEWrapper", fake_wrapper, create=True
        ):
            method.create_weights(
                layer,
                num_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                params_dtype=None,
            )
        self.assertEqual(captured.get("swiglu_alpha"), 1.702)
        self.assertEqual(captured.get("swiglu_limit"), 7.0)


if __name__ == "__main__":
    unittest.main()
