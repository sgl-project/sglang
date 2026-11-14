# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the native expert wrapper's activation configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import kt_ep_wrapper

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKTExpertClamp(CustomTestCase):
    def create_weights(self, limit, tp_rank=0):
        # Exercise create_weights without constructing native kernels or a GPU runner.
        method = kt_ep_wrapper.KTEPWrapperMethod.__new__(
            kt_ep_wrapper.KTEPWrapperMethod
        )
        method.gpu_method = Mock()
        method.num_gpu_experts = 1
        method.tp_rank = tp_rank
        method.wrapper = None
        method.kt_config = SimpleNamespace(
            layer_idx=0,
            num_layers=2,
            max_deferred_experts_per_token=0,
            cpuinfer_threads=2,
            threadpool_count=1,
            weight_path="unused",
            chunked_prefill_size=512,
            method="MXFP4",
        )
        layer = SimpleNamespace(
            top_k=2,
            intermediate_size_per_partition=128,
            moe_tp_size=2,
            moe_runner_config=SimpleNamespace(swiglu_limit=limit),
        )

        # The optional kt_kernel dependency need not be installed in CPU CI.
        # Return its constructor configuration so the native-boundary value is checked.
        def native_config(swiglu_limit=0.0, **kwargs):
            return SimpleNamespace(swiglu_limit=swiglu_limit, **kwargs)

        with (
            patch.object(kt_ep_wrapper, "KTMoEWrapper", native_config, create=True),
            patch.object(
                kt_ep_wrapper, "is_building_neighbour_layer", return_value=False
            ),
        ):
            method.create_weights(
                layer=layer,
                num_experts=4,
                hidden_size=128,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
            )
        return method.wrapper

    def test_checkpoint_clamp_reaches_cpu_experts(self):
        self.assertEqual(self.create_weights(10.0).swiglu_limit, 10.0)

    def test_preserves_non_default_clamp(self):
        self.assertEqual(self.create_weights(7.0).swiglu_limit, 7.0)

    def test_unset_clamp_disables_native_clamping(self):
        self.assertEqual(self.create_weights(None).swiglu_limit, 0.0)

    def test_zero_clamp_remains_disabled(self):
        self.assertEqual(self.create_weights(0.0).swiglu_limit, 0.0)

    def test_other_tp_rank_does_not_construct_cpu_wrapper(self):
        self.assertIsNone(self.create_weights(10.0, tp_rank=1))


if __name__ == "__main__":
    unittest.main()
