import unittest
from unittest.mock import patch

import torch

import sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent as xpu_fused
import sglang.srt.layers.attention.linear.kernels.kda_triton as kda_triton
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeKernel:
    def __init__(self):
        self.kwargs = None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.kwargs = kwargs

        return launch


class TestKDALowerBoundForwarding(unittest.TestCase):
    def test_triton_decode_forwards_explicit_lower_bound(self):
        expected = object()
        inputs = [object() for _ in range(10)]

        with patch.object(
            kda_triton,
            "fused_sigmoid_gating_delta_rule_update",
            return_value=expected,
            create=True,
        ) as update:
            actual = kda_triton.TritonKDAKernel().decode(
                *inputs[:5],
                A_log=inputs[5],
                dt_bias=inputs[6],
                ssm_states=inputs[7],
                cache_indices=inputs[8],
                query_start_loc=inputs[9],
                lower_bound=-5.0,
            )

        self.assertIs(actual, expected)
        self.assertEqual(update.call_args.kwargs["lower_bound"], -5.0)

    def test_xpu_kernel_forwards_explicit_lower_bound(self):
        fake_kernel = _FakeKernel()
        q = torch.empty(1, 2, 1, 4)
        k = torch.empty_like(q)
        v = torch.empty(1, 2, 1, 4)
        a = torch.empty(1, 2, 4)
        b = torch.empty(1, 2, 1)
        initial_state = torch.empty(2, 1, 4, 4)
        state_indices = torch.tensor([0, 1], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)

        with patch.object(
            xpu_fused,
            "fused_sigmoid_gating_delta_rule_update_kernel",
            fake_kernel,
        ):
            xpu_fused.fused_sigmoid_gating_delta_rule_update(
                A_log=torch.empty(1),
                a=a,
                dt_bias=torch.empty(4),
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=initial_state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                lower_bound=-5.0,
            )

        self.assertEqual(fake_kernel.kwargs["lower_bound"], -5.0)
        self.assertTrue(fake_kernel.kwargs["USE_LOWER_BOUND"])


if __name__ == "__main__":
    unittest.main()
