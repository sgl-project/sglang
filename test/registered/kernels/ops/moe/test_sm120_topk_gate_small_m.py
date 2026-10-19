import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe import moe_fused_gate as gate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestSmallMTop22(CustomTestCase):
    def route(self, scores, bias, enabled, **kwargs):
        with patch.dict(
            os.environ, {"SGLANG_SM120_MTP_ROUTER_TOP22": str(int(enabled))}
        ):
            return gate.moe_fused_gate(scores, bias, topk=22, **kwargs)

    def assert_bits(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_sorted_ties_weights_and_live_graph(self):
        torch.manual_seed(73)
        for m in range(2, 17):
            scores = torch.randn(m, 512, device="cuda")
            bias = torch.randn(512, device="cuda")
            for case in range(5):
                with self.subTest(m=m, case=case):
                    if case == 1:
                        scores.zero_()
                        bias.zero_()
                    elif case == 2:
                        scores[:, ::2] = 200
                        scores[:, 1::2] = -200
                        bias.mul_(1e-8)
                    elif case == 3:
                        scores.normal_()
                        scores[:, :25] = float("nan")
                        bias[25:50] = -float("inf")
                        bias[50:75] = float("inf")
                    elif case == 4:
                        scores.zero_()
                        bias.fill_(-float("inf"))
                        bias[5:10] = 0
                    reference = self.route(scores, bias, False)
                    # Prove that the production wrapper takes the new kernel.
                    with patch.object(gate, "_router_triton_kernel", None):
                        self.assert_bits(self.route(scores, bias, True), reference)
            scores.normal_()
            bias.normal_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = self.route(scores, bias, True)
            for _ in range(3):
                scores.normal_()
                bias.normal_()
                graph.replay()
                self.assert_bits(result, self.route(scores, bias, False))

    def test_normalization_scaling_and_fallback(self):
        scores = torch.randn(4, 512, device="cuda")
        bias = torch.randn(512, device="cuda")
        for norm in (False, True):
            for scale in (False, True):
                kwargs = dict(
                    renormalize=norm,
                    routed_scaling_factor=5.0,
                    apply_routed_scaling_factor_on_output=scale,
                )
                self.assert_bits(
                    self.route(scores, bias, True, **kwargs),
                    self.route(scores, bias, False, **kwargs),
                )
        for m in (1, 17):
            x = torch.randn(m, 512, device="cuda")
            with patch.object(gate, "_topk_gate_small_m", side_effect=AssertionError):
                self.assert_bits(self.route(x, bias, True), self.route(x, bias, False))
        with patch.object(gate, "_topk_gate_small_m", side_effect=AssertionError):
            self.assert_bits(
                self.route(scores.bfloat16(), bias, True),
                self.route(scores.bfloat16(), bias, False),
            )

    def test_new_router_options_fall_back(self):
        scores = torch.randn(4, 512, device="cuda")
        bias = torch.randn(512, device="cuda")
        cases = [
            dict(renormalize_epsilon=0.1),
            dict(
                num_token_non_padded=torch.tensor(2, device="cuda", dtype=torch.int32)
            ),
            dict(
                input_ids=torch.tensor([0, 1, 0, 1], device="cuda"),
                bias_alt=-bias,
                bias_alt_token_id=1,
            ),
            dict(packed_out=torch.empty((4, 22), device="cuda", dtype=torch.int32)),
        ]
        for kwargs in cases:
            with (
                self.subTest(options=list(kwargs)),
                patch.object(gate, "_topk_gate_small_m", side_effect=AssertionError),
            ):
                expected = self.route(scores, bias, False, **kwargs)
                packed = kwargs.get("packed_out")
                expected_packed = packed.clone() if packed is not None else None
                if packed is not None:
                    packed.zero_()
                self.assert_bits(self.route(scores, bias, True, **kwargs), expected)
                if packed is not None:
                    self.assertTrue(torch.equal(packed, expected_packed))


if __name__ == "__main__":
    unittest.main()
