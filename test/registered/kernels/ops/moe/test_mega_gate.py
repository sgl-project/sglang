"""DeepGEMM MegaGate parity and graph replay with nonfinite padding."""

import unittest

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate, is_mega_gate_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestMegaGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("MegaGate requires SM10x")
        if not is_mega_gate_available():
            raise unittest.SkipTest("DeepGEMM does not expose bf16_mega_gate")

    @staticmethod
    def _reference(
        x, weight, top_k, bias, image_bias, image_mask, valid_mask, hash_ids
    ):
        x = torch.where(valid_mask[:, None], x, 0)
        scores = torch.log1p(torch.exp(linear_bf16_fp32(x, weight))).sqrt()
        if hash_ids is not None:
            ids = hash_ids
        else:
            row_bias = (
                torch.where(image_mask[:, None], image_bias, bias)
                if image_mask is not None
                else bias
            )
            ids = (scores + row_bias).topk(top_k, dim=-1).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.5
        return weights.masked_fill(~valid_mask[:, None], 0), ids.masked_fill(
            ~valid_mask[:, None], -1
        )

    def _assert_routes_close(self, actual, expected):
        actual_weights, actual_ids = actual
        expected_weights, expected_ids = expected
        actual_ids, order = actual_ids.sort(dim=-1)
        actual_weights = actual_weights.gather(1, order)
        expected_ids, order = expected_ids.sort(dim=-1)
        expected_weights = expected_weights.gather(1, order)
        torch.testing.assert_close(actual_ids, expected_ids)
        torch.testing.assert_close(
            actual_weights, expected_weights, rtol=1e-3, atol=1e-4
        )

    def test_routing_and_cuda_graph_replay(self):
        torch.manual_seed(42)
        # Target Flash and reduced expert-count draft gates, at exact H=5120.
        for num_experts, top_k in ((384, 6), (128, 6)):
            for tokens in (17, 32, 128, 512):
                for routing in ("text", "image", "hash"):
                    with self.subTest(
                        num_experts=num_experts, tokens=tokens, routing=routing
                    ):
                        x = torch.randn(
                            (tokens, 5120), device="cuda", dtype=torch.bfloat16
                        )
                        weight = (
                            torch.randn(
                                (num_experts, 5120), device="cuda", dtype=torch.bfloat16
                            )
                            * 0.001
                        )
                        # Separate selection boundaries from BF16 GEMM roundoff.
                        bias = torch.arange(num_experts, device="cuda").float() * 0.25
                        image_bias = bias.flip(0) if routing == "image" else None
                        image_mask = (
                            torch.zeros(tokens, device="cuda", dtype=torch.bool)
                            if routing == "image"
                            else None
                        )
                        hash_ids = (
                            torch.arange(top_k, device="cuda")
                            .expand(tokens, -1)
                            .clone()
                            if routing == "hash"
                            else None
                        )
                        valid_mask = torch.ones(tokens, device="cuda", dtype=torch.bool)

                        def run():
                            return bf16_mega_gate(
                                x,
                                weight,
                                top_k,
                                routed_scaling_factor=1.5,
                                ep_rank=0,
                                bias=bias,
                                image_bias=image_bias,
                                image_token_mask=image_mask,
                                valid_token_mask=valid_mask,
                                hash_topk_ids=hash_ids,
                            )

                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.stream(stream):
                            # DeepGEMM owns per-stream barriers: initialize them
                            # on the capture stream before entering capture.
                            for _ in range(3):
                                run()
                            with torch.cuda.graph(graph, stream=stream):
                                actual = run()
                        torch.cuda.current_stream().wait_stream(stream)
                        for valid_rows, padding in (
                            (tokens, float("nan")),
                            (tokens // 2, float("nan")),
                            (0, float("inf")),
                            (tokens, float("nan")),
                        ):
                            x.normal_()
                            valid_mask.copy_(
                                torch.arange(tokens, device="cuda") < valid_rows
                            )
                            x[valid_rows:] = padding
                            if image_mask is not None:
                                image_mask.logical_not_()
                            if hash_ids is not None:
                                # The external kernel also writes this buffer.
                                # Refresh routes after a replay masks padded IDs.
                                hash_ids.copy_(
                                    torch.arange(top_k, device="cuda")
                                    .expand(tokens, -1)
                                    .roll(1, dims=-1)
                                )
                            expected = self._reference(
                                x,
                                weight,
                                top_k,
                                bias,
                                image_bias,
                                image_mask,
                                valid_mask,
                                hash_ids,
                            )
                            graph.replay()
                            self._assert_routes_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
