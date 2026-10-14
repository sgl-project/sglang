"""Fixed arithmetic/stride/Graph cases; not broad source-device qualification."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.activation import SiluClampAndMulOp
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # BF16 integer views retain NaN payloads and distinguish signed zero.
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


class TestSiluClampAndMulCUDA(CustomTestCase):
    def test_single_bf16_store_and_post_silu_clamp(self):
        # A BF16 SiLU intermediate produces 0.734375 on the first case.
        for limit, values, expected in (
            (7.0, [[1.0, 129 / 128]], [[0.73828125]]),
            (1.0, [[2.0, 2.0, 2.0, -2.0]], [[1.0, -1.0]]),
        ):
            with self.subTest(limit=limit):
                packed = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
                expected = torch.tensor(expected, dtype=torch.bfloat16, device="cuda")
                actual = SiluClampAndMulOp(limit=limit)(
                    packed, backend=KernelBackend.TRITON
                )
                _assert_bytes_equal(actual, expected)

    def test_tiles_row_padding_and_offsets_preserve_unwritten_bytes(self):
        for width in (3, 1023, 1024, 1025, 2049):
            with self.subTest(width=width):
                input_storage = torch.full(
                    (3, 2 * width + 5),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                packed = input_storage[:, 1 : 1 + 2 * width]
                # Saturation gives exactly +/-49 independently of sigmoid's
                # small approximation error. Include both signs of zero.
                gates = torch.tensor([20.0, 20.0, 0.0, -0.0], device="cuda")
                ups = torch.tensor([20.0, -20.0, 1.0, 1.0], device="cuda")
                index = torch.arange(width, device="cuda") % 4
                packed[:, :width] = gates[index]
                packed[:, width:] = ups[index]
                expected = (
                    torch.tensor(
                        [49.0, -49.0, 0.0, -0.0], dtype=torch.bfloat16, device="cuda"
                    )[index]
                    .expand(3, width)
                    .contiguous()
                )
                output_storage = torch.full(
                    (3, width + 5), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                out = output_storage[:, 2 : 2 + width]
                input_before = input_storage.clone()
                output_expected = output_storage.clone()
                output_expected[:, 2 : 2 + width] = expected
                self.assertIs(
                    SiluClampAndMulOp()(packed, out=out, backend=KernelBackend.TRITON),
                    out,
                )
                _assert_bytes_equal(out, expected)
                _assert_bytes_equal(input_storage, input_before)
                _assert_bytes_equal(output_storage, output_expected)

    def test_empty_rows_return_without_launch(self):
        from sglang.kernels.ops.activation import silu_clamp_and_mul_triton

        packed = torch.empty((0, 6), dtype=torch.bfloat16, device="cuda")
        out = torch.empty((0, 3), dtype=torch.bfloat16, device="cuda")
        with patch.object(
            silu_clamp_and_mul_triton,
            "_silu_clamp_and_mul",
            side_effect=AssertionError("empty input launched a kernel"),
        ):
            actual = SiluClampAndMulOp()(packed, out=out, backend=KernelBackend.TRITON)
        self.assertIs(actual, out)

    def test_graph_replay_rewrites_poisoned_output_and_uses_current_input(self):
        packed = torch.tensor([[20.0, 20.0]], dtype=torch.bfloat16, device="cuda")
        out = torch.empty((1, 1), dtype=torch.bfloat16, device="cuda")
        op = SiluClampAndMulOp()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            op(packed, out=out, backend=KernelBackend.TRITON)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(packed, out=out, backend=KernelBackend.TRITON)
        for up, expected in ((20.0, 49.0), (-20.0, -49.0), (20.0, 49.0)):
            packed[:, 1] = up
            before = packed.clone()
            out.fill_(float("nan"))
            graph.replay()
            _assert_bytes_equal(out, torch.full_like(out, expected))
            _assert_bytes_equal(packed, before)


if __name__ == "__main__":
    unittest.main()
