"""FLA and KDA launch decisions with a data-dependent token count.

Under torch.compile with an unbacked token count, a Python comparison on a
size raises "Could not guard on data-dependent expression". Each launch
decision must instead pick a choice that is valid for any size.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import _chunk_count_bucket
from sglang.kernels.ops.attention.fla.fused_norm_gate import (
    _is_statically_noncontiguous,
    _positive_cdiv,
    _uses_eight_row_tile,
)
from sglang.kernels.ops.attention.fla.kda import _is_small_kda_grid
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    _tree_token_tile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _compile(fn):
    return torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)


def _unbacked(num_rows: int) -> torch.Tensor:
    rows = torch.zeros(num_rows)
    torch._dynamo.decorators.mark_unbacked(rows, 0)
    return rows


class TestFLALaunchDecisionsUnderCompile(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_chunk_count_bucket_takes_the_largest_bucket_when_symbolic(self):
        self.assertEqual(_chunk_count_bucket(32), 0)
        self.assertEqual(_chunk_count_bucket(33), 1)
        self.assertEqual(_chunk_count_bucket(128), 1)
        self.assertEqual(_chunk_count_bucket(129), 2)

        compiled = _compile(
            lambda chunks: chunks + _chunk_count_bucket(chunks.shape[0])
        )
        torch.testing.assert_close(compiled(_unbacked(4)), torch.full((4,), 2.0))

    def test_kda_grid_takes_the_large_grid_path_when_symbolic(self):
        self.assertTrue(_is_small_kda_grid(256))
        self.assertFalse(_is_small_kda_grid(257))

        compiled = _compile(
            lambda chunks: chunks + int(_is_small_kda_grid(96 * chunks.shape[0]))
        )
        torch.testing.assert_close(compiled(_unbacked(2)), torch.zeros(2))

    def test_norm_gate_tile_reads_no_symbolic_rows_without_pdl(self):
        """Raised "Could not guard on data-dependent expression 6*u0 >= 8"
        with six local heads per token."""

        def choose_tile(tokens: torch.Tensor) -> torch.Tensor:
            use_bt8 = _uses_eight_row_tile(
                tokens.shape[0] * 6,
                use_pdl=False,
                is_rms_norm=True,
                activation="sigmoid",
            )
            return torch.full((1,), 8 if use_bt8 else 32)

        self.assertEqual(_compile(choose_tile)(_unbacked(5)).item(), 32)
        for rows, activation, expected in (
            (8, "sigmoid", True),
            (257, "sigmoid", False),
            (64, "swish", True),
            (65, "swish", False),
        ):
            with self.subTest(rows=rows, activation=activation):
                self.assertEqual(
                    _uses_eight_row_tile(
                        rows, use_pdl=True, is_rms_norm=True, activation=activation
                    ),
                    expected,
                )

    def test_norm_gate_grid_accepts_symbolic_rows(self):
        compiled = _compile(lambda rows: torch.ones(_positive_cdiv(rows.shape[0], 32)))
        for num_rows, expected_grid in ((1, 1), (32, 1), (33, 2), (96, 3)):
            with self.subTest(num_rows=num_rows):
                self.assertEqual(
                    compiled(torch.empty(num_rows)).shape[0], expected_grid
                )

    def test_norm_gate_copies_a_gate_whose_contiguity_is_symbolic(self):
        self.assertFalse(_is_statically_noncontiguous(torch.zeros(1, 3, 8)))
        self.assertTrue(_is_statically_noncontiguous(torch.zeros(2, 2, 3, 8)[:, 0]))

        def normalize_gate(
            storage: torch.Tensor, logical_size: torch.Tensor
        ) -> torch.Tensor:
            num_tokens = int(logical_size[0])
            torch._check_is_size(num_tokens)
            gate = storage[:num_tokens, 0]
            if _is_statically_noncontiguous(gate):
                return gate + 1
            return gate.contiguous()

        storage = torch.arange(384, dtype=torch.float32).reshape(4, 2, 6, 8)
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            actual = _compile(normalize_gate)(storage, torch.tensor([2]))
        torch.testing.assert_close(actual, storage[:2, 0].contiguous())

    def test_kda_decode_tree_tile_is_constant_without_a_tree_mask(self):
        """next_power_of_2 of the decode token count specialized it, failing
        with GuardOnDataDependentSymNode."""

        def allocate_tile(logical_size: torch.Tensor) -> torch.Tensor:
            num_tokens = int(logical_size[0])
            torch._check_is_size(num_tokens)
            return torch.ones(_tree_token_tile(num_tokens, False))

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            actual = _compile(allocate_tile)(torch.tensor([5]))
        self.assertEqual(actual.shape, (1,))
        self.assertEqual(_tree_token_tile(5, True), 8)
        self.assertEqual(_tree_token_tile(8, True), 8)


if __name__ == "__main__":
    unittest.main()
