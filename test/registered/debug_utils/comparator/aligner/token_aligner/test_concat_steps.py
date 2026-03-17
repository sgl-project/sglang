import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.concat_steps import (
    execute_token_aligner_concat_steps,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestExecuteConcat:
    def test_single_step_equal_length(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(x={0: x}, y={0: y}),
        )
        assert torch.equal(result.x, x)
        assert torch.equal(result.y, y)

    def test_truncates_to_min(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([5.0, 6.0])
        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(x={0: x}, y={0: y}),
        )
        assert torch.equal(result.x, torch.tensor([1.0, 2.0]))
        assert torch.equal(result.y, y)

    def test_multi_step_sorted_concat(self) -> None:
        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(
                x={1: torch.tensor([3.0, 4.0]), 0: torch.tensor([1.0, 2.0])},
                y={0: torch.tensor([5.0, 6.0, 7.0, 8.0])},
            ),
        )
        assert torch.equal(result.x, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.equal(result.y, torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_named_token_dim_nonzero(self) -> None:
        """Token dim at dim=1 (not dim=0) — concat and truncate along correct dim."""
        # shape [2, 3, 4]: dim0=batch, dim1=token, dim2=hidden
        x_step0 = torch.randn(2, 3, 4).refine_names("b", "t", "h")
        x_step1 = torch.randn(2, 5, 4).refine_names("b", "t", "h")
        y_step0 = torch.randn(2, 6, 4).refine_names("b", "t", "h")

        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(
                x={0: x_step0, 1: x_step1},
                y={0: y_step0},
            ),
        )

        # x: 3+5=8 tokens; y: 6 tokens → truncate to 6
        assert result.x.shape == (2, 6, 4)
        assert result.y.shape == (2, 6, 4)

    def test_named_dims_no_token_dim_fallback(self) -> None:
        """Named dims without t or s → fallback to dim 0."""
        x = torch.randn(4, 8).refine_names("b", "h")
        y = torch.randn(3, 8).refine_names("b", "h")
        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(x={0: x}, y={0: y}),
        )
        assert result.x.shape == (3, 8)
        assert result.y.shape == (3, 8)

    def test_seq_dim_fallback(self) -> None:
        """Named dims with s but no t → uses s as token dim."""
        x = torch.randn(2, 5, 4).refine_names("b", "s", "h")
        y = torch.randn(2, 3, 4).refine_names("b", "s", "h")
        result: Pair[torch.Tensor] = execute_token_aligner_concat_steps(
            tensor_of_step_pair=Pair(x={0: x}, y={0: y}),
        )
        assert result.x.shape == (2, 3, 4)
        assert result.y.shape == (2, 3, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
