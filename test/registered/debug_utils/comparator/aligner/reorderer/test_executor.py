import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    _reorder_zigzag_to_natural,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestZigzagToNatural:
    def test_zigzag_to_natural_cp2(self) -> None:
        """cp_size=2: zigzag order [0,3,1,2] -> natural [0,1,2,3]."""
        natural = torch.arange(24).reshape(4, 6)
        chunks = list(natural.chunk(4, dim=0))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=2)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_cp3(self) -> None:
        """cp_size=3: zigzag 162534 -> natural 123456 (1-indexed)."""
        natural = torch.arange(60).reshape(6, 10)
        chunks = list(natural.chunk(6, dim=0))

        zigzag_order: list[int] = [0, 5, 1, 4, 2, 3]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=3)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_arbitrary_dim(self) -> None:
        """Reorder along dim=1 instead of dim=0."""
        natural = torch.arange(48).reshape(3, 4, 4)
        chunks = list(natural.chunk(4, dim=1))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=1)

        result = _reorder_zigzag_to_natural(zigzagged, dim=1, cp_size=2)
        assert torch.equal(result, natural)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
