import pytest
import torch
from sgl_kernel import moe_sum_reduce


# 8 and 104 are 8 mod 16: the shapes the bf16 vectorized path used to take and
# leave half-written. 2048 stays on that path and guards it against the gate change.
@pytest.mark.parametrize("hidden", [8, 104, 2048])
def test_bf16_vectorized_path_writes_every_column(hidden):
    """Above 256 tokens, bf16 rows that are 8 mod 16 get all columns written.

    The vectorized bf16 path sums 16-element chunks; routing such rows into it
    left their last 8 columns untouched.
    """
    x = torch.randn(257, 8, hidden, dtype=torch.bfloat16, device="cuda")
    out = torch.full((257, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe_sum_reduce(x, out, 2.5)
    expected = (x.float().sum(dim=1) * 2.5).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
