import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    precompute_freqs_cis,
    set_batched_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _build(batch: int, n_heads: int, rope_dim: int):
    torch.manual_seed(0)
    x = torch.randn(batch, n_heads, rope_dim, device="cuda", dtype=torch.float32)
    freqs = precompute_freqs_cis(rope_dim, batch, 0, 10000.0, 1.0, 32, 1).to("cuda")
    return x, freqs


def _apply(x, freqs, batched: bool):
    previous = getattr(
        sys.modules[apply_rotary_emb_triton.__module__], "_USE_BATCHED_ROPE"
    )
    set_batched_rope(batched)
    try:
        return apply_rotary_emb_triton(x.clone(), freqs)
    finally:
        set_batched_rope(previous)


# rope_dim 6 / 96 / 192 are not powers of two, so the flat kernel pads its column
# block to RD = next_power_of_2(rope_dim); those extra columns must not be read,
# rotated or written back. 64 and 128 are the power-of-two controls.
@pytest.mark.parametrize("rope_dim", [6, 64, 96, 128, 192])
@pytest.mark.parametrize("batch,n_heads", [(4, 8), (2, 4)])
class TestApplyRotaryEmbFlat:
    def test_matches_per_token_kernel(self, rope_dim: int, batch: int, n_heads: int):
        """The flat kernel must agree with the per-token kernel it replaces."""
        x, freqs = _build(batch, n_heads, rope_dim)
        reference = _apply(x, freqs, batched=False)
        result = _apply(x, freqs, batched=True)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_leaves_padding_columns_alone(
        self, rope_dim: int, batch: int, n_heads: int
    ):
        """The rotation must not reach past rope_dim into the next head's data."""
        x, freqs = _build(batch, n_heads, rope_dim)
        # A wider buffer whose tail columns are a known sentinel: anything the
        # kernel writes past rope_dim lands on them.
        padded = torch.full(
            (batch, n_heads, 2 * rope_dim), -12345.0, device="cuda", dtype=torch.float32
        )
        padded[:, :, :rope_dim] = x
        view = padded[:, :, :rope_dim]

        set_batched_rope(True)
        try:
            apply_rotary_emb_triton(view, freqs)
        finally:
            set_batched_rope(False)

        untouched = padded[:, :, rope_dim:]
        assert torch.equal(
            untouched, torch.full_like(untouched, -12345.0)
        ), f"{int((untouched != -12345.0).sum())} sentinel elements past rope_dim were overwritten"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
