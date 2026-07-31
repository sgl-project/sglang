"""The contiguous GDN qkvzba split is an identity copy, so views must match it.

`qkvzba_split_reshape_cat_contiguous_views` replaces a kernel launch per
linear-attention layer on the single-token decode path. That is only sound if
its outputs are bit-identical to (and as contiguous as) the fused kernel's.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
    qkvzba_split_is_pure_view,
    qkvzba_split_reshape_cat_contiguous_views,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")


# (num_heads_qk, num_heads_v, head_qk, head_v): Qwen3.5-397B at TP8/TP4/TP2 and
# Qwen3-Next-80B at TP4.
_SHAPES = [(2, 8, 128, 128), (4, 16, 128, 128), (8, 32, 128, 128), (4, 8, 128, 128)]


@pytest.mark.parametrize("num_heads_qk,num_heads_v,head_qk,head_v", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
class TestQkvzbaSplitViews:
    @staticmethod
    def _inputs(num_heads_qk, num_heads_v, head_qk, head_v, dtype, tokens):
        qkvz = torch.randn(
            tokens,
            num_heads_qk * head_qk * 2 + num_heads_v * head_v * 2,
            dtype=dtype,
            device="cuda",
        )
        ba = torch.randn(tokens, num_heads_v * 2, dtype=dtype, device="cuda")
        return qkvz, ba

    def test_views_match_kernel(
        self, num_heads_qk, num_heads_v, head_qk, head_v, dtype
    ) -> None:
        qkvz, ba = self._inputs(num_heads_qk, num_heads_v, head_qk, head_v, dtype, 1)
        assert qkvzba_split_is_pure_view(qkvz, ba)
        args = (qkvz, ba, num_heads_qk, num_heads_v, head_qk, head_v)
        expected = fused_qkvzba_split_reshape_cat_contiguous(*args)
        actual = qkvzba_split_reshape_cat_contiguous_views(*args)
        for name, want, got in zip(("mixed_qkv", "z", "b", "a"), expected, actual):
            assert got.shape == want.shape, name
            assert got.dtype == want.dtype, name
            assert got.is_contiguous(), f"{name} must stay contiguous for consumers"
            assert torch.equal(got, want), name

    def test_multi_token_is_not_a_pure_view(
        self, num_heads_qk, num_heads_v, head_qk, head_v, dtype
    ) -> None:
        # Column slices of a multi-row projection are strided, so the fused
        # kernel must stay in charge there.
        qkvz, ba = self._inputs(num_heads_qk, num_heads_v, head_qk, head_v, dtype, 3)
        assert not qkvzba_split_is_pure_view(qkvz, ba)

    def test_non_contiguous_input_is_not_a_pure_view(
        self, num_heads_qk, num_heads_v, head_qk, head_v, dtype
    ) -> None:
        qkvz, ba = self._inputs(num_heads_qk, num_heads_v, head_qk, head_v, dtype, 1)
        wide_ba = torch.randn(1, ba.shape[1] * 2, dtype=dtype, device="cuda")
        assert not qkvzba_split_is_pure_view(qkvz, wide_ba[:, ::2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
