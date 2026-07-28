import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="fused_eh_norm requires CUDA",
)


def _reference(
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    embeds = inputs_embeds.float()
    prev = previous_hidden.float()
    embeds_var = embeds.pow(2).mean(dim=-1, keepdim=True)
    prev_var = prev.pow(2).mean(dim=-1, keepdim=True)
    return torch.cat(
        (
            (embeds * torch.rsqrt(embeds_var + eps) * enorm_weight.float()).to(
                inputs_embeds.dtype
            ),
            (prev * torch.rsqrt(prev_var + eps) * hnorm_weight.float()).to(
                previous_hidden.dtype
            ),
        ),
        dim=-1,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [6144, 7168])
@pytest.mark.parametrize("num_tokens", [1, 6, 128])
def test_fused_eh_norm_matches_reference(
    dtype: torch.dtype, hidden_size: int, num_tokens: int
):
    torch.manual_seed(0)
    eps = 1e-6
    inputs_embeds = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    previous_hidden = torch.randn_like(inputs_embeds)
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    actual = fused_eh_norm(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
    )
    expected = _reference(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fused_eh_norm_zero_tokens():
    hidden_size = 7168
    inputs_embeds = torch.empty(0, hidden_size, device="cuda", dtype=torch.bfloat16)
    previous_hidden = torch.empty_like(inputs_embeds)
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    actual = fused_eh_norm(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, 1e-6
    )
    assert actual.shape == (0, hidden_size * 2)
    assert actual.dtype == inputs_embeds.dtype
    assert actual.device == inputs_embeds.device


def test_fused_eh_norm_row_strided_inputs():
    torch.manual_seed(1)
    hidden_size = 7168
    eps = 1e-6
    base = torch.randn(12, hidden_size, device="cuda", dtype=torch.bfloat16)
    prev_base = torch.randn_like(base)
    inputs_embeds = base[::2]
    previous_hidden = prev_base[::2]
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    actual = fused_eh_norm(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
    )
    expected = _reference(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fused_eh_norm_rejects_unsupported_dtype():
    hidden_size = 7168
    inputs_embeds = torch.randn(1, hidden_size, device="cuda", dtype=torch.float32)
    previous_hidden = torch.randn_like(inputs_embeds)
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="unsupported dtype"):
        fused_eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, 1e-6)


def test_fused_eh_norm_rejects_unsupported_hidden_size():
    hidden_size = 5000
    inputs_embeds = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16)
    previous_hidden = torch.randn_like(inputs_embeds)
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="unsupported hidden_size"):
        fused_eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
