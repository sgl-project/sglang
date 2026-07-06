import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.commit_kv_proj import (
    _dequant_linear_weight,
    commit_kv_proj,
    commit_kv_proj_fused,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kv proj needs CUDA"
)

HIDDEN = 1024
HEAD_DIM = 576
NUM_STAGES = 3


class _Bf16Linear(torch.nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(x, self.weight), None


def _make_linears(device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return [
        _Bf16Linear(
            (torch.randn(HEAD_DIM, HIDDEN, device=device, generator=g) * 0.02).to(
                torch.bfloat16
            )
        )
        for _ in range(NUM_STAGES)
    ]


@pytest.mark.parametrize("num_tokens", [1, 8, 56])
def test_fused_matches_per_stage_loop(num_tokens):
    device = torch.device("cuda")
    linears = _make_linears(device, seed=num_tokens)
    g = torch.Generator(device=device).manual_seed(999 + num_tokens)
    main_x = (torch.randn(num_tokens, HIDDEN, device=device, generator=g) * 0.5).to(
        torch.bfloat16
    )

    ref = commit_kv_proj(main_x=main_x, wkv_linears=linears)
    got = commit_kv_proj_fused(main_x=main_x, wkv_linears=linears)

    assert len(got) == len(ref) == NUM_STAGES
    for kv_got, kv_ref in zip(got, ref):
        assert kv_got.shape == kv_ref.shape
        assert kv_got.is_contiguous()
        torch.testing.assert_close(kv_got.float(), kv_ref.float(), rtol=2e-2, atol=2e-3)


def test_dequant_fp8_blockwise_weight():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(3)
    out_dim, in_dim, block = 192, 384, 128
    weight_f32 = torch.randn(out_dim, in_dim, device=device, generator=g)
    weight_fp8 = weight_f32.to(torch.float8_e4m3fn)
    scale = (
        torch.rand(
            (out_dim + block - 1) // block,
            (in_dim + block - 1) // block,
            device=device,
            generator=g,
        )
        + 0.5
    )

    class _Fp8Stub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = weight_fp8
            self.weight_scale_inv = scale

    got = _dequant_linear_weight(_Fp8Stub())

    expected = (
        weight_fp8.to(torch.float32)
        * scale.repeat_interleave(block, dim=0)[:out_dim].repeat_interleave(
            block, dim=1
        )[:, :in_dim]
    ).to(torch.bfloat16)
    assert torch.equal(got, expected)
