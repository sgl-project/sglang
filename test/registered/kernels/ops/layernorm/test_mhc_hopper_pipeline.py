import pytest
import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("num_tokens", [33, 257, 4097])
def test_hopper_compensated_mhc_repeatable_and_sharded(num_tokens):
    if not torch.cuda.is_available() or not get_platform().is_sm90:
        pytest.skip("Hopper compensated mHC projection")
    torch.manual_seed(5716 + num_tokens)
    x = torch.randn(num_tokens, 20480, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(24, 20480, device="cuda") * 0.01
    parts = mhc.split_bf16_hc_weight(weight)
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    frozen = [v.clone() for v in (x, *parts)]

    def project(rows):
        return mhc.hc_mix_stats_sinkhorn_bf16x3(
            rows, parts, scale, base, 20, 1e-20, 1e-6
        )

    expected = project(x)
    for _ in range(8):
        # Alternate a short prefill and force scratch allocation reuse.
        project(x[:32])
        actual = project(x)
        for a, b in zip(actual, expected):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))

    if num_tokens >= 257:
        # Interleaved CP ownership changes local row counts and the M tile.
        sharded = [torch.empty_like(v) for v in expected]
        for rank in range(8):
            local = project(x[rank::8].contiguous())
            for out, value in zip(sharded, local):
                out[rank::8] = value
        for a, b in zip(sharded, expected):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    for value, original in zip((x, *parts), frozen):
        assert torch.equal(value.view(torch.int16), original.view(torch.int16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
