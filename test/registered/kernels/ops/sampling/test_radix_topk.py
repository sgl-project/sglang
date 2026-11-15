import sys

import pytest
import torch

from sglang.kernels.ops.sampling.topk import radix_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 20, 2048, 2049, 4096])
@pytest.mark.parametrize("strided", [False, True])
def test_radix_topk_matches_torch(device, dtype, k, strided):
    g = torch.Generator(device=device).manual_seed(k)
    scores = torch.randn((37, 2 * 151936), generator=g, device=device).to(dtype)
    # A column slice is not contiguous; the op must still read it correctly.
    scores = scores[:, ::2] if strided else scores[:, :151936].contiguous()
    values, ids = radix_topk(scores, k)
    ref_values, _ = torch.topk(scores, k, dim=-1)
    assert ids.dtype == torch.int64
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(scores.gather(-1, ids), ref_values, rtol=0, atol=0)
    assert torch.all(ids.sort(-1).values.diff(dim=-1) > 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
