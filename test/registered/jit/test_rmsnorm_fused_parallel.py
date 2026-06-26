import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="base-b-kernel-unit-1-gpu-large")

EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]

# (q_lora_rank, kv_lora_rank) pairs cover the DeepSeek-V3 q/k_nope shapes plus
# equal-dim and small-dim splits.
DIM_PAIRS = [(1536, 512), (2048, 2048), (5120, 512), (576, 192)]
BS_LIST = [1, 9, 37, 256, 2049]


def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    return out.to(x.dtype)


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("dims", DIM_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_rmsnorm_fused_parallel_matches_reference(
    batch_size: int,
    dims: tuple,
    dtype: torch.dtype,
    strided: bool,
    enable_pdl: bool,
) -> None:
    from sglang.jit_kernel.cutedsl_norm import rmsnorm_fused_parallel_cute

    torch.manual_seed(0)
    d1, d2 = dims

    if strided:
        # q and k_nope are slices of one packed latent buffer in production, so
        # they arrive row-strided; exercise that path here.
        packed = torch.randn(batch_size, d1 + d2 + 64, device=DEVICE, dtype=dtype)
        x1 = packed[:, :d1]
        x2 = packed[:, d1 : d1 + d2]
    else:
        x1 = torch.randn(batch_size, d1, device=DEVICE, dtype=dtype)
        x2 = torch.randn(batch_size, d2, device=DEVICE, dtype=dtype)

    w1 = torch.randn(d1, device=DEVICE, dtype=dtype)
    w2 = torch.randn(d2, device=DEVICE, dtype=dtype)
    out1 = torch.empty(batch_size, d1, device=DEVICE, dtype=dtype)
    out2 = torch.empty(batch_size, d2, device=DEVICE, dtype=dtype)

    rmsnorm_fused_parallel_cute(
        x1, w1, out1, x2, w2, out2, eps=EPS, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(out1, _ref(x1, w1), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out2, _ref(x2, w2), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
