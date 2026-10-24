"""Cake Hopper (sm_90a) variable block-sparse attention through sglang.kernels.

``VariableBlockSparseAttentionWrapper(backend="cake")``: BF16 HND q/k/v,
64-token blocks, noncausal. Checks registry resolution, bitwise parity with
FlashInfer called directly, and an FP32 dense-masked reference within BF16
tolerance. Skips off sm_90a or when FlashInfer lacks ``cake_vsa_sm90``.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_sparse as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_create_variable_block_sparse_attention_wrapper_sm90,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OP = "attention.create_variable_block_sparse_attention_wrapper_sm90"


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_sparse:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_VSA_SM90_MODULE, cake.FI_VSA_SM90_JIT_MODULE, cake.FI_SPARSE_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.cake_vsa_sm90")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.VSA_SM90_ARCHS:
        pytest.skip(f"Cake VSA SM90 is built for sm_90a, device is {cc}")


@pytest.mark.parametrize("h,mb,nb,capacity", [(2, 4, 8, 3), (1, 1, 2, 2)])
def test_vsa_sm90_matches_flashinfer_and_reference(h, mb, nb, capacity):
    _skip_unless_supported()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(30)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool, device=device)
    for head in range(h):
        for row in range(mb):
            count = 1 + int(
                torch.randint(0, capacity, (1,), generator=gen, device=device)
            )
            cols = torch.randperm(nb, generator=gen, device=device)[:count]
            mask[head, row, cols] = True
    rows = torch.full((h, mb), 64, dtype=torch.int32, device=device)
    cols = torch.full((h, nb), 64, dtype=torch.int32, device=device)
    q = torch.randn(
        (h, mb * 64, 128), dtype=torch.bfloat16, device=device, generator=gen
    )
    k = torch.randn(
        (h, nb * 64, 128), dtype=torch.bfloat16, device=device, generator=gen
    )
    v = torch.randn(
        (h, nb * 64, 128), dtype=torch.bfloat16, device=device, generator=gen
    )
    assert cake.supports_variable_block_sparse_attention_sm90(q, k, v)
    scale = 128**-0.5

    def plan(wrapper):
        wrapper.plan(
            mask,
            rows,
            cols,
            h,
            h,
            128,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=scale,
            non_blocking=False,
        )
        return wrapper

    workspace = torch.empty(0, dtype=torch.uint8, device=device)
    wrapper = plan(cake_create_variable_block_sparse_attention_wrapper_sm90(workspace))
    out = wrapper.run(q, k, v)

    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    fi_wrapper = plan(VariableBlockSparseAttentionWrapper(workspace, backend="cake"))
    out_fi = fi_wrapper.run(q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    scores = torch.einsum("hmd,hnd->hmn", q.float(), k.float()) * scale
    dense = mask.repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense, float("-inf"))
    ref = torch.einsum("hmn,hnd->hmd", torch.softmax(scores, dim=-1), v.float())
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
