import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.build_step_local import (
    BuildStepLocal,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@requires_cuda
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize(
    "org_width,per_partition", [(32320, 32320), (32320, 32384), (5000, 8192)]
)
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_torch(bs, org_width, per_partition, bias_dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    bias = (torch.randn(bs, org_width, device=device) * 3.0).to(bias_dtype)
    base_local = torch.randn(bs, per_partition, device=device, dtype=torch.float32)
    ref = BuildStepLocal.torch(bias=bias, base_local=base_local)
    got = BuildStepLocal.triton(bias=bias, base_local=base_local)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)


@requires_cuda
def test_padding_columns_are_pure_base():
    device = torch.device("cuda")
    org_width, per_partition = 100, 128
    bias = torch.randn(1, org_width, device=device)
    base_local = torch.randn(1, per_partition, device=device, dtype=torch.float32)
    got = BuildStepLocal.triton(bias=bias, base_local=base_local)
    assert torch.equal(got[:, org_width:], base_local[:, org_width:])


def test_torch_reference_matches_manual_pad_add():
    torch.manual_seed(1)
    org_width, per_partition = 100, 128
    bias = torch.randn(1, org_width)
    base_local = torch.randn(1, per_partition, dtype=torch.float32)
    padded = torch.nn.functional.pad(bias.float(), (0, per_partition - org_width))
    manual = base_local + padded
    got = BuildStepLocal.torch(bias=bias, base_local=base_local)
    assert torch.equal(got, manual)
