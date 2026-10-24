"""Cake frozen MegaMoE top-k reducer through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for both
entries, that the facade output is bitwise identical to calling FlashInfer's
``run_cake_megamoe_topk_reduce`` directly, and that the BF16 reduction matches
an FP32-accumulated torch reference within BF16 tolerance (rows past
``num_tokens`` untouched). Skips (with the reason) when FlashInfer lacks the
module or the GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_megamoe_topk_reduce as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_load_megamoe_topk_reduce_module,
    cake_megamoe_topk_reduce,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = ("moe.load_megamoe_topk_reduce_module", "moe.megamoe_topk_reduce")


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.moe_megamoe_topk_reduce:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.jit.cake_megamoe_topk_reduce"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(
            f"the frozen reducer is published for sm_100a/sm_103a, device is {cc}"
        )
    return torch.device("cuda", 0)


@pytest.mark.parametrize("capacity,num_tokens", [(256, 1), (256, 200), (4096, 4096)])
def test_matches_flashinfer_and_reference(capacity, num_tokens):
    device = _skip_unless_supported()
    gen = torch.Generator(device=device).manual_seed(capacity + num_tokens)
    partials = torch.randn(
        capacity,
        adapter.TOP_K,
        adapter.HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    out = torch.full(
        (capacity, adapter.HIDDEN), float("nan"), device=device, dtype=torch.bfloat16
    )
    assert adapter.supports_megamoe_topk_reduce(partials, out, num_tokens)

    module = cake_load_megamoe_topk_reduce_module(device)
    from flashinfer.jit.cake_megamoe_topk_reduce import (
        get_cake_megamoe_topk_reduce_module,
        run_cake_megamoe_topk_reduce,
    )

    assert module is get_cake_megamoe_topk_reduce_module(device)
    assert cake_megamoe_topk_reduce(partials, out, num_tokens) is out

    out_fi = torch.full_like(out, float("nan"))
    run_cake_megamoe_topk_reduce(partials, out_fi, num_tokens)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out_fi.view(torch.int16))

    expected = partials[:num_tokens].float().sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(
        out[:num_tokens].float(), expected.float(), atol=1e-2, rtol=1e-2
    )
    if num_tokens < capacity:
        assert torch.isnan(out[num_tokens:]).all()


def test_supports_rejects_other_shapes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", 0)
    partials = torch.zeros(
        128, adapter.TOP_K, adapter.HIDDEN, device=device, dtype=torch.bfloat16
    )
    out = torch.zeros(128, adapter.HIDDEN, device=device, dtype=torch.bfloat16)
    assert not adapter.supports_megamoe_topk_reduce(partials, out, 8)  # capacity 128
    partials = torch.zeros(256, 8, adapter.HIDDEN, device=device, dtype=torch.bfloat16)
    out = torch.zeros(256, adapter.HIDDEN, device=device, dtype=torch.bfloat16)
    assert not adapter.supports_megamoe_topk_reduce(partials, out, 8)  # top-k 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
