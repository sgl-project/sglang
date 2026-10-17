"""Cake MXFP8 MegaMoE EP16 (NVSHMEM, exactly 16 ranks) through sglang.kernels.

The kernel is a multi-rank collective. In a single process this file checks
that the registry resolves the explicit FlashInfer backend for both entries,
that the admission check returns ``False`` without an initialized 16-rank
process group, that the session factory raises cleanly without
``torch.distributed``, and that the weight preprocessing (a rank-local,
process-group-free setup path) is bitwise identical to FlashInfer's. The
16-rank parity test skips with "needs 16 ranks" unless ``torch.distributed``
is initialized with world size 16 on sm_103a devices.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_mxfp8_megamoe_ep16 as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_create_mxfp8_megamoe_ep16_session,
    cake_preprocess_mxfp8_megamoe_ep16_weights,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.preprocess_mxfp8_megamoe_ep16_weights",
    "moe.create_mxfp8_megamoe_ep16_session",
)


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_mxfp8_megamoe_ep16:")


def _dist_world_size():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 0


def _skip_unless_module_and_sm103():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks moe_ep.cake_mxfp8_megamoe_ep16")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(
            f"Cake MXFP8 MegaMoE EP16 is built for sm_103a only, device is {cc}"
        )
    return torch.device("cuda", torch.cuda.current_device())


def test_supports_is_false_without_process_group():
    if _dist_world_size():
        pytest.skip(
            "torch.distributed is initialized; single-process check not applicable"
        )
    topk_ids = torch.zeros(16, adapter.TOP_K, dtype=torch.int64)
    assert adapter.supports_mxfp8_megamoe_ep16(topk_ids) is False


def test_session_factory_raises_cleanly_without_process_group():
    if _dist_world_size():
        pytest.skip(
            "torch.distributed is initialized; single-process check not applicable"
        )
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks moe_ep.cake_mxfp8_megamoe_ep16")
    weights_cls = adapter.get_mxfp8_megamoe_ep16_weights_class()
    dummy = torch.zeros(1)
    weights = weights_cls(w13=dummy, w13_scale=dummy, w2=dummy, w2_scale=dummy)
    topk_ids = torch.zeros(16, adapter.TOP_K, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
        cake_create_mxfp8_megamoe_ep16_session(weights, topk_ids)


def test_weight_preprocessing_matches_flashinfer():
    device = _skip_unless_module_and_sm103()
    gen = torch.Generator(device=device).manual_seed(16)
    w13 = torch.randn(
        adapter.LOCAL_EXPERTS,
        2 * adapter.INTERMEDIATE,
        adapter.HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    w2 = torch.randn(
        adapter.LOCAL_EXPERTS,
        adapter.HIDDEN,
        adapter.INTERMEDIATE,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    assert adapter.supports_mxfp8_megamoe_ep16_weights(w13, w2)
    got = cake_preprocess_mxfp8_megamoe_ep16_weights(w13, w2)

    from flashinfer.moe_ep import preprocess_cake_mxfp8_megamoe_ep16_weights

    ref = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)
    torch.cuda.synchronize()
    for name in ("w13", "w13_scale", "w2", "w2_scale"):
        a, b = getattr(got, name), getattr(ref, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    assert got.w13.dtype == torch.float8_e4m3fn
    assert got.w13_scale.dtype == torch.uint8


def test_session_parity_needs_16_ranks():
    if _dist_world_size() != adapter.WORLD_SIZE:
        pytest.skip("needs 16 ranks")
    device = _skip_unless_module_and_sm103()
    import torch.distributed as dist

    gen = torch.Generator(device=device).manual_seed(100 + dist.get_rank())
    w13 = torch.randn(
        adapter.LOCAL_EXPERTS,
        2 * adapter.INTERMEDIATE,
        adapter.HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    w2 = torch.randn(
        adapter.LOCAL_EXPERTS,
        adapter.HIDDEN,
        adapter.INTERMEDIATE,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    weights = cake_preprocess_mxfp8_megamoe_ep16_weights(w13, w2)
    tokens = 16
    topk_ids = (
        torch.arange(tokens * adapter.TOP_K, device=device).view(tokens, adapter.TOP_K)
        + dist.get_rank() * tokens * adapter.TOP_K
    ) % adapter.NUM_EXPERTS
    topk_ids = topk_ids.to(torch.int64).contiguous()
    assert adapter.supports_mxfp8_megamoe_ep16(topk_ids)
    session = cake_create_mxfp8_megamoe_ep16_session(weights, topk_ids)
    hidden = torch.randn(
        tokens, adapter.HIDDEN, device=device, dtype=torch.bfloat16, generator=gen
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, adapter.TOP_K, device=device, generator=gen), -1
    ).float()
    out = session.run(
        hidden, topk_ids, topk_weights, out=session.workspace_output
    ).clone()
    out_again = session.run(
        hidden, topk_ids, topk_weights, out=session.workspace_output
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_again)
    assert out.shape == (tokens, adapter.HIDDEN) and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
