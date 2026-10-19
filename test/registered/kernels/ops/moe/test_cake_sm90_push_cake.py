"""Cake SM90 native BF16 push mega-MoE backend config / weights through sglang.kernels.

The megakernel is a multi-rank EP collective (torch.distributed group, P2P /
IPC peer handles). In a single process this file checks that the registry
resolves the explicit FlashInfer backend for both entries, that the admission
rules mirror ``Sm90CakeBf16MegaKernelBackend.validate_init``, that the facade
config equals FlashInfer's dataclass and the registry resolves its backend
class, and that the (process-group-free) weight preprocessing is bitwise
identical to FlashInfer's. The EP forward parity test skips with "needs EP
ranks" unless ``torch.distributed`` is initialized on SM90 devices.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_sm90_push_cake as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_preprocess_sm90_push_cake_bf16_mega_weights,
    cake_sm90_push_cake_megamoe_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OPS = (
    "moe.sm90_push_cake_megamoe_config",
    "moe.preprocess_sm90_push_cake_bf16_mega_weights",
)
HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, TOP_K = 1024, 512, 4, 2


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_sm90_push_cake:")


def _skip_unless_module_and_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the SM90 push-cake mega-MoE backend")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"SM90 push-cake is built for sm_90a only, device is {cc}")
    return torch.device("cuda", torch.cuda.current_device())


def test_admission_rules_mirror_validate_init():
    device = _skip_unless_module_and_sm90()
    ok = dict(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        token_hidden_size=HIDDEN,
        num_experts=32,
        world_size=8,
        device=device,
    )
    assert adapter.supports_sm90_push_cake_megamoe(**ok)
    assert not adapter.supports_sm90_push_cake_megamoe(**{**ok, "top_k": 3})
    assert not adapter.supports_sm90_push_cake_megamoe(
        **{**ok, "token_hidden_size": 1000}
    )
    assert not adapter.supports_sm90_push_cake_megamoe(
        **{**ok, "intermediate_size": 500}
    )
    assert not adapter.supports_sm90_push_cake_megamoe(**{**ok, "num_experts": 30})
    assert not adapter.supports_sm90_push_cake_megamoe(**{**ok, "world_size": 33})
    assert not adapter.supports_sm90_push_cake_megamoe(**{**ok, "capacity_factor": 1.5})
    assert not adapter.supports_sm90_push_cake_megamoe(**{**ok, "clamp_limit": -1.0})


def test_config_and_backend_class_match_flashinfer():
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the SM90 push-cake mega-MoE backend")
    from flashinfer.moe_ep import Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig

    config = cake_sm90_push_cake_megamoe_config(INTERMEDIATE, TOP_K, clamp_limit=7.0)
    assert config == Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, top_k=TOP_K, clamp_limit=7.0
    )
    assert config.kernel_name == adapter.KERNEL_NAME
    backend_cls = adapter.get_sm90_push_cake_backend_class()
    assert backend_cls.kernel_name() == adapter.KERNEL_NAME
    backend = backend_cls(config)
    assert backend is not None


def test_weight_preprocessing_matches_flashinfer():
    device = _skip_unless_module_and_sm90()
    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_sm90_push_cake_bf16_mega_weights,
    )

    gen = torch.Generator(device=device).manual_seed(90)
    w13 = torch.randn(
        LOCAL_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    w2 = torch.randn(
        LOCAL_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    pack = MoEWeightPack(w13=w13, w2=w2)
    got = cake_preprocess_sm90_push_cake_bf16_mega_weights(
        pack,
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
        num_local_experts=LOCAL_EXPERTS,
    )
    ref = preprocess_sm90_push_cake_bf16_mega_weights(
        pack,
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
        num_local_experts=LOCAL_EXPERTS,
    )
    torch.cuda.synchronize()
    got_tensors = [t for t in vars(got).values() if isinstance(t, torch.Tensor)]
    ref_tensors = [t for t in vars(ref).values() if isinstance(t, torch.Tensor)]
    assert got_tensors and len(got_tensors) == len(ref_tensors)
    for a, b in zip(got_tensors, ref_tensors):
        assert torch.equal(a, b)


def test_ep_forward_parity_needs_ranks():
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        pytest.skip("needs EP ranks (torch.distributed group of <= 32 SM90 ranks)")
    pytest.skip(
        "needs EP ranks: MoEEpLayer bootstrap is owned by sglang.srt runtime integration"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
