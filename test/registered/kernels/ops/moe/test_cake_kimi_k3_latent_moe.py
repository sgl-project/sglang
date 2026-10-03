"""Cake Kimi-K3 LatentMoE front / tail through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for the four
LatentMoE entries, that the facade outputs are bitwise identical to runners
prepared directly through FlashInfer, and that front / tail match pure-torch
references within BF16 tolerance (logits 1e-3). Skips (with the reason) when
FlashInfer lacks the Cake module, the GPU is not sm_100a / sm_103a, or the
device does not expose the 148 SMs the plans were frozen for.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_kimi_k3_latent as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_kimi_k3_latent_moe_front,
    cake_kimi_k3_latent_moe_tail,
    cake_prepare_kimi_k3_latent_moe_front,
    cake_prepare_kimi_k3_latent_moe_tail,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.prepare_kimi_k3_latent_moe_front",
    "moe.kimi_k3_latent_moe_front",
    "moe.prepare_kimi_k3_latent_moe_tail",
    "moe.kimi_k3_latent_moe_tail",
)
HIDDEN, LATENT, NUM_EXPERTS, SHARED = (
    adapter.HIDDEN,
    adapter.LATENT,
    adapter.NUM_EXPERTS,
    adapter.SHARED_INTERMEDIATE,
)


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_kimi_k3_latent:")


def _skip_unless_supported(stage, tp, tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the Kimi-K3 LatentMoE programs")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"Kimi-K3 LatentMoE is built for sm_100a/sm_103a, device is {cc}")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != adapter.SM_COUNT:
        pytest.skip(f"plans frozen for {adapter.SM_COUNT} SMs, device has {sms}")
    device = torch.device("cuda", 0)
    from flashinfer.experimental.kimi_k3_latent_moe import cake_backend

    if not cake_backend.generated_program_available(device, stage, tp, tokens):
        pytest.skip(f"no generated LatentMoE {stage} program for tp={tp}, T={tokens}")
    return device


def _weights(device, tp, rank=0, seed=621):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def lin(n, k):
        w = torch.randn(n, k, generator=gen, dtype=torch.float32) * (k**-0.5)
        return w.to(torch.bfloat16).to(device)

    i_local = SHARED // tp
    rows = slice(rank * i_local, (rank + 1) * i_local)
    gate_weight = lin(NUM_EXPERTS, HIDDEN)
    torch.randn(NUM_EXPERTS, generator=gen)
    norm_weight = (
        (1.0 + 0.1 * torch.randn(LATENT, generator=gen)).to(torch.bfloat16).to(device)
    )
    shared_gate = lin(SHARED, HIDDEN)[rows].contiguous()
    shared_up = lin(SHARED, HIDDEN)[rows].contiguous()
    return dict(
        gate_weight=gate_weight,
        norm_weight=norm_weight,
        down_weight=lin(LATENT, HIDDEN),
        up_weight=lin(HIDDEN, LATENT),
        shared_gate_weight=shared_gate,
        shared_up_weight=shared_up,
        shared_down_weight=lin(HIDDEN, SHARED)[:, rows].contiguous(),
    )


def _situ_and_mul(gate_up):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    a = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    b = 25.0 * torch.tanh(up / 25.0)
    return (a * b).to(gate_up.dtype)


def _rmsnorm(x, weight, eps=adapter.RMS_EPS):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


@pytest.mark.parametrize("tp,tokens", [(8, 1), (8, 16), (1, 128), (8, 512)])
def test_front_matches_flashinfer_and_reference(tp, tokens):
    device = _skip_unless_supported("front", tp, tokens)
    w = _weights(device, tp)
    i_local = SHARED // tp
    gen = torch.Generator(device="cpu").manual_seed(621 + tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16).to(device)
    shared_gate_up = torch.cat(
        [w["shared_gate_weight"], w["shared_up_weight"]], 0
    ).contiguous()

    def outputs():
        return (
            torch.full(
                (tokens, NUM_EXPERTS), float("nan"), dtype=torch.float32, device=device
            ),
            torch.full(
                (tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device
            ),
            torch.full(
                (tokens, i_local), float("nan"), dtype=torch.bfloat16, device=device
            ),
        )

    logits, latent, shared_act = outputs()
    args = (x, w["gate_weight"], w["down_weight"], shared_gate_up)
    assert adapter.supports_kimi_k3_latent_moe_front(*args, logits, latent, shared_act)
    runner = cake_prepare_kimi_k3_latent_moe_front(*args, logits, latent, shared_act)
    runner()

    from flashinfer.kimi_k3_latent_moe import (
        prepare_kimi_k3_latent_moe_front as fi_prepare,
    )

    logits_fi, latent_fi, shared_fi = outputs()
    fi_prepare(*args, logits_fi, latent_fi, shared_fi, backend="cake")()
    torch.cuda.synchronize()
    assert torch.equal(logits, logits_fi)
    assert torch.equal(latent, latent_fi)
    assert torch.equal(shared_act, shared_fi)

    ref_logits = torch.nn.functional.linear(x.float(), w["gate_weight"].float())
    ref_latent = torch.nn.functional.linear(x, w["down_weight"])
    ref_shared = _situ_and_mul(
        torch.cat(
            [
                torch.nn.functional.linear(x, w["shared_gate_weight"]),
                torch.nn.functional.linear(x, w["shared_up_weight"]),
            ],
            -1,
        )
    )
    torch.testing.assert_close(logits, ref_logits, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(latent.float(), ref_latent.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        shared_act.float(), ref_shared.float(), atol=1e-2, rtol=1e-2
    )

    one_shot = cake_kimi_k3_latent_moe_front(*args, *outputs())
    torch.cuda.synchronize()
    assert torch.equal(one_shot[0], logits)


@pytest.mark.parametrize(
    "tp,rank,tokens", [(8, 3, 1), (8, 0, 16), (1, 0, 128), (8, 7, 512)]
)
def test_tail_matches_flashinfer_and_reference(tp, rank, tokens):
    device = _skip_unless_supported("tail", tp, tokens)
    w = _weights(device, tp, rank)
    i_local = SHARED // tp
    gen = torch.Generator(device="cpu").manual_seed(777 + tokens)
    # FlashInfer's own tail tests use one routed partial; the decode programs are
    # registered per exact plan and the plan depends on ``P`` (see
    # ``moe_kimi_k3_latent._tail_route_registered``).
    P = 1
    routed = torch.randn(P, tokens, LATENT, generator=gen).to(torch.bfloat16).to(device)
    shared_act = (
        torch.randn(tokens, i_local, generator=gen).to(torch.bfloat16).to(device)
    )

    def outputs():
        return (
            torch.full(
                (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
            ),
            torch.full(
                (tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device
            ),
        )

    out, y = outputs()
    args = (
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
    )
    assert adapter.supports_kimi_k3_latent_moe_tail(
        *args, out, tp=tp, rank=rank, y_workspace=y
    )
    # Two partials select a different decode instance; admission must agree with
    # FlashInfer's registry instead of letting prepare raise NotImplementedError.
    from flashinfer.experimental.kimi_k3_latent_moe import cake_backend as cb

    routed2 = torch.cat([routed, routed], 0)
    if tokens <= cb.DECODE_MAX_T:
        key2 = cb.decode_kernel_key(cb.decode_tail_plan(tokens, i_local, tp, 2))
        arch = cb.SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability()]
        registered2 = key2 in cb.KERNELS.get(arch, {})
    else:
        registered2 = True
    assert (
        adapter.supports_kimi_k3_latent_moe_tail(
            routed2, *args[1:], out, tp=tp, rank=rank, y_workspace=y
        )
        is registered2
    )
    runner = cake_prepare_kimi_k3_latent_moe_tail(
        *args, out, tp=tp, rank=rank, y_workspace=y
    )
    runner()

    from flashinfer.kimi_k3_latent_moe import (
        prepare_kimi_k3_latent_moe_tail as fi_prepare,
    )

    out_fi, y_fi = outputs()
    fi_prepare(*args, out_fi, tp=tp, rank=rank, y_workspace=y_fi, backend="cake")()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(y, y_fi)

    acc = routed[0].float()
    for p in range(1, P):
        acc = acc + routed[p].float()
    ref_y = _rmsnorm(acc.to(torch.bfloat16), w["norm_weight"])
    k_up = LATENT // tp
    cols = slice(rank * k_up, (rank + 1) * k_up)
    up = torch.nn.functional.linear(
        ref_y[:, cols].float(), w["up_weight"][:, cols].float()
    )
    shared = torch.nn.functional.linear(
        shared_act.float(), w["shared_down_weight"].float()
    )
    torch.testing.assert_close(y.float(), ref_y.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out.float(), up + shared, atol=1e-2, rtol=1e-2)

    out2, y2 = outputs()
    assert (
        cake_kimi_k3_latent_moe_tail(*args, out2, tp=tp, rank=rank, y_workspace=y2)
        is out2
    )
    torch.cuda.synchronize()
    assert torch.equal(out2, out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
