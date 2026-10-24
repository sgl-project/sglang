"""Cake packed Kimi-K3 KDA decode (T=1, H=12, K=V=128) through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend; that the
facade output and the in-place BF16 state pool are bitwise identical to
``flashinfer.kda_decode.packed_kda_decode``; that the result matches a
pure-torch step within BF16 tolerance (``scale=1/sqrt(128)``, L2 eps 1e-6,
``lower_bound=-5``); and that a second launch into the same caller-owned
output with new data and new indices is correct (the allocation-free form that
CUDA-graph replay relies on; ``-1`` rows are inactive). Skips with the reason
when FlashInfer lacks the module or the GPU is outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_kda as cake_kda
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import cake_kda_packed_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "attention.kda_packed_decode"
HEADS = cake_kda.PACKED_HEADS
HEAD_DIM = cake_kda.HEAD_DIM
WIDTH = HEADS * HEAD_DIM


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_linear_kda:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_kda.FI_MODULE_DECODE, cake_kda.FI_JIT_MODULE_PACKED
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.jit.cake_kda_packed_t1")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_kda.ARCHS:
        pytest.skip(f"Cake packed KDA decode is built for sm_100a/sm_103a, got {cc}")


def _reference(mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state, indices):
    batch = mixed_qkv.shape[0]
    packed = mixed_qkv.float().view(batch, 3, HEADS, HEAD_DIM)
    q_raw, k_raw, value = packed.unbind(1)
    q = q_raw * torch.rsqrt(q_raw.square().sum(-1, keepdim=True) + 1e-6)
    q = q * HEAD_DIM**-0.5
    k = k_raw * torch.rsqrt(k_raw.square().sum(-1, keepdim=True) + 1e-6)
    gate = raw_gate.float().view(batch, HEADS, HEAD_DIM) + dt_bias.view(HEADS, HEAD_DIM)
    decay = torch.exp(
        cake_kda.PACKED_LOWER_BOUND * torch.sigmoid(A_log.exp()[None, :, None] * gate)
    )
    beta = torch.sigmoid(raw_beta.float())
    active = indices >= 0
    safe = indices.clamp_min(0).long()
    selected = state.index_select(0, safe).float() * decay[:, :, None, :]
    pred = torch.einsum("bhvk,bhk->bhv", selected, k)
    delta = (value - pred) * beta[:, :, None]
    updated = selected + delta[..., None] * k[:, :, None, :]
    out = torch.einsum("bhvk,bhk->bhv", updated, q)
    state.index_copy_(0, indices[active].long(), updated[active].to(state.dtype))
    return torch.where(active[:, None, None], out, 0.0).unsqueeze(1)


def _inputs(batch, slots, device):
    mixed_qkv = torch.randn(batch, 3 * WIDTH, device=device, dtype=torch.bfloat16)
    raw_gate = torch.randn(batch, WIDTH, device=device, dtype=torch.bfloat16)
    raw_beta = torch.randn(batch, HEADS, device=device, dtype=torch.bfloat16)
    perm = torch.randperm(slots, device=device)[:batch].to(torch.int32)
    perm[0] = -1  # inactive graph-padding row
    return mixed_qkv, raw_gate, raw_beta, perm


@pytest.mark.parametrize("batch", [5, 40])  # tile8 (<32 rows) and tile16 paths
def test_matches_flashinfer_and_reference(batch):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    slots = batch + 3
    A_log = torch.randn(HEADS, device=device) * 0.1
    dt_bias = torch.randn(WIDTH, device=device) - 2.0
    state = (torch.randn(slots, HEADS, HEAD_DIM, HEAD_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    output = torch.empty(batch, 1, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    from flashinfer.kda_decode import packed_kda_decode as fi_direct

    for _round in range(2):  # second launch: new data + indices, same buffers
        mixed_qkv, raw_gate, raw_beta, indices = _inputs(batch, slots, device)
        assert cake_kda.supports_kda_packed_decode(
            mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state, indices, output
        )
        state_fi = state.clone()
        state_ref = state.clone()
        got = cake_kda_packed_decode(
            mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state, indices, output
        )
        assert got is output
        got_fi = fi_direct(
            mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_fi, indices
        )
        torch.cuda.synchronize()
        assert torch.equal(output, got_fi)
        assert torch.equal(state, state_fi)
        expected = _reference(
            mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_ref, indices
        )
        torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            state.float(), state_ref.float(), atol=1e-2, rtol=1e-2
        )
        assert torch.equal(output[0], torch.zeros_like(output[0]))


def test_supports_rejects_other_head_counts():
    _skip_unless_supported()
    device = torch.device("cuda")
    batch, heads = 2, 8
    mixed_qkv = torch.randn(batch, 3 * heads * HEAD_DIM, device=device).to(
        torch.bfloat16
    )
    raw_gate = torch.randn(batch, heads * HEAD_DIM, device=device).to(torch.bfloat16)
    raw_beta = torch.randn(batch, heads, device=device).to(torch.bfloat16)
    state = torch.zeros(4, heads, HEAD_DIM, HEAD_DIM, device=device).to(torch.bfloat16)
    indices = torch.arange(batch, device=device, dtype=torch.int32)
    assert not cake_kda.supports_kda_packed_decode(
        mixed_qkv,
        raw_gate,
        raw_beta,
        torch.zeros(heads, device=device),
        torch.zeros(heads * HEAD_DIM, device=device),
        state,
        indices,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
