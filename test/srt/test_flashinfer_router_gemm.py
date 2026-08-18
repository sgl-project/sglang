from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.gemm import flashinfer_router_gemm as router_gemm
from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v2, glm4_moe, glm4_moe_lite


@pytest.mark.parametrize(
    "hidden_dim,num_experts,out_dtype,op_name",
    [
        (7168, 128, torch.bfloat16, "mm_M1_16_K7168_N128"),
        (7168, 256, torch.float32, "mm_M1_16_K7168_N256"),
        (6144, 256, torch.float32, "mm_M1_16_K6144_N256"),
    ],
)
def test_flashinfer_router_gemm_dispatch(
    monkeypatch, hidden_dim, num_experts, out_dtype, op_name
):
    calls = []

    def fake_op(mat_a, mat_b, out, launch_with_pdl):
        calls.append((mat_a, mat_b, out, launch_with_pdl))
        assert mat_b.shape == (hidden_dim, num_experts)
        assert mat_b.stride(0) == 1
        assert out.is_contiguous()
        out.zero_()

    monkeypatch.setattr(
        router_gemm,
        "_get_flashinfer_router_gemm_ops",
        lambda: {op_name: fake_op},
    )
    hidden_states = torch.randn(7, hidden_dim, dtype=torch.bfloat16)
    router_weights = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16)

    out = router_gemm.try_flashinfer_router_gemm(
        hidden_states, router_weights, launch_with_pdl=False
    )

    assert out.shape == (7, num_experts)
    assert out.dtype == out_dtype
    assert len(calls) == 1
    assert calls[0][3] is False


@pytest.mark.parametrize(
    "hidden_states,router_weights",
    [
        (
            torch.empty(0, 7168, dtype=torch.bfloat16),
            torch.empty(256, 7168, dtype=torch.bfloat16),
        ),
        (
            torch.empty(17, 7168, dtype=torch.bfloat16),
            torch.empty(256, 7168, dtype=torch.bfloat16),
        ),
        (
            torch.empty(4, 4096, dtype=torch.bfloat16),
            torch.empty(256, 4096, dtype=torch.bfloat16),
        ),
        (
            torch.empty(4, 7168, dtype=torch.float32),
            torch.empty(256, 7168, dtype=torch.bfloat16),
        ),
    ],
)
def test_flashinfer_router_gemm_preserves_unsupported_fallbacks(
    monkeypatch, hidden_states, router_weights
):
    monkeypatch.setattr(
        router_gemm,
        "_get_flashinfer_router_gemm_ops",
        lambda: pytest.fail("unsupported inputs must not resolve FlashInfer ops"),
    )
    assert (
        router_gemm.try_flashinfer_router_gemm(hidden_states, router_weights) is None
    )


def test_flashinfer_router_gemm_kill_switch(monkeypatch):
    monkeypatch.setattr(
        router_gemm,
        "_get_flashinfer_router_gemm_ops",
        lambda: pytest.fail("disabled dispatch must not resolve FlashInfer ops"),
    )
    hidden_states = torch.empty(1, 7168, dtype=torch.bfloat16)
    router_weights = torch.empty(256, 7168, dtype=torch.bfloat16)
    with envs.SGLANG_ENABLE_FLASHINFER_ROUTER_GEMM.override(False):
        assert (
            router_gemm.try_flashinfer_router_gemm(hidden_states, router_weights)
            is None
        )


@pytest.mark.parametrize(
    "gate_cls,module",
    [
        (glm4_moe.Glm4MoeGate, glm4_moe),
        (glm4_moe_lite.Glm4MoeLiteGate, glm4_moe_lite),
    ],
)
def test_glm_gate_uses_flashinfer_router_gemm(monkeypatch, gate_cls, module):
    sentinel = torch.empty(3, 256, dtype=torch.float32)
    monkeypatch.setattr(
        module, "try_flashinfer_router_gemm", lambda *_args, **_kwargs: sentinel
    )
    gate = gate_cls(SimpleNamespace(n_routed_experts=256, hidden_size=6144))

    assert gate(torch.empty(3, 6144, dtype=torch.bfloat16)) is sentinel
    assert gate._weight_fp32 is None


def test_deepseek_gate_uses_flashinfer_router_gemm(monkeypatch):
    sentinel = torch.empty(3, 256, dtype=torch.float32)
    monkeypatch.setattr(
        deepseek_v2,
        "try_flashinfer_router_gemm",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )
    gate = deepseek_v2.MoEGate.__new__(deepseek_v2.MoEGate)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.empty(256, 7168, dtype=torch.bfloat16)
    )
    gate.is_deepseek_v4 = False
    gate.dsa_enable_prefill_cp = False
    gate.mla_enable_prefill_cp = False

    assert gate(torch.empty(3, 7168, dtype=torch.bfloat16)) is sentinel
