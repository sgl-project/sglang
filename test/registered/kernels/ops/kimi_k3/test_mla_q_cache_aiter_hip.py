# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime

from sglang.kernels.ops.kimi_k3 import mla_q_cache_aiter_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return (
            get_gfx_runtime() == "gfx950"
            and mla_q_cache_aiter_hip.supports_compute_all_q_rope()
        )
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _available(), reason="Kimi-K3 fused MLA Q/cache tests require gfx950"
)


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("SGLANG_K3_AITER_MLA_Q_CACHE_FUSION", "1")


def _inputs(tokens: int, cache_dtype=torch.bfloat16, output_dtype=None):
    torch.manual_seed(23)
    heads = 12
    q_nope = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    q_pe = torch.randn(
        tokens, heads, 64, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_nope = torch.randn(
        tokens, 1, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_pe = torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16).contiguous()
    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "k_nope": k_nope,
        "k_pe": k_pe,
        "kv_cache": torch.zeros(tokens + 4, 1, 576, device="cuda", dtype=cache_dtype),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        "positions": torch.arange(tokens, device="cuda", dtype=torch.int64),
        "k_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "q_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "cos_cache": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin_cache": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "out": torch.empty(
            tokens,
            heads,
            576,
            device="cuda",
            dtype=cache_dtype if output_dtype is None else output_dtype,
        ),
    }


def _reference(values):
    return (
        torch.cat((values["q_nope"], values["q_pe"]), dim=-1).to(values["out"].dtype),
        torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(
            values["kv_cache"].dtype
        ),
    )


@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize(
    ("cache_dtype", "output_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (dtypes.fp8, dtypes.fp8),
        (dtypes.fp8, torch.bfloat16),
    ],
)
def test_fused_mla_q_cache_identity_rope(tokens, cache_dtype, output_dtype):
    values = _inputs(tokens, cache_dtype, output_dtype)
    expected_q, expected_k = _reference(values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    assert actual is values["out"]
    torch.testing.assert_close(actual.float(), expected_q.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        values["kv_cache"][:tokens].float(),
        expected_k.float(),
        atol=0.02,
        rtol=0.02,
    )


def test_fused_mla_q_cache_skips_negative_slots():
    values = _inputs(4)
    values["slot_mapping"][-2:] = -1
    expected_q, expected_k = _reference(values)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    torch.testing.assert_close(values["out"][:2], expected_q[:2])
    torch.testing.assert_close(values["kv_cache"][:2], expected_k[:2])
    assert torch.count_nonzero(values["kv_cache"][2:]).item() == 0


def test_fused_mla_q_cache_graph_replay_uses_changed_inputs():
    values = _inputs(64)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_q_cache_aiter_hip.run(**values)
    graph.replay()
    torch.cuda.synchronize()
    first = values["out"].clone()
    values["q_nope"].copy_(torch.randn_like(values["q_nope"]))
    values["q_pe"].copy_(torch.randn_like(values["q_pe"]))
    values["k_nope"].copy_(torch.randn_like(values["k_nope"]))
    values["k_pe"].copy_(torch.randn_like(values["k_pe"]))
    values["kv_cache"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_q, expected_k = _reference(values)
    assert not torch.equal(first, values["out"])
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:64], expected_k)


def test_fused_mla_q_cache_support_is_narrow():
    values = _inputs(1)
    assert mla_q_cache_aiter_hip.covered(**values)
    values["positions"] = values["positions"].to(torch.int32)
    assert not mla_q_cache_aiter_hip.covered(**values)


def test_fused_mla_q_cache_is_opt_in(monkeypatch):
    monkeypatch.delenv("SGLANG_K3_AITER_MLA_Q_CACHE_FUSION", raising=False)
    assert not mla_q_cache_aiter_hip.enabled()
    monkeypatch.setenv("SGLANG_K3_AITER_MLA_Q_CACHE_FUSION", "1")
    assert mla_q_cache_aiter_hip.enabled()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
