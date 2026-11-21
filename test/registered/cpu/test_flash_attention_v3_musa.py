import sys
from types import SimpleNamespace

from sglang.kernels.ops.attention.flash_attention_v3 import (
    _load_fa3_kernels_from_mate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=1, suite="base-b-test-cpu-arm64")


def test_mate_fa3_adapters(monkeypatch):
    calls = {}

    def fake_varlen(*args, **kwargs):
        calls["varlen"] = (args, kwargs)
        return "varlen"

    def fake_kvcache(*args, **kwargs):
        calls["kvcache"] = (args, kwargs)
        return "kvcache"

    monkeypatch.setitem(
        sys.modules,
        "mate",
        SimpleNamespace(
            flash_attn_varlen_func=fake_varlen,
            flash_attn_with_kvcache=fake_kvcache,
        ),
    )

    kernels = _load_fa3_kernels_from_mate()
    assert (
        kernels["flash_attn_varlen_func"](
            "q",
            "k",
            "v",
            "cu_seqlens_q",
            "cu_seqlens_k",
            only_qv=True,
            sinks="sinks",
            causal=True,
        )
        == "varlen"
    )
    assert calls["varlen"][1] == {"causal": True}

    assert (
        kernels["flash_attn_with_kvcache"](
            "q",
            "k_cache",
            "v_cache",
            sinks="sinks",
            causal=True,
        )
        == "kvcache"
    )
    assert calls["kvcache"][1] == {"causal": True}
