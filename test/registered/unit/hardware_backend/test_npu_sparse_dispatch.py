"""CPU contracts for selecting the NPU sparse-KV implementations."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import config, dispatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

MODEL_CONFIG = SimpleNamespace(
    num_attention_heads=128,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_head_dim=128,
    hf_config=SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
    ),
)


@pytest.mark.parametrize("fused", [False, True])
def test_manager_and_attention_follow_one_switch(monkeypatch, fused):
    prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
    calls = []
    expected = {}
    for mode, manager_name, attention_name in (
        (False, "manager", "attention"),
        (True, "fused_manager", "fused_attention"),
    ):
        manager_module = ModuleType(prefix + manager_name)
        attention_module = ModuleType(prefix + attention_name)

        def manager(*args, _mode=mode, **kwargs):
            calls.append((_mode, args, kwargs))
            return _mode

        def attention(*args, _mode=mode):
            return _mode

        manager_module.SparseKVCacheManager = manager
        attention_module.forward_sparsity_driven_kv_offload = attention
        monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
        monkeypatch.setitem(sys.modules, attention_module.__name__, attention_module)
        expected[mode] = attention

    result = dispatch.create_sparse_kv_manager(
        "pool", "allocator", 2048, fused=fused, options={"graph_batch_capacity": 4}
    )
    assert result is fused
    assert calls == [
        (
            fused,
            ("pool", "allocator", 2048),
            {"graph_batch_capacity": 4},
        )
    ]
    assert dispatch.get_sparse_kv_attention(fused=fused) is expected[fused]


def test_legacy_budget_requires_fused_mode(monkeypatch):
    monkeypatch.setattr(config, "get_schedule", lambda: SimpleNamespace(page_size=128))
    monkeypatch.setattr(config, "get_eager_max_batch_size", lambda _: 4)
    with envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(False):
        with pytest.raises(ValueError):
            config.get_sparse_kv_budget(
                model_config=MODEL_CONFIG,
                request_rows=3,
                context_len=1024,
                num_layers=2,
                top_k=512,
                kv_bytes_per_token=1152,
                graph_batch_capacity=4,
            )


def test_fused_budget_keeps_its_constraints(monkeypatch):
    monkeypatch.setattr(config, "get_parallel", lambda: SimpleNamespace(attn_tp_size=1))
    monkeypatch.setattr(
        config,
        "get_schedule",
        lambda: SimpleNamespace(page_size=128, prefill_max_requests=None),
    )
    monkeypatch.setattr(
        config,
        "get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode="null", enable_pd_role_switch=False
        ),
    )
    monkeypatch.setattr(config, "get_eager_max_batch_size", lambda _: 4)
    monkeypatch.setattr(config, "get_sparse_kv_max_query_width", lambda: 4)
    with envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True):
        with pytest.raises(ValueError, match="K=2048"):
            config.get_sparse_kv_budget(
                model_config=MODEL_CONFIG,
                request_rows=3,
                context_len=1024,
                num_layers=2,
                top_k=512,
                kv_bytes_per_token=1152,
                graph_batch_capacity=4,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
