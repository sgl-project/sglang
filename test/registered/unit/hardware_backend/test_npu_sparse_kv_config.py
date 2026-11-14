"""CPU contracts for NPU sparse-KV configuration."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import config
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


def test_fused_policy_rejects_invalid_model_topk(monkeypatch):
    monkeypatch.setattr(config, "is_npu", lambda: True)
    monkeypatch.setattr(config, "is_deepseek_dsa", lambda _: True)
    monkeypatch.setattr(config, "attention_backends", lambda: ("ascend", "ascend"))
    monkeypatch.setattr(
        config, "get_schedule", lambda: SimpleNamespace(max_running_requests=4)
    )
    monkeypatch.setattr(
        config, "get_disagg", lambda: SimpleNamespace(disaggregation_mode="null")
    )
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
    ):
        with pytest.raises(ValueError, match="index_topk=2048"):
            config.is_sparsity_driven_kv_offload_enabled(
                model_config=MODEL_CONFIG, use_mla_backend=True
            )


@pytest.mark.parametrize(
    "heads,width,expected_heads,valid",
    [
        (-1, 1, -1, False),
        (0, 1, 0, False),
        (64, 1, 64, True),
        (65, 1, 128, True),
        (127, 1, 128, True),
        (128, 1, 128, True),
        (129, 1, 129, False),
        (64, -1, 64, False),
        (64, 0, 64, True),
        (64, 14, 64, True),
        (64, 15, 64, False),
    ],
)
def test_budget_validates_normalized_query_geometry(
    monkeypatch, heads, width, expected_heads, valid
):
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
    monkeypatch.setattr(
        config, "get_spec", lambda: SimpleNamespace(speculative_num_draft_tokens=width)
    )
    monkeypatch.setattr(config, "get_eager_max_batch_size", lambda _: 4)
    model = SimpleNamespace(**vars(MODEL_CONFIG))
    model.hf_config = SimpleNamespace(**vars(MODEL_CONFIG.hf_config))
    model.hf_config.index_topk = 2048
    model.num_attention_heads = heads

    def make_plan():
        return config.get_sparse_kv_budget(
            model_config=model,
            request_rows=3,
            context_len=1024,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
            graph_batch_capacity=4,
        )

    with (
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
        envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(0),
        envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
    ):
        if valid:
            plan = make_plan()
            assert plan.copy_sfa_query_heads == expected_heads
            assert plan.max_query_width == (width or 1)
        else:
            with pytest.raises(ValueError, match="Invalid native LIM query geometry"):
                make_plan()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_draft_policy_is_private_to_fused_mode(monkeypatch, fused, draft):
    monkeypatch.setattr(config, "is_npu", lambda: True)
    monkeypatch.setattr(config, "is_deepseek_dsa", lambda _: True)
    monkeypatch.setattr(config, "attention_backends", lambda: ("ascend", "ascend"))
    monkeypatch.setattr(
        config,
        "get_schedule",
        lambda: SimpleNamespace(max_running_requests=4, page_size=128),
    )
    monkeypatch.setattr(
        config, "get_parallel", lambda: SimpleNamespace(pp_size=1, dcp_enabled=False)
    )
    monkeypatch.setattr(
        config, "get_spec", lambda: SimpleNamespace(speculative_algorithm="NONE")
    )
    monkeypatch.setattr(
        config, "get_disagg", lambda: SimpleNamespace(disaggregation_mode="null")
    )
    model = SimpleNamespace(**vars(MODEL_CONFIG))
    model.hf_config = SimpleNamespace(**vars(MODEL_CONFIG.hf_config))
    model.hf_config.index_topk = 2048 if fused else 512
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(fused),
    ):
        enabled = config.is_sparsity_driven_kv_offload_enabled(
            model_config=model, use_mla_backend=True, is_draft_worker=draft
        )
    assert enabled is not (fused and draft)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
