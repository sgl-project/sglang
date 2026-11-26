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
    mla_kv_head_num=1,
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
        lambda: SimpleNamespace(
            page_size=128,
            prefill_max_requests=None,
            max_running_requests=2,
            max_total_tokens=None,
        ),
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
    model = SimpleNamespace(**vars(MODEL_CONFIG))
    model.hf_config = SimpleNamespace(**vars(MODEL_CONFIG.hf_config))
    model.hf_config.index_topk = 2048
    model.num_attention_heads = heads

    def make_plan():
        return config.get_sparse_kv_budget(
            model_config=model,
            available_bytes=1 << 30,
            index_k_and_draft_bytes_per_token=2 * 256,
            context_len=1024,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
        )

    with (
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
        envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(0),
    ):
        if valid:
            plan = make_plan()
            assert plan.copy_sfa_query_heads == expected_heads
            assert plan.max_query_width == (width or 1)
        else:
            with pytest.raises(ValueError, match="Invalid native LIM query geometry"):
                make_plan()


@pytest.mark.parametrize(
    "offload,fused,draft,expected_mode",
    [
        (False, False, False, None),
        (False, True, False, None),
        (True, False, False, "legacy"),
        (True, False, True, "legacy"),
        (True, True, False, "fused"),
        (True, True, True, None),
    ],
)
def test_sparse_kv_offload_mode(monkeypatch, offload, fused, draft, expected_mode):
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
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(offload),
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(fused),
    ):
        args = dict(model_config=MODEL_CONFIG, use_mla_backend=True)
        assert (
            config.get_sparse_kv_offload_mode(**args, is_draft_worker=draft)
            == expected_mode
        )
        assert config.is_fused_sparse_kv_offload_enabled(
            **args, is_draft_worker=draft
        ) == (expected_mode == "fused")
        assert config.is_sparsity_driven_kv_offload_enabled(**args) == offload


def test_fused_draft_bypasses_target_validation(monkeypatch):
    def fail_if_target_validation_runs():
        raise AssertionError("draft worker must bypass target validation")

    monkeypatch.setattr(config, "attention_backends", fail_if_target_validation_runs)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
    ):
        assert not config.is_fused_sparse_kv_offload_enabled(
            model_config=None, use_mla_backend=True, is_draft_worker=True
        )


def test_fused_draft_keeps_full_kv_cell_size(monkeypatch):
    monkeypatch.setattr(config, "is_npu", lambda: True)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
    ):
        assert (
            config.get_sparsity_driven_kv_offload_cell_size(
                model_config=None,
                use_mla_backend=True,
                num_layers=1,
                element_size=2,
                is_draft_worker=True,
            )
            is None
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
