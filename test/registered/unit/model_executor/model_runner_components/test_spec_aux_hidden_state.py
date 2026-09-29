import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _map_muse_target_layer_ids,
    _resolve_dflash_draft_cell_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("target_model_type", "draft_architecture", "expected"),
    [
        ("muse_glimmer", "MuseGlimmerAssistantModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlash2DraftModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlashDraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "DFlash2DraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "MuseGlimmerAssistantModel", [1, 13, 25, 37, 49]),
    ],
)
def test_muse_target_layer_id_mapping(target_model_type, draft_architecture, expected):
    """The +1 belongs to Muse targets, which report layer outputs where the rest
    report layer inputs. The draft architecture alone does not earn it."""
    assert (
        _map_muse_target_layer_ids(
            target_hf_config=SimpleNamespace(model_type=target_model_type),
            draft_hf_config=SimpleNamespace(architectures=[draft_architecture]),
            layer_ids=[1, 13, 25, 37, 49],
        )
        == expected
    )


@pytest.mark.parametrize(
    ("attn_tp_size", "expected_bytes_per_token"),
    [
        (1, 20_480),
        (32, 1_280),
    ],
)
def test_dflash_draft_cell_size_uses_attention_tp(
    monkeypatch, attn_tp_size, expected_bytes_per_token
):
    """Draft KV geometry follows attention TP, which differs under DP attention."""
    draft_model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        head_dim=64,
        v_head_dim=64,
        get_num_kv_heads=lambda tp_size: max(1, 16 // tp_size),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_model",
        lambda: SimpleNamespace(kv_cache_dtype="auto"),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_draft_kv_cache_dtype="auto",
            speculative_draft_attention_backend=None,
        ),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=32, attn_tp_size=attn_tp_size),
    )
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
        lambda **_kwargs: (None, torch.bfloat16),
    )

    assert (
        _resolve_dflash_draft_cell_size(
            draft_model_config=draft_model_config,
            draft_num_layers=5,
        )
        == expected_bytes_per_token
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
