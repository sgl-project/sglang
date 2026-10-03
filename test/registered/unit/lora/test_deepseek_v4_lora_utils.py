from types import SimpleNamespace

from sglang.srt.lora.utils import (
    ATTN_TP_LORA_MODULE_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    get_hidden_dim,
    get_normalized_target_modules,
    matches_lora_target,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _NamedModel:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(name, object()) for name in self._names]


def _v4_config():
    return SimpleNamespace(
        model_type="deepseek_v4",
        hidden_size=4096,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        num_attention_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
    )


def test_deepseek_v4_attention_dimensions_and_parallelism():
    config = _v4_config()
    base_model = object()

    assert get_hidden_dim("wq_a", config, base_model, 0) == (4096, 1024)
    assert get_hidden_dim("wq_b", config, base_model, 0) == (1024, 32768)
    assert get_hidden_dim("wkv", config, base_model, 0) == (4096, 512)
    assert get_hidden_dim("wo_a", config, base_model, 0) == (4096, 8192)
    assert get_hidden_dim("wo_b", config, base_model, 0) == (8192, 4096)

    assert {"wq_a", "wkv"} <= set(REPLICATED_LINEAR_LORA_NAMES)
    assert "wo_b" in ROW_PARALLELISM_LINEAR_LORA_NAMES
    assert {"wq_b", "wo_a", "wo_b"} <= ATTN_TP_LORA_MODULE_NAMES


def test_deepseek_v4_shared_expert_dimensions():
    config = _v4_config()
    base_model = object()

    assert get_hidden_dim("gate_up_proj", config, base_model, 0) == (4096, 4096)
    assert get_hidden_dim("down_proj", config, base_model, 0) == (2048, 4096)


def test_wq_b_normalization_preserves_dsa_only_adapters():
    dsa_only = _NamedModel(["model.layers.0.self_attn.indexer.wq_b"])

    assert get_normalized_target_modules(["wq_b"]) == {"indexer.wq_b"}
    assert get_normalized_target_modules(["wq_b"], base_model=dsa_only) == {
        "indexer.wq_b"
    }


def test_wq_b_normalization_disambiguates_deepseek_v4_paths():
    v4 = _NamedModel(
        [
            "model.layers.0.self_attn.wq_b",
            "model.layers.0.self_attn.indexer.wq_b",
        ]
    )

    assert get_normalized_target_modules(["wq_b"], base_model=v4) == {"wq_b"}
    assert get_normalized_target_modules(["indexer.wq_b"], base_model=v4) == {
        "indexer.wq_b"
    }

    assert matches_lora_target("model.layers.0.self_attn.wq_b", {"wq_b"})
    assert not matches_lora_target("model.layers.0.self_attn.indexer.wq_b", {"wq_b"})
    assert matches_lora_target(
        "model.layers.0.self_attn.indexer.wq_b", {"indexer.wq_b"}
    )
