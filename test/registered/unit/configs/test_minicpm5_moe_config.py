import json

from sglang.srt.configs.minicpm5_moe import MiniCPM5MoEConfig
from sglang.srt.hf_transformers import get_config, get_rope_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_loads_native_config_with_explicit_attention_head_dim(tmp_path):
    data = {
        "model_type": "minicpm5_moe",
        "architectures": ["MiniCPM5MoEForCausalLM"],
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "use_gated_attention": True,
        "rope_theta": 500000,
        "auto_map": {"AutoConfig": "missing_config.MiniCPM5MoEConfig"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, MiniCPM5MoEConfig)
    assert config.architectures == ["MiniCPM5MoEForCausalLM"]
    assert config.head_dim == 32
    assert config.num_key_value_heads == 2
    assert config.use_gated_attention is True
    assert get_rope_config(config)[0] == 500000


def test_rope_parameters_precedence_and_round_trip(tmp_path):
    legacy = {"type": "linear", "factor": 2.0}
    legacy_config = MiniCPM5MoEConfig(rope_theta=20000, rope_scaling=legacy)
    assert get_rope_config(legacy_config) == (
        20000,
        {"rope_theta": 20000, "rope_type": "linear", "factor": 2.0},
    )
    assert legacy == {"type": "linear", "factor": 2.0}

    explicit = {"rope_theta": 30000, "rope_type": "linear", "factor": 3.0}
    config = MiniCPM5MoEConfig(rope_theta=20000, rope_parameters=explicit)
    config.save_pretrained(tmp_path)
    reloaded = MiniCPM5MoEConfig.from_pretrained(tmp_path)

    assert get_rope_config(reloaded) == (30000, explicit)
