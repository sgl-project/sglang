from transformers import AutoConfig

import sglang.srt.utils.hf_transformers.common  # noqa: F401
from sglang.srt.configs.telechat4 import TeleChat4Config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_telechat4_native_config_registration_and_legacy_rope_scaling():
    config = AutoConfig.for_model(
        "telechat4",
        architectures=["TeleChat4ForCausalLM"],
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 64,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        },
        rope_theta=10000,
    )

    assert isinstance(config, TeleChat4Config)
    assert config.architectures == ["TeleChat4ForCausalLM"]
    assert config.model_type == "telechat4"
    assert config.rope_parameters["factor"] == 64
    assert config.rope_parameters["rope_theta"] == 10000
    assert config.rope_parameters["rope_type"] == "rope"
