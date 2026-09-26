"""A rope_scaling override must not silently drop the rope base.

Under transformers v5 the rope base lives inside ``rope_parameters`` (with
``rope_scaling`` as a deprecated alias whose setter replaces the dict
wholesale), and there is no top-level ``rope_theta`` attribute. So applying
``--json-model-override-args '{"rope_scaling": {...}}'`` through a plain
setattr drops the base, and ``get_rope_config`` silently falls back to 10000:
the model keeps generating fluent text with the wrong RoPE base (issue #41227:
Llama-3.2-3B-Instruct GSM8K first-200 161 -> 106).
"""

import json

from sglang.srt.utils.hf_transformers_utils import (
    get_config,
    get_hf_text_config,
    get_rope_config,
)

RESTATED_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _write_llama_checkpoint(tmp_path, rope_theta=500000.0):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 131072,
                "rope_theta": rope_theta,
                "rope_scaling": dict(RESTATED_SCALING),
            }
        )
    )
    return str(model_dir)


def _override_rope(tmp_path, scaling):
    model = _write_llama_checkpoint(tmp_path)
    config = get_config(
        model,
        trust_remote_code=False,
        model_override_args={"rope_scaling": scaling},
    )
    return get_rope_config(get_hf_text_config(config))


def test_rope_scaling_override_preserves_base(tmp_path):
    # Restating the checkpoint's own rope_scaling must be a no-op for the base.
    theta, params = _override_rope(tmp_path, dict(RESTATED_SCALING))
    assert theta == 500000.0, (
        f"rope base silently changed to {theta} by a rope_scaling override; "
        "get_rope_config fell back to 10000"
    )
    assert params["rope_type"] == "llama3"


def test_rope_scaling_override_explicit_theta_wins(tmp_path):
    # An override that names the base explicitly must win over the checkpoint.
    theta, _ = _override_rope(tmp_path, {**RESTATED_SCALING, "rope_theta": 1e7})
    assert theta == 1e7
