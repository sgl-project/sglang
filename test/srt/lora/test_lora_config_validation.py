import json

import pytest

from sglang.srt.lora.lora_config import LoRAConfig


def _write_adapter(tmp_path, payload):
    (tmp_path / "adapter_config.json").write_text(json.dumps(payload))

    return str(tmp_path)


def test_an_adapter_config_without_its_required_fields_is_named(tmp_path):
    path = _write_adapter(tmp_path, {"r": 8, "lora_alpha": 16})

    with pytest.raises(ValueError, match="missing required"):
        LoRAConfig(path=path)


def test_an_adapter_config_that_is_not_an_object_is_named(tmp_path):
    path = _write_adapter(tmp_path, ["target_modules"])

    with pytest.raises(ValueError, match="must contain a JSON object"):
        LoRAConfig(path=path)


def test_a_valid_adapter_config_still_loads(tmp_path):
    path = _write_adapter(
        tmp_path,
        {"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16},
    )
    config = LoRAConfig(path=path)

    assert config.target_modules == ["q_proj"]
    assert (config.r, config.lora_alpha) == (8, 16)


def test_from_dict_still_works():
    config = LoRAConfig.from_dict({"target_modules": ["q_proj"], "r": 4, "lora_alpha": 8})

    assert config.r == 4
