import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_loader.loader import _get_quantization_config
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=10,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


class _FakeQuarkConfig:
    def __init__(self):
        self.exclude_layers = ["model.layers.46"]

    def get_name(self):
        return "quark"

    def get_min_capability(self):
        return 75

    def get_supported_act_dtypes(self):
        return ["auto"]

    def apply_weight_name_mapper(self, mapper):
        self.exclude_layers = mapper.apply_list(self.exclude_layers)


def test_loader_mapper_hook_flows_into_quark_nextn_exclusion(monkeypatch):
    monkeypatch.setattr(
        Glm5NextForConditionalGenerationNextN, "packed_modules_mapping", {}
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader.get_model_architecture",
        lambda mc: (Glm5NextForConditionalGenerationNextN, None),
    )
    fake = _FakeQuarkConfig()
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader.get_quant_config",
        lambda *a, **k: fake,
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader.get_device_capability",
        lambda: (None, None),
    )

    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=46)),
        quantization="quark",
        dtype="auto",
        is_fp4_experts=False,
        dequant_fp4_to_fp8=False,
        nvfp4_moe_meta=None,
        packed_modules_mapping={},
    )
    quant_config = _get_quantization_config(model_config, SimpleNamespace())

    assert quant_config.exclude_layers == ["model.decoder"]

    resolved = Glm5NextForConditionalGenerationNextN.__new__(
        Glm5NextForConditionalGenerationNextN
    )._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=46), quant_config)
    assert resolved is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
