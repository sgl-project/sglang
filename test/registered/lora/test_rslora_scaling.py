import math
from types import SimpleNamespace

import pytest

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig


def _make_config(r, lora_alpha, use_rslora=False):
    return LoRAConfig.from_dict({
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": ["q_proj"],
        "use_rslora": use_rslora,
    })


def _make_adapter(config):
    return LoRAAdapter(
        uid="test",
        config=config,
        base_hf_config=SimpleNamespace(num_hidden_layers=1),
        load_config=None,
        lora_backend=None,
    )


@pytest.mark.parametrize("r,lora_alpha", [(16, 32), (64, 128), (256, 256)])
class TestRSLoRAScaling:
    def test_standard_scaling(self, r, lora_alpha):
        config = _make_config(r, lora_alpha, use_rslora=False)
        adapter = _make_adapter(config)
        expected = lora_alpha / r
        assert adapter.scaling == pytest.approx(expected), (
            f"standard LoRA scaling: expected {expected}, got {adapter.scaling}"
        )

    def test_rslora_scaling(self, r, lora_alpha):
        config = _make_config(r, lora_alpha, use_rslora=True)
        adapter = _make_adapter(config)
        expected = lora_alpha / math.sqrt(r)
        assert adapter.scaling == pytest.approx(expected), (
            f"rsLoRA scaling: expected {expected}, got {adapter.scaling}"
        )

    def test_rslora_differs_from_standard(self, r, lora_alpha):
        standard = _make_adapter(_make_config(r, lora_alpha, use_rslora=False))
        rslora = _make_adapter(_make_config(r, lora_alpha, use_rslora=True))
        assert rslora.scaling > standard.scaling, (
            f"rsLoRA scaling ({rslora.scaling}) should be larger "
            f"than standard ({standard.scaling}) for r={r}"
        )


def test_use_rslora_default_false():
    config = LoRAConfig.from_dict({
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
    })
    assert config.use_rslora is False
