import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT


class _TestDiT(CachableDiT):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states


def test_dit_runtime_keeps_architecture_and_component_config_separate():
    component_config = DiTConfig(prefix="Wan")
    model = _TestDiT(config=component_config, hf_config={})

    assert model.config is component_config.arch_config
    assert model.prefix == "Wan"
    assert model._supports_cfg_cache
    assert model._spectrum_supports_cfg_cache
