import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
    use_vae_fast_path,
)


def test_vae_fast_path_gate_is_decode_scoped_and_nestable():
    vae = nn.Module()
    gate = VaeFastPathGate()
    register_vae_fast_path_gate(vae, gate)

    with use_vae_fast_path(vae, True):
        assert gate.enabled
        with use_vae_fast_path(vae, False):
            assert not gate.enabled
        assert gate.enabled

    assert not gate.enabled
