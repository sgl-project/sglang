import pytest
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    approximate_decode_enabled,
    register_vae_fast_path_gate,
    use_vae_decode_quality,
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


def test_decode_quality_restores_both_tiers_after_nested_error():
    vae = nn.Module()
    gate = VaeFastPathGate()
    register_vae_fast_path_gate(vae, gate)
    with use_vae_decode_quality(vae, "high"):
        assert gate.enabled and approximate_decode_enabled()
        with pytest.raises(RuntimeError, match="decode failed"):
            with use_vae_decode_quality(vae, "lossless"):
                assert gate.enabled and not approximate_decode_enabled()
                raise RuntimeError("decode failed")
        assert gate.enabled and approximate_decode_enabled()
    assert not gate.enabled and not approximate_decode_enabled()
