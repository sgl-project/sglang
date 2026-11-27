from types import SimpleNamespace

import torch
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


@torch.no_grad()
def test_flux2_encoder_cache_and_stage_dedup_follow_quality():
    from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import (
        AutoencoderKLFlux2,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
        ImageVAEEncodingStage,
    )

    gate = VaeFastPathGate()
    calls = []

    class VAE(nn.Module):
        encode = AutoencoderKLFlux2.encode
        supports_fast_encode = True
        use_slicing = False

        def __init__(self):
            super().__init__()
            self.encoder = nn.Identity()
            register_vae_fast_path_gate(self.encoder, gate)

        def _encode(self, x):
            calls.append(gate.enabled)
            return torch.cat([x + int(gate.enabled), x], dim=1)

    vae = VAE().eval()
    x = torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16)
    stage = object.__new__(ImageVAEEncodingStage)
    stage.vae = vae
    config = SimpleNamespace(
        vae_config=SimpleNamespace(encode_sample_mode=lambda: "argmax"),
        vae_precision="bf16",
        vae_tiling=False,
    )
    req = SimpleNamespace(
        condition_image=x, vae_image=x, image_path=None, height=8, width=8, num_frames=1
    )
    keys = []
    with ConditioningCache(1 << 16).scope():
        for quality in ("high", "lossless", "exact", "high"):
            req.quality = quality
            keys.append(
                stage.build_dedup_fingerprint(
                    req, SimpleNamespace(pipeline_config=config)
                )
            )
            kwargs = {"use_fast_path": True} if stage._use_fast_encode(req) else {}
            actual = vae.encode(x, **kwargs).mean
            torch.testing.assert_close(
                actual, torch.full_like(x, float(quality == "high")), rtol=0, atol=0
            )
            assert not gate.enabled
        assert calls == [True, False]
    assert keys[0] == keys[3] and keys[1] == keys[2] and keys[0] != keys[1]
    vae.encode(x.float(), use_fast_path=True)
    assert calls[-1] is False
