# SPDX-License-Identifier: Apache-2.0
"""GAP 2 regression: ``Kandinsky6AudioVAE`` must not re-renormalize an
already-loaded (final) checkpoint's audio-VAE Conv1d weights.

The diffusers reference calls ``MMAudioVAE.remove_weight_norm()`` exactly
once, inside ``MMAudioVAE.__init__`` -- BEFORE ``from_pretrained`` overwrites
those freshly-constructed (random) weights with the checkpoint's real,
already-final values, so a real checkpoint's own weights are never
renormalized. The pre-fix SGLang port instead triggered the same
RMS-renormalization lazily, on the first ``decode()`` call -- i.e. AFTER a
real checkpoint's weights were loaded -- corrupting them (measured by the
audit at ~9.7% decoded-waveform rel-L2 error against the diffusers
reference).
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.mmaudio_vae import (
    MMAudioVAE,
)

# A small, self-contained BigVGANV2 config (not a real checkpoint's) just so
# Kandinsky6AudioVAE.__init__ can build the whole module tree on CPU quickly;
# none of these tests call vocode()/wrapped_decode(), only the mel-level
# decode() where the audited bug lives.
_TINY_VOCODER_CONFIG = {
    "resblock": "1",
    "num_mels": 80,  # matches mode="16k"'s data_dim
    "upsample_rates": [2],
    "upsample_kernel_sizes": [4],
    "upsample_initial_channel": 8,
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 1, 1]],
    "activation": "snake",
    "snake_logscale": False,
}


def _tiny_audio_vae_config() -> Kandinsky6AudioVAEConfig:
    config = Kandinsky6AudioVAEConfig()
    config.arch_config.mode = "16k"
    config.arch_config.vocoder_config = dict(_TINY_VOCODER_CONFIG)
    return config


def test_mmaudio_vae_remove_weight_norm_is_idempotent():
    torch.manual_seed(0)
    vae = MMAudioVAE(mode="16k", need_encoder=False)
    vae.remove_weight_norm()
    latents = torch.randn(1, vae.embed_dim, 4)
    mel_once = vae.decode(latents.clone())

    # A second, redundant call must be a true no-op: it must not re-derive
    # (and thus change) already-normalized weights.
    vae.remove_weight_norm()
    mel_twice = vae.decode(latents.clone())

    assert torch.equal(mel_once, mel_twice)


def test_audio_vae_weights_are_not_renormalized_after_checkpoint_load():
    """Simulates VAELoader's real flow: construct, then load_state_dict with
    a checkpoint's own (already-final) weights -- decode() must use those
    weights as-is, matching the diffusers reference's "already normalized,
    no-op at inference" contract, not silently re-derive a different set of
    weights from them.
    """
    torch.manual_seed(0)
    config = _tiny_audio_vae_config()
    vae = Kandinsky6AudioVAE(config)
    assert vae.vae._weights_normalized is True

    # `load_state_dict(vae.state_dict())` mirrors VAELoader's real call
    # (`vae.load_state_dict(loaded, strict=strict_load)`) with the module's
    # own current weights standing in for "a checkpoint's own weights" --
    # exactly what happens when a real checkpoint's stored tensors are
    # loaded: no code path scales or otherwise disturbs them afterward.
    vae.load_state_dict(vae.state_dict())

    latents = torch.randn(1, vae.vae.embed_dim, 4)
    mel_a = vae.decode(latents.clone())

    # An explicit extra call (this wrapper's own public API) must also be a
    # no-op now, not a second renormalization of the loaded weights.
    vae.remove_weight_norm()
    mel_b = vae.decode(latents.clone())

    assert torch.equal(mel_a, mel_b)
