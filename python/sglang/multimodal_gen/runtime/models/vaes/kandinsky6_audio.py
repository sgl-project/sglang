# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 audio VAE + vocoder: a thin nesting wrapper, not a new
architecture.

Mirrors FastVideo's ``fastvideo/models/audio/kandinsky6_audio_vae.py``
(reverse-engineered against a real checkpoint's audio_vae component,
``kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers``): every weight-tensor
shape in the safetensors file matches the ported ``MMAudioVAE`` (mel<->latent
codec) and ``BigVGANV2`` (mel->waveform vocoder), nested directly under
``vae.*`` / ``vocoder.*``, matching the current diffusers reference's
flattened ``MMAudioVAE`` (``vae``, ``vocoder``, ``mel_converter`` as direct
attributes -- no intermediate ``tod``/``native`` wrapper, which the diffusers
reference removed when it flattened its ``AutoEncoderModule`` indirection).
Only the module *nesting* is new here -- the layer implementations
(``kandinsky6_audio_vae/``) are ported unmodified.

Weight loading: sglang's generic ``VAELoader`` (component_loaders/vae_loader.py)
already routes ``component_name == "audio_vae"`` to this module via a plain
``nn.Module.load_state_dict`` over the whole tree -- no key remapping is
applied for VAE components, so this module's own attribute names MUST match
the checkpoint's safetensors key prefixes exactly (verified). No custom
``ComponentLoader`` subclass is needed: this file's module-level
``EntryClass = Kandinsky6AudioVAE`` is auto-discovered by
``ModelRegistry``'s AST scan (it only scans ``.py`` files directly under
``runtime/models/vaes/``, hence this file living here rather than inside the
``kandinsky6_audio_vae/`` subpackage), and is resolved via the checkpoint's
``audio_vae/config.json``'s ``_class_name`` field, which reads ``"MMAudioVAE"``
in the official checkpoints (see ``_aliases`` below).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.bigvgan import (
    BigVGANV2,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.mel_converter import (
    MelConverter,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.mmaudio_vae import (
    MMAudioVAE,
)


class Kandinsky6AudioVAE(nn.Module, LayerwiseOffloadableModuleMixin):
    """Nests the ported ``MMAudioVAE`` + ``BigVGANV2`` (plus a
    state-dict-completeness ``MelConverter``) to match the checkpoint's exact
    parameter names, and exposes the diffusers reference's ``wrapped_decode``
    (mel-VAE decode -> vocode in one call).

    Public API (consumed by ``Kandinsky6AudioDecodingStage``):
      - ``decode(latents: [B, embed_dim, A]) -> mel [B, num_mels, T]``
      - ``vocode(mel: [B, num_mels, T]) -> waveform [B, 1, samples]``
      - ``wrapped_decode(latents) -> waveform`` (decode + vocode in one call)
      - ``encode_audio(waveform) -> DiagonalGaussianDistribution`` (best-effort,
        unverified -- see module docstring on ``MelConverter``)
      - ``wrapped_encode(waveform) -> mean latent`` (matches the diffusers
        reference's ``MMAudioVAE.wrapped_encode``)
      - ``scaling_factor: float`` / ``mean_value: float`` plain attributes,
        for the pipeline's audio-latent denormalization
        (``latents / scaling_factor + mean_value``) before calling
        ``wrapped_decode``.
    """

    # The real checkpoint's audio_vae/config.json uses "MMAudioVAE" as its
    # _class_name (matching the diffusers reference, which folds the
    # vocoder and mel converter into the same top-level class); resolve
    # that name to this wrapper too.
    _aliases = ["MMAudioVAE"]

    # Decode-only small model relative to the DiT; no benefit from DiT-group
    # offload tuning knobs. Mirrors MiniMaxH3AudioVAE's choice.
    layerwise_offload_dit_group_enabled = False

    def __init__(self, config: Kandinsky6AudioVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.scaling_factor = float(arch.scaling_factor)
        self.mean_value = float(arch.mean_value)
        # API-parity metadata mirroring the diffusers reference's
        # MMAudioVAE.downsample_factor; NOT consumed by decode()/vocode()
        # themselves -- audio-latent-frame-count math lives in the pipeline
        # config's own audio_downsample_factor field, not here.
        self.downsample_factor = 1024

        self.mel_converter = MelConverter(
            sampling_rate=44100,
            n_fft=2048,
            num_mels=128,
            hop_size=512,
            win_size=2048,
            fmin=0,
            fmax=22050,
        )
        # The real checkpoint's audio_vae weights include the encoder half
        # regardless of what its config.json's own need_vae_encoder says
        # (verified against the actual safetensors keys) -- always build it,
        # matching the diffusers reference's VAE, which builds both encoder
        # and decoder unconditionally.
        self.vae = MMAudioVAE(mode=arch.mode, need_encoder=True)
        # Matches the diffusers reference's MMAudioVAE.__init__, which calls
        # `self.vae.remove_weight_norm()` right here -- BEFORE the loader's
        # `load_state_dict` overwrites these freshly-constructed (random)
        # Conv1d weights with the checkpoint's real, already-final values.
        # Applying it at construction time means a real checkpoint's own
        # weights are never renormalized: only the about-to-be-discarded
        # random init is, and `remove_weight_norm()`'s own idempotency guard
        # keeps any later call (e.g. this wrapper's public
        # `remove_weight_norm()`) a no-op.
        self.vae.remove_weight_norm()
        self.vocoder = BigVGANV2(dict(arch.vocoder_config))
        # BigVGANV2's constructor applies PyTorch's weight_norm
        # parametrization (state-dict keys `parametrizations.weight.
        # original{0,1}`), but real checkpoints are saved with it already
        # removed (plain `.weight`). Strip it here, before the loader's
        # load_state_dict, so the module structure matches what's on disk.
        # remove_weight_norm() is idempotent (catches the ValueError from a
        # double-removal), so this is safe even if arch.vocoder_config's own
        # "weight_norm_removed" flag already triggered it once inside
        # BigVGANV2.__init__.
        #
        # This is DIFFERENT from MMAudioVAE.remove_weight_norm() above, which
        # is a custom RMS weight renormalization applied to this module's own
        # freshly-constructed (pre-load) Conv1d weights -- not PyTorch
        # parametrization removal. Do not conflate the two.
        self.vocoder.remove_weight_norm()

        # resolution containers are not called; hooks belong on their blocks
        self.layer_names = (
            [
                f"{path}.{index}.{group}"
                for path, levels in (
                    ("vae.encoder.down", self.vae.encoder.down),
                    ("vae.decoder.up", self.vae.decoder.up),
                )
                for index in range(len(levels))
                for group in ("block", "attn")
            ]
            + [f"vocoder.ups.{index}" for index in range(len(self.vocoder.ups))]
            + ["vocoder.resblocks"]
        )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: [B, embed_dim, A] (1D-conv channel-first) -> mel: [B, num_mels, T]."""
        # No renormalization here: `self.vae` was already marked normalized
        # in __init__, before the loader's `load_state_dict` replaced its
        # weights with the checkpoint's real, already-final values -- so the
        # checkpoint's own weights reach this call untouched, matching the
        # diffusers reference's "already normalized, no-op at inference"
        # contract.
        return self.vae.decode(latents, unnormalize_output=True)

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, num_mels, T] -> waveform: [B, 1, samples]."""
        # MMAudioVAE.decode(..., unnormalize_output=True) denormalizes
        # against float32 data_mean/data_std buffers, so `mel` comes back
        # float32 even when the input latents were bf16. Cast to the
        # vocoder's own loaded weight dtype here -- same "cast to the
        # module's own weight dtype directly" pattern already used for the
        # initial latents cast in Kandinsky6AudioDecodingStage.
        vocoder_dtype = next(self.vocoder.parameters()).dtype
        return self.vocoder(mel.to(dtype=vocoder_dtype))

    def wrapped_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents -> waveform in one call, matching the diffusers
        reference's MMAudioVAE.wrapped_decode."""
        return self.vocode(self.decode(latents))

    def encode_audio(self, waveform: torch.Tensor):
        """waveform -> mel -> VAE posterior. See MelConverter's module
        docstring: not exercised by / verified against the current
        (decode-only) inference path."""
        mel = self.mel_converter(waveform)
        return self.vae.encode(mel)

    def wrapped_encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform -> mean audio latent in one call, matching the
        diffusers reference's MMAudioVAE.wrapped_encode."""
        return self.encode_audio(waveform).mean

    def remove_weight_norm(self) -> Kandinsky6AudioVAE:
        """Explicit/manual trigger for MMAudioVAE's custom RMS weight
        renormalization. Already applied once in __init__ (before loading),
        so this is a no-op in the normal loaded-checkpoint path --
        `MMAudioVAE.remove_weight_norm()`'s own idempotency guard makes a
        redundant call here harmless rather than re-mutating already-final
        weights. Kept for API parity / explicit external use (e.g. a
        freshly-constructed, not-yet-loaded VAE). The vocoder's (unrelated,
        PyTorch-parametrization) weight_norm was already stripped at
        construction time too -- see __init__."""
        self.vae.remove_weight_norm()
        return self


EntryClass = Kandinsky6AudioVAE

__all__ = ["Kandinsky6AudioVAE"]
