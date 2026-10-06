# SPDX-License-Identifier: Apache-2.0
# Adapted from FastVideo's fastvideo/models/audio/kandinsky6_audio_vae.py.
"""Checkpoint-compatible waveform-to-log-mel frontend.

Included for state-dict completeness; generation only decodes audio latents,
so waveform encoding has not been validated end-to-end."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelConverter(nn.Module):
    """Waveform -> log-mel-spectrogram STFT frontend. See module docstring:
    not exercised by the current (decode-only) inference path."""

    def __init__(
        self,
        *,
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        num_mels: int = 128,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: int = 0,
        fmax: int | None = 22050,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer(
            "hann_window", torch.hann_window(win_size), persistent=True
        )
        try:
            from librosa.filters import mel as librosa_mel_fn

            mel_basis = torch.from_numpy(
                librosa_mel_fn(
                    sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
                )
            ).float()
        except Exception:
            # generation does not use this buffer; the checkpoint supplies its real values
            mel_basis = torch.zeros(num_mels, n_fft // 2 + 1)
        self.register_buffer("mel_basis", mel_basis, persistent=True)

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1.0, max=1.0)
        pad = (self.n_fft - self.hop_size) // 2
        padded = F.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            padded,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        magnitude = torch.sqrt(spec.pow(2).sum(-1) + 1e-9).float()
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))
