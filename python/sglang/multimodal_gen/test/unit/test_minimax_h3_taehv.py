# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
    MiniMaxH3DecodingStage,
)


class _FakeTAEHV:
    instances = []

    def __init__(self, *, checkpoint_path, arch_name):
        self.checkpoint_path = checkpoint_path
        self.arch_name = arch_name
        self.decode_inputs = []
        self.device = None
        self.dtype = None
        self.instances.append(self)

    def eval(self):
        return self

    def requires_grad_(self, value):
        assert value is False
        return self

    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self

    def decode_video(self, latents, *, parallel, show_progress_bar):
        self.decode_inputs.append(latents.clone())
        assert parallel is True
        assert show_progress_bar is False
        batch, _, _, height, width = latents.shape
        return torch.full((batch, 5, 3, height * 16, width * 16), 0.25)


def test_h3_taehv_uses_taeh3_architecture_and_diffusion_latents(monkeypatch):
    _FakeTAEHV.instances.clear()
    monkeypatch.setitem(sys.modules, "taehv", SimpleNamespace(TAEHV=_FakeTAEHV))
    stage = MiniMaxH3DecodingStage(video_vae=None, audio_vae=None)
    latents = torch.arange(1 * 24 * 7 * 2 * 3, dtype=torch.float32).reshape(
        1, 24, 7, 2, 3
    )

    frames = stage._decode_taehv_video(
        latents,
        checkpoint_path="/models/taeh3.pth",
        dtype=torch.float32,
    )

    decoder = _FakeTAEHV.instances[0]
    assert decoder.checkpoint_path == "/models/taeh3.pth"
    assert decoder.arch_name == "taeh3"
    assert frames.shape == (1, 3, 5, 32, 48)
    assert torch.equal(decoder.decode_inputs[0], latents.transpose(1, 2))

    cached = stage._get_taehv_decoder(
        "/models/taeh3.pth", device=latents.device, dtype=torch.float32
    )
    assert cached is decoder
    assert len(_FakeTAEHV.instances) == 1
