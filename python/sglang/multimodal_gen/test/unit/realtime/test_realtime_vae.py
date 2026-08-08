# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
    RealtimeVAEDecodeState,
)


def test_realtime_vae_decode_state_clears_model_cache_on_dispose():
    calls = []
    state = RealtimeVAEDecodeState()
    state.reset_causal_decode_state = lambda: calls.append("reset")

    state.dispose()

    assert calls == ["reset"]
    assert state.reset_causal_decode_state is None


def test_causal_vae_decoding_stage_prefers_native_causal_decode(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _NativeCausalVAE:
        def __init__(self):
            self.config = SimpleNamespace(patch_size=None)
            self.calls = []

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def reset_causal_decode_state(self):
            self.calls.append("reset")

        def post_quant_conv(self, latents):
            self.calls.append("post_quant_conv")
            return latents

        def decoder(self, x):
            del x
            self.calls.append("decoder")

        def causal_decode(self, latents):
            self.calls.append("causal_decode")
            return latents

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False

        def get_decode_scale_and_shift(self, device, dtype, vae):
            del device, dtype, vae
            return 1.0, None

        def preprocess_decoding(self, latents, server_args, vae=None):
            del server_args, vae
            return latents

    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    vae = _NativeCausalVAE()
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.vae = vae
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    frames = stage.decode_causal(torch.zeros(1, 1, 1, 1, 1), server_args)

    assert tuple(frames.shape) == (1, 1, 1, 1, 1)
    assert vae.calls == ["causal_decode"]
