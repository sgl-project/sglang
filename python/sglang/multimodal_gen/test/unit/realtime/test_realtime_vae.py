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


def test_causal_vae_decoding_stage_keeps_wan_decoder_cache(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _WanVAE:
        def __init__(self):
            self.config = SimpleNamespace(patch_size=None)
            self.clear_calls = 0
            self.decoder_first_chunk_flags = []
            self._feat_map = []
            self._conv_idx = [0]

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def clear_cache(self):
            self.clear_calls += 1
            self._feat_map = [None]
            self._conv_idx = [0]

        def post_quant_conv(self, latents):
            return latents

        def decoder(self, x, *, feat_cache, feat_idx, first_chunk=False):
            self.decoder_first_chunk_flags.append(first_chunk)
            if feat_cache[0] is None:
                feat_cache[0] = x.detach().clone()
            else:
                feat_cache[0] = torch.cat([feat_cache[0], x.detach().clone()], dim=2)
            feat_idx[0] += 1
            return x

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False

        def get_decode_scale_and_shift(self, device, dtype, vae):
            del device, dtype, vae
            return 1.0, None

        def preprocess_decoding(self, latents, server_args, vae=None):
            del server_args, vae
            return latents

        def post_decoding(self, frames, server_args):
            del server_args
            return frames

    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    vae = _WanVAE()
    vae.clear_cache()
    vae.clear_calls = 0
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.vae = vae
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    first = stage.decode_causal(
        torch.zeros(1, 1, 2, 1, 1),
        server_args,
        first_chunk=True,
    )
    second = stage.decode_causal(
        torch.ones(1, 1, 1, 1, 1),
        server_args,
        first_chunk=False,
    )

    assert tuple(first.shape) == (1, 1, 2, 1, 1)
    assert tuple(second.shape) == (1, 1, 1, 1, 1)
    assert vae.clear_calls == 0
    assert vae.decoder_first_chunk_flags == [True, False, False]
    assert tuple(vae._feat_map[0].shape) == (1, 1, 3, 1, 1)


def test_causal_vae_decoding_stage_prefers_native_causal_decode(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _NativeCausalVAE:
        def __init__(self):
            self.config = SimpleNamespace(patch_size=None)
            self.calls = []
            self._feat_map = [None]
            self._conv_idx = [0]

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def clear_cache(self):
            self.calls.append("clear_cache")

        def reset_causal_decode_state(self):
            self.calls.append("reset")

        def post_quant_conv(self, latents):
            self.calls.append("post_quant_conv")
            return latents

        def decoder(self, x, *, feat_cache, feat_idx, first_chunk=False):
            del x, feat_cache, feat_idx, first_chunk
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

    frames = stage.decode_causal(
        torch.zeros(1, 1, 1, 1, 1),
        server_args,
        first_chunk=True,
    )

    assert tuple(frames.shape) == (1, 1, 1, 1, 1)
    assert vae.calls == ["causal_decode"]
