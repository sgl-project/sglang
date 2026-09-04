# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

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
    assert state.taehv_streaming_decoder is None
    assert state.taehv_output_queue == []


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


def test_causal_vae_decoding_stage_can_use_streaming_taehv(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _TAEHV:
        t_upscale = 4
        frames_to_trim = 3
        init_count = 0

        def __init__(self, checkpoint_path):
            type(self).init_count += 1
            self.checkpoint_path = checkpoint_path

        def to(self, device=None, dtype=None):
            self.device = device
            self.dtype = dtype
            return self

        def eval(self):
            self.training = False

    class _StreamingTAEHV:
        def __init__(self, taehv):
            self.taehv = taehv
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def decode(self, latent=None):
            if latent is None:
                return None
            batch, _, channels, height, width = latent.shape
            value = float(latent.flatten()[0].item())
            return torch.full(
                (batch, self.taehv.t_upscale, channels, height, width),
                value,
                dtype=latent.dtype,
            )

    fake_taehv = ModuleType("taehv")
    fake_taehv.TAEHV = _TAEHV
    fake_taehv.StreamingTAEHV = _StreamingTAEHV
    monkeypatch.setitem(sys.modules, "taehv", fake_taehv)
    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_config = SimpleNamespace(taehv_checkpoint_path="/tmp/taehv-test.pth")

        def post_decoding(self, frames, server_args):
            del server_args
            return frames

    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    state = RealtimeVAEDecodeState()
    server_args = SimpleNamespace(pipeline_config=_PipelineConfig())

    first = stage.decode_taehv_streaming(
        torch.arange(6, dtype=torch.float32).reshape(1, 2, 3, 1, 1),
        server_args,
        state,
        first_chunk=True,
    )
    second = stage.decode_taehv_streaming(
        torch.arange(6, 12, dtype=torch.float32).reshape(1, 2, 3, 1, 1),
        server_args,
        state,
        first_chunk=False,
    )

    assert tuple(first.shape) == (1, 2, 9, 1, 1)
    assert tuple(second.shape) == (1, 2, 12, 1, 1)
    assert state.taehv_streaming_decoder.reset_calls == 1
    assert _TAEHV.init_count == 1
    assert len(state.taehv_output_queue) == 1
    assert tuple(state.taehv_output_queue[0].shape) == (1, 3, 2, 1, 1)
