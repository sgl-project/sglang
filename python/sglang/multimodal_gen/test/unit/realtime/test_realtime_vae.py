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


def test_remote_vae_request_round_trips_prepared_latents(monkeypatch):
    from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
        payload_to_tensor,
    )

    monkeypatch.setenv("SGLANG_REALTIME_REMOTE_VAE_URL", "http://vae:31000")
    latents = torch.randn(1, 4, 3, 2, 2, dtype=torch.bfloat16)
    batch = SimpleNamespace(
        realtime_session_id="session",
        request_id="request",
        block_idx=7,
        realtime_event_id=9,
        extra={"realtime_is_final_chunk": True},
        width=1248,
        height=704,
        fps=24,
        realtime_output_format="raw",
        latents=latents,
    )

    request = CausalVaeDecodingStage._remote_vae_request(batch)

    assert request is not None
    assert request["block_idx"] == 7
    assert request["is_final_chunk"] is True
    assert request["realtime_output_format"] == "raw"
    assert request["response_transport"] == "http"
    torch.testing.assert_close(payload_to_tensor(request["latents"]), latents)


def test_raw_transport_batches_are_prejoined_and_split_at_wire_limit():
    from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
        RAW_RGB_FRAMES_PER_TRANSPORT_BATCH,
        build_raw_transport_batches,
    )

    frames = [bytes([index % 256]) for index in range(17)]
    batches = build_raw_transport_batches([frames])

    assert batches == [
        [
            {
                "num_frames": RAW_RGB_FRAMES_PER_TRANSPORT_BATCH,
                "payload": b"".join(frames[:RAW_RGB_FRAMES_PER_TRANSPORT_BATCH]),
            },
            {"num_frames": 1, "payload": frames[-1]},
        ]
    ]


def test_remote_vae_client_posts_msgpack_as_requests_data(monkeypatch):
    from sglang.multimodal_gen.runtime.remote.vae_decode_client import (
        RemoteVAEDecodeClient,
    )
    from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
        RAW_RGB_CONTENT_TYPE,
        SCHEMA_VERSION,
        packb,
        unpackb,
    )

    request = {"schema_version": SCHEMA_VERSION, "session_id": "session"}
    expected_frames = [[b"rgb"]]
    client = RemoteVAEDecodeClient("http://vae:31000", timeout=17)

    def fake_post(url, **kwargs):
        assert url == "http://vae:31000/decode"
        assert "content" not in kwargs
        assert unpackb(kwargs["data"]) == request
        assert kwargs["headers"] == {"content-type": "application/msgpack"}
        assert kwargs["timeout"] == 17
        return SimpleNamespace(
            content=packb(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "raw_frame_batches": expected_frames,
                    "raw_frame_content_type": RAW_RGB_CONTENT_TYPE,
                }
            ),
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client.session, "post", fake_post)

    result = client.decode(request)

    assert result.raw_frame_batches == expected_frames
    assert result.raw_transport_batches is None
    assert result.raw_frame_content_type == RAW_RGB_CONTENT_TYPE


def test_remote_vae_client_accepts_shared_memory_raw_transport(monkeypatch, tmp_path):
    from sglang.multimodal_gen.runtime.remote.vae_decode_client import (
        RemoteVAEDecodeClient,
    )
    from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
        RAW_RGB_CONTENT_TYPE,
        SCHEMA_VERSION,
        packb,
        store_raw_transport_batches_in_shared_memory,
    )

    transport = [[{"num_frames": 2, "payload": b"abcdef"}]]
    stored_transport = store_raw_transport_batches_in_shared_memory(
        transport, root=tmp_path
    )
    monkeypatch.setenv("SGLANG_REALTIME_VAE_SHM_DIR", str(tmp_path))
    client = RemoteVAEDecodeClient("http://vae:31000")
    monkeypatch.setattr(
        client.session,
        "post",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=packb(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "raw_transport_batches": stored_transport,
                    "raw_transport_storage": "shared_memory",
                    "raw_frame_content_type": RAW_RGB_CONTENT_TYPE,
                }
            ),
            raise_for_status=lambda: None,
        ),
    )

    result = client.decode({"schema_version": SCHEMA_VERSION})

    assert result.raw_frame_batches is None
    assert result.raw_transport_batches == transport
    assert not any(tmp_path.iterdir())


def test_loopback_remote_vae_selects_shared_memory_transport():
    from sglang.multimodal_gen.runtime.remote.vae_decode_client import (
        get_remote_vae_response_transport,
    )

    assert (
        get_remote_vae_response_transport("http://127.0.0.1:31000") == "shared_memory"
    )
    assert get_remote_vae_response_transport("http://vae:31000") == "http"


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


def test_causal_vae_decoding_stage_uses_streaming_taehv_decoder(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _NativeVAE:
        def __init__(self):
            self.config = SimpleNamespace(
                patch_size=2,
                z_dim=48,
                latents_mean=(0.0,) * 48,
                latents_std=(1.0,) * 48,
            )
            self.calls = []

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def causal_decode(self, latents):
            del latents
            self.calls.append("causal_decode")
            raise AssertionError("TAEHV decode should bypass native causal_decode")

    class _StreamingTAEHV:
        def __init__(self):
            self.reset_calls = 0
            self.inputs = []
            self.pending = []

        def reset(self):
            self.reset_calls += 1

        def decode(self, latents=None):
            if latents is not None:
                self.inputs.append(latents.detach().clone())
                self.pending.append(torch.full((1, 1, 3, 16, 16), 0.25))
            if self.pending:
                return self.pending.pop(0)
            return None

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False
        vae_config = SimpleNamespace(
            taehv_checkpoint_path="/opt/taehv/taew2_2.pth",
        )

        def get_decode_scale_and_shift(self, device, dtype, vae):
            del device, dtype, vae
            return 1.0, None

        def preprocess_decoding(self, latents, server_args, vae=None):
            del server_args, vae
            return latents + 2

    streaming = _StreamingTAEHV()
    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        realtime_vae.CausalVaeDecodingStage,
        "_load_taehv_model",
        lambda self, checkpoint_path, vae_dtype: object(),
    )
    monkeypatch.setattr(
        realtime_vae.CausalVaeDecodingStage,
        "_create_streaming_taehv_decoder",
        lambda self, taehv_model: streaming,
    )

    native_vae = _NativeVAE()
    decode_state = RealtimeVAEDecodeState()
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.vae = native_vae
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    frames = stage.decode_causal(
        torch.zeros(1, 48, 1, 4, 4),
        server_args,
        first_chunk=True,
        decode_state=decode_state,
    )

    assert native_vae.calls == []
    assert streaming.reset_calls == 1
    assert tuple(streaming.inputs[0].shape) == (1, 1, 48, 4, 4)
    assert torch.equal(streaming.inputs[0], torch.full((1, 1, 48, 4, 4), 2.0))
    assert tuple(frames.shape) == (1, 3, 1, 16, 16)
    assert torch.equal(frames, torch.full((1, 3, 1, 16, 16), 0.25))


def test_causal_vae_decoding_stage_preloads_taehv_model(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_precision="fp32",
            vae_config=SimpleNamespace(
                taehv_checkpoint_path="/opt/taehv/taew2_2.pth",
            ),
        )
    )
    load_calls = []
    taehv_model = object()

    def fake_decoding_stage_init(self, vae, pipeline=None, component_name="vae"):
        self.vae = vae
        self.pipeline = pipeline
        self.component_name = component_name
        self.server_args = server_args

    monkeypatch.setattr(
        realtime_vae.DecodingStage,
        "__init__",
        fake_decoding_stage_init,
    )
    monkeypatch.setattr(
        realtime_vae.CausalVaeDecodingStage,
        "_load_taehv_model",
        lambda self, checkpoint_path, vae_dtype: (
            load_calls.append((checkpoint_path, vae_dtype)) or taehv_model
        ),
        raising=False,
    )

    stage = realtime_vae.CausalVaeDecodingStage(vae=object())

    assert load_calls == [("/opt/taehv/taew2_2.pth", torch.float32)]
    assert stage._taehv_models == {
        ("/opt/taehv/taew2_2.pth", torch.float32): taehv_model,
    }


def test_causal_vae_decoding_stage_shares_taehv_weights_between_sessions(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
        vae as realtime_vae,
    )

    class _StreamingDecoder:
        def __init__(self, model):
            self.taehv = model

        def reset(self):
            pass

    stage = realtime_vae.CausalVaeDecodingStage.__new__(
        realtime_vae.CausalVaeDecodingStage
    )
    stage._taehv_models = {}
    load_calls = []
    taehv_model = object()
    monkeypatch.setattr(
        stage,
        "_load_taehv_model",
        lambda checkpoint_path, vae_dtype: (
            load_calls.append((checkpoint_path, vae_dtype)) or taehv_model
        ),
        raising=False,
    )
    monkeypatch.setattr(
        stage,
        "_create_streaming_taehv_decoder",
        lambda model: _StreamingDecoder(model),
        raising=False,
    )

    first_state = RealtimeVAEDecodeState()
    second_state = RealtimeVAEDecodeState()
    first = stage._get_or_create_streaming_taehv_decoder(
        first_state,
        "/opt/taehv/taew2_2.pth",
        torch.float32,
    )
    second = stage._get_or_create_streaming_taehv_decoder(
        second_state,
        "/opt/taehv/taew2_2.pth",
        torch.float32,
    )

    assert load_calls == [("/opt/taehv/taew2_2.pth", torch.float32)]
    assert first is not second
    assert first.taehv is taehv_model
    assert second.taehv is taehv_model


def test_causal_vae_decoding_stage_reads_parallel_decode_from_vae_config():
    class _Session:
        def __init__(self):
            self.state = None

        def get_or_create_state(self, state_cls):
            if self.state is None:
                self.state = state_cls()
            return self.state

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False
        vae_config = SimpleNamespace(
            use_parallel_decode=True,
            parallel_decode_mode="auto",
        )

        def post_decoding(self, frames, server_args):
            del server_args
            return frames

    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.load_model = lambda: None
    stage._get_causal_decode_reset_fn = lambda: None
    stage.decode_causal = lambda latents, server_args, **kwargs: latents + 1
    batch = SimpleNamespace(
        block_idx=0,
        latents=torch.zeros(1, 1, 1, 1, 1),
        session=_Session(),
        trajectory_timesteps=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
        metrics=None,
    )
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    output = stage.forward(batch, server_args)

    assert torch.equal(output.output, torch.ones(1, 1, 1, 1, 1))
