# SPDX-License-Identifier: Apache-2.0
"""Nomas realtime contracts: FIFO cache eviction, streaming decode, adapter events."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.nomas import NomasPipelineConfig
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    nomas_realtime_adapter as adapter_module,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.nomas_realtime_adapter import (
    NomasRealtimeAdapter,
    parse_segment_payload,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.models.dits.nomas_kv_cache import NomasCache
from sglang.multimodal_gen.runtime.models.vaes.nomas import NomasStreamingDecoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
    NomasSegmentCondition,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import (
    NomasDenoisingStage,
)
from sglang.multimodal_gen.runtime.realtime.nomas import (
    NOMAS_FLUSH_CONDITION,
    NOMAS_SEGMENT_CONDITION,
)
from sglang.multimodal_gen.runtime.realtime.states.nomas import NomasRealtimeState
from sglang.multimodal_gen.test.unit.test_nomas import (
    condition,
    tiny_decoder,
    tiny_model,
)

# The framework's parallel linears resolve the TP group at construction.
pytestmark = pytest.mark.usefixtures("single_process_model_parallel")

SHAPE = (2, 2, 2)  # 8 video tokens + 8 text tokens = 16 tokens per segment


def _session(model, cache_capacity=48, steps=2, **kwargs):
    return NomasLatentSession(
        model,
        cache_capacity=cache_capacity,
        seed=7,
        steps=steps,
        cache_noise_sigma=0.0,
        **kwargs,
    )


def _plain_condition():
    cond = condition()
    return NomasSegmentCondition(cond.text_states, cond.text_mask)


@torch.inference_mode()
def test_fifo_eviction_matches_upstream_cache_write_order():
    """Denoising sees up to max_cache_segments of history; the oldest segment is
    evicted before the cache forward, and surviving rows, masks and action
    tails shift left while the RoPE frame offset keeps counting."""
    model = tiny_model()
    cond = condition()
    keep_all = _session(model)
    evicting = _session(model, max_cache_segments=2)
    noise = [
        torch.randn(1, 32, *SHAPE, generator=torch.Generator().manual_seed(i))
        for i in range(3)
    ]
    kept = [keep_all.generate_segment(SHAPE, cond, initial_latents=n) for n in noise]
    outputs = []
    snapshots = []
    for n in noise:
        outputs.append(
            evicting.generate_segment(SHAPE, cond, initial_latents=n, evict=True)
        )
        snapshots.append(
            [
                {k: v.clone() for k, v in layer.items()}
                for layer in evicting.cache.layers
            ]
        )
    for a, b in zip(kept, outputs):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    cache = evicting.cache
    assert cache.segment_lengths == [16, 16] and cache.valid_length == 32
    assert cache.frame_offset == keep_all.cache.frame_offset == 6
    # Segment 2 was written with the same history in both sessions.
    for layer_e, layer_k in zip(cache.layers, keep_all.cache.layers):
        for key in ("k", "v"):
            torch.testing.assert_close(
                layer_e[key][:, :16], layer_k[key][:, 16:32], atol=0, rtol=0
            )
    # Segment 3 was written after eviction: its K/V equal a cache forward whose
    # history is only segment 2 but whose RoPE frame offset still counts the
    # evicted segment (4 frames, not 2).
    reference = model.allocate_cache(48)
    reference.commit(
        [{k: v[:, 16:32].clone() for k, v in layer.items()} for layer in snapshots[1]],
        cond.text_mask,
        8,
        4,
        cond.text_offsets[0],
    )
    _, entries = model.predict(
        outputs[2],
        torch.zeros(1),
        cond.text_states,
        cond.text_mask,
        cache=reference,
        camera_embedding=cond.camera_embedding,
        camera_has_mask=cond.camera_has_mask,
        text_offsets=cond.text_offsets,
        video_windows=cond.video_windows,
        hide_cached_action_tails=True,
        collect_cache=True,
    )
    for layer, entry in zip(cache.layers, entries):
        for key in ("k", "v"):
            torch.testing.assert_close(layer[key][:, 16:32], entry[key], atol=0, rtol=0)
    assert cache.action_tails == [(10, 16), (26, 32)]
    assert torch.equal(cache.visible[:, :32], keep_all.cache.visible[:, 16:48])
    assert not cache.visible[:, 32:].any()
    with pytest.raises(ValueError, match="exceeds the cache capacity"):
        evicting.generate_segment((2, 4, 8), cond, evict=True)
    evicting.reset()
    assert cache.segment_lengths == [] and cache.valid_length == 0


@torch.inference_mode()
def test_cache_grows_before_evicting_and_stops_at_max_capacity():
    """Upstream starts small and grows geometrically; FIFO only once the budget
    is reached. Growth must keep committed rows and visibility intact."""
    model = tiny_model()
    cond = condition()
    session = _session(
        model,
        cache_capacity=16,
        steps=1,
        max_cache_capacity=40,
        cache_growth_factor=1.5,
    )
    noise = torch.randn(1, 32, *SHAPE, generator=torch.Generator().manual_seed(3))
    session.generate_segment(SHAPE, cond, initial_latents=noise, evict=True)
    assert session.cache.capacity == 16 and session.cache.valid_length == 16
    before = [
        {k: v[:, :16].clone() for k, v in layer.items()}
        for layer in session.cache.layers
    ]
    session.generate_segment(SHAPE, cond, initial_latents=noise, evict=True)
    # 16 -> max(32, 24) = 32: grown, nothing evicted.
    assert session.cache.capacity == 32 and session.cache.segment_lengths == [16, 16]
    for layer, old in zip(session.cache.layers, before):
        for key in ("k", "v"):
            torch.testing.assert_close(layer[key][:, :16], old[key], atol=0, rtol=0)
    session.generate_segment(SHAPE, cond, initial_latents=noise, evict=True)
    # 32 -> min(48, 40) = 40 holds only two segments, so the oldest is evicted.
    assert session.cache.capacity == 40 and session.cache.segment_lengths == [16, 16]
    assert session.cache.action_tails == [(10, 16), (26, 32)]


@pytest.mark.parametrize("failure", ["cache_forward", "growth"])
@torch.inference_mode()
def test_failure_after_eviction_requires_reset(monkeypatch, failure):
    """Eviction rewrites history before the cache forward (and any growth); if
    either then fails, a retry must not silently use the shorter history."""
    model = tiny_model()
    plain = _plain_condition()
    session = _session(
        model, cache_capacity=16, steps=1, max_cache_segments=1, max_cache_capacity=64
    )
    session.generate_segment(SHAPE, plain, evict=True)
    rng = session.generator.get_state()
    if failure == "cache_forward":
        predict = model.predict

        def fail(*args, collect_cache=False, **kwargs):
            if collect_cache:
                raise RuntimeError("injected failure")
            return predict(*args, **kwargs)

        monkeypatch.setattr(model, "predict", fail)
        shape, error = SHAPE, RuntimeError
    else:

        def fail(*args):
            raise torch.OutOfMemoryError("injected failure")

        monkeypatch.setattr(NomasCache, "grow", fail)
        # 16 video + 8 text tokens: the only cached segment goes, then 16 -> 24.
        shape, error = (4, 2, 2), torch.OutOfMemoryError
    with pytest.raises(error, match="injected"):
        session.generate_segment(shape, plain, evict=True)
    assert session.cache.valid_length == 0 and session.cache.failed
    assert torch.equal(session.generator.get_state(), rng)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="reset the session"):
        session.generate_segment(shape, plain, evict=True)
    session.reset()
    session.generate_segment(shape, plain, evict=True)
    tokens = shape[0] * shape[1] * shape[2] + 8
    assert session.cache.segment_lengths == [tokens]
    if failure == "growth":
        # A segment cap that leaves too little room grows the cache instead.
        assert session.cache.capacity == 24


@torch.inference_mode()
def test_realtime_noise_bank_and_cache_noise_share_upstream_rng_order():
    """Upstream draws the bank, then per segment any bank growth and that
    segment's cache noise, all from one generator seeded with the seed."""
    model = tiny_model()
    plain = _plain_condition()
    config = NomasPipelineConfig(
        interactive_noise_max_seconds=1.0, interactive_noise_growth_seconds=1.0
    )
    server_args = SimpleNamespace(pipeline_config=config)
    stage = NomasDenoisingStage(model)
    state = NomasRealtimeState()
    reference = torch.Generator().manual_seed(7)
    # 24 frames at 24 fps -> 6 latent frames, drawn before any cache noise.
    expected_bank = torch.randn((1, 32, 6, 2, 2), generator=reference)
    for index, frames in enumerate((2, 2, 4)):
        batch = SimpleNamespace(
            latents=torch.zeros(1, 32, frames, 2, 2),
            seed=7,
            num_inference_steps=1,
            scheduler=SimpleNamespace(sigmas=None),
            realtime_session_id="s",
            fps=24,
        )
        if index == 0:
            stage._start_session(state, model, batch, server_args, plain)
            assert state.session.generator is state.noise_generator
        noise = stage._noise_slice(state, config, batch.latents, batch)
        if index == 2:
            # 4 + 4 > 6: grow by one second (6 latents) after two cache noises.
            expected_bank = torch.cat(
                (expected_bank, torch.randn((1, 32, 6, 2, 2), generator=reference)),
                dim=2,
            )
        start = state.generated_latents
        torch.testing.assert_close(
            noise, expected_bank[:, :, start : start + frames], atol=0, rtol=0
        )
        state.session.generate_segment(
            (frames, 2, 2), plain, initial_latents=noise, evict=True
        )
        state.generated_latents += frames
        torch.randn((1, 32, frames, 2, 2), generator=reference)  # cache noise
        assert torch.equal(state.noise_generator.get_state(), reference.get_state())
    torch.testing.assert_close(state.noise_bank, expected_bank, atol=0, rtol=0)


def stream_events(decoder, segments, frames):
    """Push each event onto one timeline, then finish; returns per-call outputs."""
    arch = decoder.config.arch_config
    timeline = NomasStreamingDecoder(
        arch.temporal_window_latents, arch.temporal_overlap_latents
    )

    def decode_fn(latents, frame_offset):
        with decoder.timeline_offset(frame_offset):
            return decoder.decode(latents).sample

    outputs = [timeline.push(decode_fn, s, f) for s, f in zip(segments, frames)]
    outputs.append(timeline.finish(decode_fn))
    return timeline, outputs


@pytest.mark.parametrize(
    "frames,counts",
    [
        ((17, 32, 9), (0, 12, 9, 37)),
        ((13, 33, 9, 1), (0, 12, 9, 0, 35)),
        ((40, 40, 40), (4, 40, 40, 36)),
        ((80, 3), (44, 0, 39)),
        ((5,), (0, 5)),
    ],
)
@torch.inference_mode()
def test_streaming_decode_emits_the_stable_prefix_of_the_full_decode(frames, counts):
    """Upstream stable-prefix streaming: a frame leaves only once no later
    window can change it, and the concatenation is the full-clip decode with
    each event's padded tail frames dropped."""
    decoder = tiny_decoder()
    gen = torch.Generator().manual_seed(11)
    segments = [
        torch.randn(1, 32, (f - 1) // 4 + 1, 2, 2, generator=gen) for f in frames
    ]
    timeline, outputs = stream_events(decoder, segments, frames)
    assert tuple(0 if o is None else o.shape[2] for o in outputs) == counts
    raw = decoder.decode(torch.cat(segments, dim=2)).sample
    starts = [0]
    for segment in segments:
        starts.append(starts[-1] + 4 * segment.shape[2])
    expected = torch.cat([raw[:, :, a : a + f] for a, f in zip(starts, frames)], dim=2)
    emitted = torch.cat([o for o in outputs if o is not None], dim=2)
    torch.testing.assert_close(emitted, expected, atol=0, rtol=0)
    # Only latents a later window can still read are retained.
    assert (
        timeline.latents.shape[2]
        <= 2 * decoder.config.arch_config.temporal_window_latents
    )
    with pytest.raises(RuntimeError, match="already finished"):
        timeline.finish(None)


def test_streaming_decode_rejects_inconsistent_event_lengths():
    timeline = NomasStreamingDecoder(10, 5)
    latent = torch.zeros(1, 32, 3, 1, 1)
    for frames in (8, 13):
        with pytest.raises(ValueError, match="do not match"):
            timeline.push(None, latent, frames)


def test_adapter_events_chunks_and_flush(monkeypatch):
    built = {}

    def fake_builder(
        request_id, *, chunk_inputs, num_frames, num_inference_steps, chunk_size, **_
    ):
        built.update(
            num_frames=num_frames, steps=num_inference_steps, chunk_size=chunk_size
        )
        return SimpleNamespace(prompt=chunk_inputs.prompt)

    monkeypatch.setattr(adapter_module, "build_realtime_sampling_params", fake_builder)

    def segment(payload, event_id=None):
        return RealtimeEvent(
            type="event", kind="segment", payload=payload, event_id=event_id
        )

    async def run():
        server_args = SimpleNamespace(pipeline_config=NomasPipelineConfig())
        adapter = get_realtime_model_adapter(server_args)
        assert isinstance(adapter, NomasRealtimeAdapter)
        session = GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest.model_validate(
            {
                "type": "init",
                "prompt": "A red car.",
                "num_frames": 17,
                "condition_inputs": {"prompt_format": "json"},
            }
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        state = adapter._state(session)
        assert state.prompt_format == "json" and len(state.pending) == 1
        assert adapter.get_chunk_size(session, None, None) == 5
        chunk = SimpleNamespace(request_id="r0", index=0)

        def step():
            size = adapter.get_chunk_size(session, None, chunk)
            inputs = adapter.sample_chunk_inputs(session, None, chunk, size)
            params = adapter.build_sampling_params(session, None, chunk, inputs, size)
            return size, inputs, params

        # 32 frames -> 8 latents -> (8-1)*4+1 = 29 canonical poses
        camera = [torch.eye(4).tolist()] * 29
        log = adapter.ingest_event(
            session,
            segment({"caption": "turn", "frames": 32, "norm_extrinsics": camera}, 3),
        )
        assert log.startswith("kind=segment") and len(state.pending) == 2
        for event, error in (
            (
                segment({"caption": "x", "frames": 32, "norm_extrinsics": camera[:5]}),
                "4x4 matrices",
            ),
            (
                RealtimeEvent(type="event", kind="prompt", payload="x"),
                "unsupported event kind",
            ),
        ):
            with pytest.raises(ValueError, match=error):
                adapter.ingest_event(session, event)

        await adapter.wait_for_next_chunk(session)
        _, _, params = step()
        assert built == {"num_frames": 17, "steps": 8, "chunk_size": 5}
        assert params.segments == [{"frames": 17, "caption": "A red car."}]
        assert params.prompt_format == "json" and params.norm_extrinsics is None
        assert params.generator_device == "cpu"
        assert adapter.get_realtime_event_id(session) is None
        session.generate_chunk_completed()

        size, _, params = step()
        assert size == 8 and params.norm_extrinsics == camera
        assert adapter.get_realtime_event_id(session) == 3
        session.generate_chunk_completed()

        # No event pending: the loop must block until one arrives.
        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not waiter.done()
        end = RealtimeEvent(type="event", kind="end")
        assert adapter.ingest_event(session, end) == "kind=end"
        await asyncio.wait_for(waiter, 1)
        with pytest.raises(ValueError, match="after end"):
            adapter.ingest_event(session, segment({"caption": "late", "frames": 4}))
        size, inputs, params = step()
        assert size == 1 and inputs.condition_inputs[NOMAS_FLUSH_CONDITION] is True
        assert inputs.condition_inputs[NOMAS_SEGMENT_CONDITION]["frames"] == 4
        assert params.prompt_format == "plain"
        assert request.max_chunks == 3
        session.generate_chunk_completed()
        assert session.reached_max_chunks()

    asyncio.run(run())


def test_segment_payload_validation():
    assert parse_segment_payload({"caption": "a", "frames": 1}, 9).latent_frames == 1
    assert (
        parse_segment_payload({"caption": "a", "frames": 40}, None).latent_frames == 10
    )
    for payload in (
        "a",
        {"caption": " ", "frames": 4},
        {"caption": "a", "frames": 0},
        {"caption": "a", "frames": 4.0},
    ):
        with pytest.raises(ValueError):
            parse_segment_payload(payload, None)
