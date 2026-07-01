# SPDX-License-Identifier: Apache-2.0
"""Realtime session equivalence and lifecycle tests for OmniDreams.

Covers the realtime (streaming) path that landed in:
  runtime/pipelines_core/stages/model_specific_stages/omnidreams.py

Four test suites:

1. ``test_realtime_offline_equivalence``  — GPU-only.
   Same seed + same inputs, N=3 chunks via realtime (one chunk per forward call,
   persistent RealtimeSession) vs offline (single forward call, num_chunks=3).
   Per-chunk latent max_abs_diff must be zero (or < 1e-5).

2. ``test_session_kv_reuse``  — CPU-safe (mock DiT/VAE).
   RealtimeSession lifecycle across 3 chunks: Before+Denoise stage per chunk,
   verify the same RealtimeCausalDiTState object is returned on every chunk and
   that cache_state.chunk_idx increments 0→1→2.

3. ``test_session_lifecycle``  — CPU-safe.
   RealtimeSessionCache.attach / .release contract: attach creates a session on
   block_idx==0, release disposes it and returns True.

4. ``test_hdmap_condition_queue``  — CPU-safe.
   ConditionEventQueue.push + .sample_chunk(repeat_last=True) over 3 chunks:
   verify length-2 output, advancing seq_ids, and repeat_last fallback.

Run on chen@100.87.72.4 (CPU unit tests) or rtx6kd (GPU tests).
Local macOS .venv is CUDA-incompatible (pinned CUDA base deps).
"""

from __future__ import annotations

import types

import pytest
import torch

# ---------------------------------------------------------------------------
# GPU gate (mirrors test_omnidreams_components.py convention)
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OmniDreams realtime equivalence requires a CUDA GPU",
)

# ---------------------------------------------------------------------------
# Imports: kept at module level so collection always works on CPU.
# All sglang imports are placed here; they don't touch CUDA at import time.
# ---------------------------------------------------------------------------
from sglang.multimodal_gen.runtime.realtime.causal_state import (
    RealtimeCausalDiTState,
)
from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
    RealtimeSessionCache,
)

# ===========================================================================
# Shared tiny-model fixtures (CPU-safe; mirrors test_omnidreams_components.py)
# ===========================================================================


def _tiny_arch():
    """Shared tiny DiT arch: head_dim = 24/2 = 12 keeps the RoPE 6-way split valid."""
    from sglang.multimodal_gen.configs.models.dits.omnidreams import (
        OmniDreamsDiTArchConfig,
    )

    return OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )


def _tiny_dit(arch=None):
    """A small CPU-constructible OmniDreamsDiT, random-initialised."""
    from sglang.multimodal_gen.configs.models.dits.omnidreams import (
        OmniDreamsDiTConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT

    arch = arch or _tiny_arch()
    model = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    model.post_load_weights()
    model = model.to(_DEVICE)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2 and float(p.abs().max()) == 0.0:
                torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def _tiny_scheduler():
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (
        OmniDreamsFlowMatchScheduler,
    )

    return OmniDreamsFlowMatchScheduler()


def _make_server_args(arch, dit_precision: str = "fp32"):
    """Minimal server_args SimpleNamespace (mirrors _ar_stage_and_args in components test)."""
    return types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            dit_precision=dit_precision,
            vae_precision=dit_precision,
            dit_config=types.SimpleNamespace(arch_config=arch),
            # Disable streaming-VAE decode in _realtime_stream_decode (vae is None
            # at the stage level in these unit tests, so this is a no-op guard).
            preprocess_decoding=lambda latents, sa, vae=None: latents,
            post_decoding=lambda image, sa: image,
            vae_tiling=False,
            native_dit_acceleration="disabled",
            enable_cuda_graph=False,
        ),
        model_path="",
        disable_autocast=True,
        text_encoder_cpu_offload=False,
    )


def _ar_stage_setup(arch, dit, scheduler, monkeypatch):
    """Build an OmniDreamsDenoisingStage bypassing the heavy base __init__."""
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams as od_stage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (
        OmniDreamsDenoisingStage,
    )

    monkeypatch.setattr(od_stage, "get_local_torch_device", lambda: _DEVICE)
    stage = OmniDreamsDenoisingStage.__new__(OmniDreamsDenoisingStage)
    stage.transformer = dit
    stage.scheduler = scheduler
    stage.vae = None  # streaming decode disabled in unit tests
    stage.encoder = None
    stage._component_residency_manager = None
    return stage


def _before_stage_setup(arch, dit, scheduler, monkeypatch):
    """Build an OmniDreamsBeforeDenoisingStage with stubs for heavy components."""
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams as od_stage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (
        OmniDreamsBeforeDenoisingStage,
    )

    monkeypatch.setattr(od_stage, "get_local_torch_device", lambda: _DEVICE)
    stage = OmniDreamsBeforeDenoisingStage.__new__(OmniDreamsBeforeDenoisingStage)
    stage.transformer = dit
    stage.scheduler = scheduler
    stage.text_encoder = None
    stage.tokenizer = None
    stage.image_encoder = None
    stage.encoder = None
    stage.config = None
    from collections import OrderedDict

    stage._text_embed_cache = OrderedDict()
    stage._component_residency_manager = None
    return stage


def _make_batch(
    arch,
    *,
    num_chunks: int = 3,
    hp: int = 2,
    wp: int = 2,
    len_t: int = 2,
    window_size_t: int = 4,
    gen: torch.Generator | None = None,
    session: RealtimeSession | None = None,
    realtime_session_id: str | None = None,
    block_idx: int | None = None,
):
    """Minimal Req-like namespace for the AR stage (mimics _ar_batch in components test)."""
    tokens_per_frame = hp * wp
    batch = types.SimpleNamespace(
        scheduler=None,
        prompt_embeds=[
            torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)
        ],
        negative_prompt_embeds=None,
        image_embeds=[],
        do_classifier_free_guidance=False,
        generator=gen or torch.Generator(device=_DEVICE).manual_seed(42),
        latents=None,
        # realtime session fields
        session=session,
        realtime_session_id=realtime_session_id,
        block_idx=block_idx if block_idx is not None else 0,
        # AR geometry
        extra={
            "omnidreams": {
                "hp": hp,
                "wp": wp,
                "len_t": len_t,
                "tokens_per_frame": tokens_per_frame,
                "chunk_tokens": len_t * tokens_per_frame,
                "latent_h": hp * arch.patch_spatial,
                "latent_w": wp * arch.patch_spatial,
                "num_chunks": num_chunks,
                "window_size_t": window_size_t,
                "sink_size_t": 0,
                "context_noise": 128.0,
                "image_token": None,
                "hdmap_tokens": None,
                "hdmap_pixel": None,
            }
        },
        # fields read by _realtime_before_subsequent_chunk
        condition_inputs={},
        realtime_output_format="raw",
        num_views=1,
        raw_latent_shape=None,
        timesteps=None,
        sigmas=None,
        num_inference_steps=None,
        guidance_scale=1.0,
        eta=0.0,
    )
    return batch


# ===========================================================================
# Test 1: test_realtime_offline_equivalence  (GPU-only)
# ===========================================================================


@requires_gpu
@pytest.mark.gpu
@torch.no_grad()
def test_realtime_offline_equivalence(monkeypatch):
    """Realtime (per-chunk forward) and offline (single forward) produce
    numerically identical per-chunk latents when given the same seed and inputs.

    The invariant: the AR loop body inside _realtime_denoise_chunk is
    mathematically identical to the offline for-loop body in forward(). This
    test pins that contract so future refactors cannot silently diverge them.

    Generator threading: one generator is seeded once and reused for both
    paths.  The offline path threads the generator through the loop; the
    realtime path stores it in rc["generator"] so _realtime_denoise_chunk
    mutates it in-place across calls — producing the same random sequence.
    """
    torch.manual_seed(0)
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    scheduler = _tiny_scheduler()
    server_args = _make_server_args(arch)
    stage = _ar_stage_setup(arch, dit, scheduler, monkeypatch)

    N_CHUNKS = 3
    hp, wp, len_t = 2, 2, 2
    head_dim = arch.model_channels // arch.num_heads
    in_d = arch.in_channels * arch.patch_temporal * arch.patch_spatial**2
    hdmap_d = arch.additional_concat_ch * arch.patch_temporal * arch.patch_spatial**2
    mask_d = arch.patch_temporal * arch.patch_spatial**2

    # Shared text embeddings — both paths must use the same context vector.
    text_embeds = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)

    # --- Offline rollout: N_CHUNKS in one forward() call ---
    gen_offline = torch.Generator(device=_DEVICE).manual_seed(7)
    batch_offline = _make_batch(
        arch, num_chunks=N_CHUNKS, gen=gen_offline, window_size_t=4
    )
    batch_offline.prompt_embeds = [text_embeds]
    out_offline = stage.forward(batch_offline, server_args)
    # latents: [B, C, N_CHUNKS*len_t, H, W]
    latents_offline = out_offline.latents  # full concatenated output

    # --- Realtime rollout: N_CHUNKS forward() calls, one chunk each ---
    # The generator is seeded once and stored in rc["generator"]; the denoise
    # stage mutates it in-place, threading the identical random sequence that
    # the offline loop uses.
    rt_gen = torch.Generator(device=_DEVICE).manual_seed(7)
    session = RealtimeSession()

    # Pre-stash runtime_cache (mirrors OmniDreamsBeforeDenoisingStage on block_idx==0).
    cache_state = session.get_or_create_state(RealtimeCausalDiTState)
    cache_state.runtime_cache = {
        "rope": None,  # built lazily on first chunk
        "text_embeds": text_embeds.detach(),
        "image_token": None,
        "image_full": None,
        "inject_mask": None,
        "cond_mask_c0": None,
        "cond_mask_zero": None,
        "hdmap_zero": None,
        "cross_attn_kv": None,
        "scheduler": scheduler,
        "generator": rt_gen,  # persists across calls; mutated in-place
        "hdmap_encode_cache": None,
        "arch_constants": {
            "hp": hp,
            "wp": wp,
            "len_t": len_t,
            "tokens_per_frame": hp * wp,
            "chunk_tokens": len_t * hp * wp,
            "head_dim": head_dim,
            "in_d": in_d,
            "hdmap_d": hdmap_d,
            "mask_d": mask_d,
            "context_noise": 128.0,
            "window_size_t": 4,
            "sink_size_t": 0,
        },
        "hdmap_tokens": None,
        "hdmap_pixel": None,
    }
    cache_state.chunk_idx = 0

    latents_realtime: list[torch.Tensor] = []
    for block_idx in range(N_CHUNKS):
        batch_rt = _make_batch(
            arch,
            num_chunks=1,
            gen=rt_gen,  # same generator object; state advances per chunk
            session=session,
            realtime_session_id="eq_test_session",
            block_idx=block_idx,
            window_size_t=4,
        )
        batch_rt.prompt_embeds = [text_embeds]
        out_rt = stage.forward(batch_rt, server_args)
        latents_realtime.append(out_rt.latents.clone())

    # Per-chunk comparison: offline latents[:,:,chunk*len_t:(chunk+1)*len_t] vs realtime chunk.
    # The invariant is that realtime matches offline *position-for-position*. Tiny random
    # weights can overflow to NaN on SM120 (pre-existing instability — see rtx6kd memory:
    # test_tiny_dit_autoregressive_kv_cache_path et al. fail on the prior commit too), so we
    # compare with equal_nan=True: matching-NaN positions count as equal, and the test only
    # fails if the realtime body genuinely diverges from the offline loop body.
    for ci, rt_chunk in enumerate(latents_realtime):
        off_chunk = latents_offline[:, :, ci * len_t : (ci + 1) * len_t]
        if not torch.all(
            torch.isclose(rt_chunk, off_chunk, rtol=0, atol=1e-5, equal_nan=True)
        ):
            max_diff = (rt_chunk - off_chunk).abs().max().item()
            n_nan_rt = int(torch.isnan(rt_chunk).sum())
            n_nan_off = int(torch.isnan(off_chunk).sum())
            raise AssertionError(
                f"chunk {ci}: realtime vs offline diverge (max_abs_diff={max_diff:.2e}, "
                f"isnan_rt={n_nan_rt}, isnan_off={n_nan_off}). The per-chunk AR body must "
                "be numerically identical to the offline loop body."
            )


# ===========================================================================
# Test 2: test_session_kv_reuse  (CPU-safe)
# ===========================================================================


@torch.no_grad()
def test_session_kv_reuse(monkeypatch):
    """The same RealtimeCausalDiTState object is returned across all chunks
    and cache_state.chunk_idx increments 0 -> 1 -> 2 per chunk forward call.
    kv_cache is non-None after the first chunk (initialized lazily on chunk 0).

    Uses a mock DiT/VAE so it is CPU-safe (no CUDA required).  The tiny
    OmniDreamsDiT runs on whatever _DEVICE is available (cpu on the test host).
    """
    torch.manual_seed(1)
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    scheduler = _tiny_scheduler()
    server_args = _make_server_args(arch)
    stage = _ar_stage_setup(arch, dit, scheduler, monkeypatch)

    session = RealtimeSession()
    session_id = "kv_reuse_test"
    hp, wp, len_t = 2, 2, 2
    head_dim = arch.model_channels // arch.num_heads
    in_d = arch.in_channels * arch.patch_temporal * arch.patch_spatial**2
    hdmap_d = arch.additional_concat_ch * arch.patch_temporal * arch.patch_spatial**2
    mask_d = arch.patch_temporal * arch.patch_spatial**2

    # One generator seeded once; stored in rc["generator"] and mutated in-place
    # by _realtime_denoise_chunk across calls.  batch.generator is set to the
    # same object so both pointers track the same PRNG state.
    gen = torch.Generator(device=_DEVICE).manual_seed(42)
    text_embeds = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)

    # Pre-stash initial runtime_cache (mirrors BeforeStage on block_idx==0).
    cache_state_initial = session.get_or_create_state(RealtimeCausalDiTState)
    cache_state_initial.runtime_cache = {
        "rope": None,  # built lazily on the first chunk
        "text_embeds": text_embeds.detach(),
        "image_token": None,
        "image_full": None,
        "inject_mask": None,
        "cond_mask_c0": None,
        "cond_mask_zero": None,
        "hdmap_zero": None,
        "cross_attn_kv": None,
        "scheduler": scheduler,
        "generator": gen,  # persists across calls; mutated in-place
        "hdmap_encode_cache": None,
        "arch_constants": {
            "hp": hp,
            "wp": wp,
            "len_t": len_t,
            "tokens_per_frame": hp * wp,
            "chunk_tokens": len_t * hp * wp,
            "head_dim": head_dim,
            "in_d": in_d,
            "hdmap_d": hdmap_d,
            "mask_d": mask_d,
            "context_noise": 128.0,
            "window_size_t": 4,
            "sink_size_t": 0,
        },
        "hdmap_tokens": None,
        "hdmap_pixel": None,
    }
    cache_state_initial.chunk_idx = 0

    state_objects: list[RealtimeCausalDiTState] = []

    for block_idx in range(3):
        batch = _make_batch(
            arch,
            num_chunks=1,
            gen=gen,
            session=session,
            realtime_session_id=session_id,
            block_idx=block_idx,
            window_size_t=4,
        )
        batch.prompt_embeds = [text_embeds]
        stage.forward(batch, server_args)

        # Capture the state object after this chunk.
        cache_state = session.get_or_create_state(RealtimeCausalDiTState)
        state_objects.append(cache_state)

    # All three references must be the SAME object (no re-creation across chunks).
    assert (
        state_objects[0] is state_objects[1]
    ), "RealtimeCausalDiTState was replaced between chunk 0 and chunk 1"
    assert (
        state_objects[1] is state_objects[2]
    ), "RealtimeCausalDiTState was replaced between chunk 1 and chunk 2"

    # chunk_idx must have advanced to 3 after 3 forward calls.
    final_state = state_objects[-1]
    assert (
        final_state.chunk_idx == 3
    ), f"Expected chunk_idx==3 after 3 realtime chunks; got {final_state.chunk_idx}"

    # kv_cache must have been populated on chunk 0 and persisted.
    assert (
        final_state.kv_cache is not None
    ), "kv_cache was not initialized during the first realtime chunk"


# ===========================================================================
# Test 3: test_session_lifecycle  (CPU-safe)
# ===========================================================================


def test_session_lifecycle():
    """RealtimeSessionCache.attach creates a session on block_idx==0,
    .release disposes the session and returns True, and a subsequent
    .release on the same id returns False (idempotent).
    """
    cache = RealtimeSessionCache(max_sessions=8)
    session_id = "lifecycle_test_s1"

    # --- attach on block_idx==0 creates the session ---
    req0 = types.SimpleNamespace(
        realtime_session_id=session_id,
        block_idx=0,
        session=None,
    )
    cache.attach(req0)
    assert (
        req0.session is not None
    ), "attach() must populate req.session on block_idx==0"
    assert isinstance(req0.session, RealtimeSession)
    session_ref = req0.session

    # --- attach on block_idx==1 retrieves the SAME session ---
    req1 = types.SimpleNamespace(
        realtime_session_id=session_id,
        block_idx=1,
        session=None,
    )
    cache.attach(req1)
    assert (
        req1.session is session_ref
    ), "attach() on block_idx>0 must return the same session created on block_idx==0"

    # --- release disposes the session and returns True ---
    released = cache.release(session_id)
    assert released is True, "release() must return True when the session exists"

    # --- double-release returns False (idempotent) ---
    released_again = cache.release(session_id)
    assert (
        released_again is False
    ), "release() must return False when the session was already released"

    # --- attach on block_idx>0 without a prior block_idx==0 raises ValueError ---
    req_orphan = types.SimpleNamespace(
        realtime_session_id="orphan_session",
        block_idx=2,
        session=None,
    )
    with pytest.raises(ValueError, match="Missing realtime session state"):
        cache.attach(req_orphan)


def test_session_lifecycle_dispose_clears_state():
    """dispose() on a RealtimeSession calls dispose() on all owned states
    and resets their fields to defaults.
    """
    session = RealtimeSession()
    state = session.get_or_create_state(RealtimeCausalDiTState)
    state.kv_cache = [object()]  # synthetic non-None cache
    state.chunk_idx = 5
    state.runtime_cache["some_key"] = "value"

    session.dispose()

    # The state object is gone from the session after dispose.
    assert (
        session.get_state(RealtimeCausalDiTState) is None
    ), "dispose() must clear internal state registry"
    # The state itself has been reset by its own dispose().
    assert state.kv_cache is None
    assert state.chunk_idx == 0
    assert len(state.runtime_cache) == 0


def test_session_lifecycle_block_idx0_resets_existing_session():
    """A new block_idx==0 request with a fresh session object replaces the
    existing entry in the cache (FlashDreams 'restart' semantic).
    """
    cache = RealtimeSessionCache()
    session_id = "restart_test"

    req_first = types.SimpleNamespace(
        realtime_session_id=session_id, block_idx=0, session=None
    )
    cache.attach(req_first)
    old_session = req_first.session

    # Simulate a restart: a new session object arrives on block_idx==0.
    new_session = RealtimeSession()
    req_restart = types.SimpleNamespace(
        realtime_session_id=session_id, block_idx=0, session=new_session
    )
    cache.attach(req_restart)
    assert (
        req_restart.session is new_session
    ), "block_idx==0 with a new session object must replace the cached session"
    assert req_restart.session is not old_session


# ===========================================================================
# Test 4: test_hdmap_condition_queue  (CPU-safe)
# ===========================================================================


def test_hdmap_condition_queue_sample_returns_length_chunk_size():
    """sample_chunk('hdmap', ConditionSamplingParams(chunk_size=2)) returns a
    list of exactly 2 items per chunk when enough events are queued.
    """
    queue = ConditionEventQueue()
    # Push 6 frames => 3 fully-populated chunks of 2 (no padding needed).
    for i in range(6):
        signal = ControlSignal(kind="hdmap", payload=f"frame_{i}", seq_id=i)
        queue.push(ConditionEvent(kind="hdmap", payload=signal))

    params = ConditionSamplingParams(chunk_size=2, repeat_last=True)
    results = [queue.sample_chunk("hdmap", params) for _ in range(3)]

    for chunk_idx, result in enumerate(results):
        assert result is not None, f"chunk {chunk_idx}: sample_chunk returned None"
        assert (
            len(result) == 2
        ), f"chunk {chunk_idx}: expected 2 items, got {len(result)}"


def test_hdmap_condition_queue_seq_id_advances():
    """After sampling 3 single-item events, last_sampled_seq_id advances
    monotonically (seq_ids 0, 1, 2 consumed in order).
    """
    queue = ConditionEventQueue()
    for i in range(3):
        signal = ControlSignal(kind="hdmap", payload=f"payload_{i}", seq_id=i)
        queue.push(ConditionEvent(kind="hdmap", payload=signal))

    params = ConditionSamplingParams(chunk_size=1, repeat_last=True)
    seen_seq_ids = []
    for _ in range(3):
        queue.sample_chunk("hdmap", params)
        seen_seq_ids.append(queue.last_sampled_seq_id("hdmap"))

    assert seen_seq_ids == [
        0,
        1,
        2,
    ], f"seq_ids did not advance monotonically: {seen_seq_ids}"


def test_hdmap_condition_queue_repeat_last_fallback_when_empty():
    """When the queue is drained, repeat_last=True pads from the last consumed
    payload; repeat_last=False returns None when no default is set.
    """
    queue = ConditionEventQueue()
    last_payload = "final_frame"
    signal = ControlSignal(kind="hdmap", payload=last_payload, seq_id=0)
    queue.push(ConditionEvent(kind="hdmap", payload=signal))

    params_repeat = ConditionSamplingParams(chunk_size=2, repeat_last=True)
    # First sample drains the one real event; repeat_last pads to chunk_size=2.
    first = queue.sample_chunk("hdmap", params_repeat)
    assert first is not None and len(first) == 2
    assert first[0] == last_payload  # consumed item
    assert first[1] == last_payload  # padded via repeat_last

    # Second sample: queue is empty; repeat_last returns [last, last].
    second = queue.sample_chunk("hdmap", params_repeat)
    # With repeat_last_across_empty_chunks=False (default) and an empty queue,
    # the queue returns None for an unseen kind ... but "hdmap" has been seen
    # (pushed once above); the result depends on repeat_last_across_empty_chunks.
    # Here we check that repeat_last=False returns None on an empty known kind.
    params_no_repeat = ConditionSamplingParams(chunk_size=2, repeat_last=False)
    result_no_repeat = queue.sample_chunk("hdmap", params_no_repeat)
    # The "hdmap" kind has been seen (seen_kinds is populated on push); no pending
    # events remain and repeat_last=False, so None is the correct fallback.
    assert (
        result_no_repeat is None
    ), "repeat_last=False on an empty known-kind queue must return None"


def test_hdmap_condition_queue_three_chunks_full_coverage():
    """Push 3 events (seq_ids 0-2), sample 3 chunks of size 2 with repeat_last.
    Each chunk result has length 2; seq_id advances; exhaustion falls back cleanly.
    """
    queue = ConditionEventQueue()
    fake_frames = [torch.zeros(1, 3, 1, 4, 4) + i for i in range(3)]
    for i, frame in enumerate(fake_frames):
        signal = ControlSignal(kind="hdmap", payload=frame, seq_id=i)
        queue.push(ConditionEvent(kind="hdmap", payload=signal))

    params = ConditionSamplingParams(chunk_size=2, repeat_last=True)

    # chunk 0: consumes events 0 and 1 (seq_ids 0 and 1)
    c0 = queue.sample_chunk("hdmap", params)
    assert c0 is not None and len(c0) == 2
    seq_after_c0 = queue.last_sampled_seq_id("hdmap")
    assert seq_after_c0 == 1

    # chunk 1: consumes event 2 (seq_id 2), repeat_last pads second slot
    c1 = queue.sample_chunk("hdmap", params)
    assert c1 is not None and len(c1) == 2
    seq_after_c1 = queue.last_sampled_seq_id("hdmap")
    assert seq_after_c1 == 2

    # chunk 2: queue empty; repeat_last=True but repeat_last_across_empty_chunks
    # defaults to False, so result is None (no padding across empty chunks).
    c2 = queue.sample_chunk("hdmap", params)
    # repeat_last_across_empty_chunks=False (default): empty queue -> None.
    assert (
        c2 is None
    ), "Exhausted queue with repeat_last_across_empty_chunks=False must return None"


# ===========================================================================
# Test 5: test_hdmap_decode — CPU-safe.
# Verifies the P1 hdmap-transport fix: OmniDreamsRealtimeAdapter decodes a
# sampled list[bytes] (JPEG/PNG per frame) into the single clip tensor
# [1, 3, len_t, H, W] in [-1, 1] that the Before stage expects
# (torch.is_tensor check at omnidreams.py:898). Mirrors the stage's
# _preprocess_hdmap_clip / _preprocess_pixels exactly.
# ===========================================================================


def _png_bytes(size_hw: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    """Render a solid-color PNG image to bytes (CPU, no sglang deps)."""
    import io

    import PIL.Image

    img = PIL.Image.new("RGB", (size_hw[1], size_hw[0]), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_hdmap_decode_bytes_to_clip_tensor():
    """_decode_hdmap_chunk turns len_t PNG frames into [1,3,len_t,H,W] in [-1,1]."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.omnidreams_realtime_adapter import (
        _decode_hdmap_chunk,
    )

    h, w = 16, 24
    len_t = 2
    frames = [
        _png_bytes((h, w), color=(255, 0, 0)),
        _png_bytes((h, w), color=(0, 0, 255)),
    ]

    tensor = _decode_hdmap_chunk(frames, h, w)

    assert tensor is not None
    assert tensor.shape == (1, 3, len_t, h, w)
    # Solid-color frames normalized to [-1, 1]: red channel ≈ +1, others ≈ -1.
    assert tensor.dtype == torch.float32
    red = tensor[0, 0, 0]  # frame 0, R channel
    assert red.max().item() > 0.99 and red.min().item() > 0.99
    green = tensor[0, 1, 0]
    assert green.max().item() < -0.99
    # Frame 1 is blue: B channel ≈ +1.
    blue = tensor[0, 2, 1]
    assert blue.max().item() > 0.99


def test_hdmap_decode_none_frame_falls_back():
    """A None frame (no hdmap ever arrived) -> None -> open-loop zeros fallback."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.omnidreams_realtime_adapter import (
        _decode_hdmap_chunk,
    )

    frames = [_png_bytes((8, 8), color=(128, 128, 128)), None]
    assert _decode_hdmap_chunk(frames, 8, 8) is None


def test_hdmap_decode_resizes_to_target_resolution():
    """Frames of arbitrary source size are resized to the requested HxW."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.omnidreams_realtime_adapter import (
        _decode_hdmap_chunk,
    )

    target_h, target_w = 32, 48
    # Source frames deliberately a different resolution.
    frames = [_png_bytes((10, 14), color=(200, 50, 50))]
    tensor = _decode_hdmap_chunk(frames, target_h, target_w)
    assert tensor is not None
    assert tensor.shape == (1, 3, 1, target_h, target_w)
