# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
)
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlSignalQueue,
    ControlSignalSamplingParams,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


# Default number of denoising steps for the 2-step distilled checkpoint.
OMNIDREAMS_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 2

# Temporal compression ratio for the Wan VAE: 1 latent frame = 4 pixel frames,
# except the very first chunk which has one extra "anchor" pixel frame.
_OMNIDREAMS_TEMPORAL_RATIO = 4


def _len_t(server_args: ServerArgs) -> int:
    """Return the AR chunk size (latent frames per block) from the pipeline config.

    Falls back to the OmniDreamsSamplingParams default of 2 when the pipeline
    config does not expose the field (e.g. during unit tests with a mock config).
    """
    try:
        return int(server_args.pipeline_config.dit_config.arch_config.len_t)
    except AttributeError:
        pass
    # Fallback: read from OmniDreamsSamplingParams default
    try:
        from sglang.multimodal_gen.configs.sample.omnidreams import (
            OmniDreamsSamplingParams,
        )

        return OmniDreamsSamplingParams.len_t
    except AttributeError:
        return 2


def _window_size_t(server_args: ServerArgs) -> int:
    """Return the KV-cache rolling window size (latent frames) from pipeline config."""
    try:
        return int(server_args.pipeline_config.dit_config.arch_config.window_size_t)
    except AttributeError:
        pass
    try:
        from sglang.multimodal_gen.configs.sample.omnidreams import (
            OmniDreamsSamplingParams,
        )

        return OmniDreamsSamplingParams.window_size_t
    except AttributeError:
        return 6


def _sink_size_t(server_args: ServerArgs) -> int:
    """Return the KV-cache permanent sink size (latent frames) from pipeline config."""
    try:
        return int(server_args.pipeline_config.dit_config.arch_config.sink_size_t)
    except AttributeError:
        pass
    try:
        from sglang.multimodal_gen.configs.sample.omnidreams import (
            OmniDreamsSamplingParams,
        )

        return OmniDreamsSamplingParams.sink_size_t
    except AttributeError:
        return 0


def _get_num_frames(block_idx: int, len_t: int) -> int:
    """Return pixel frame count for the given AR chunk index.

    Mirrors FlashDreams ``get_num_frames``:
      - chunk 0 includes the i2v anchor frame:  1 + (len_t - 1) * temporal_ratio
      - chunk n>0 is a pure generation block:   len_t * temporal_ratio

    With the default len_t=2 this yields chunk-0 → 5 frames, chunk-n → 8 frames.
    """
    if block_idx == 0:
        return 1 + (len_t - 1) * _OMNIDREAMS_TEMPORAL_RATIO
    return len_t * _OMNIDREAMS_TEMPORAL_RATIO


def _decode_hdmap_chunk(
    frames: list[Any],
    height: int,
    width: int,
) -> torch.Tensor | None:
    """Decode per-frame HDMap rasters into a single clip tensor.

    Mirrors ``OmniDreamsBeforeDenoisingStage._preprocess_hdmap_clip`` /
    ``_preprocess_pixels`` so the Before stage receives exactly the tensor
    shape/dtype/range it expects: ``[1, 3, T, H, W]`` in ``[-1, 1]`` on CPU.

    Each frame may be JPEG/PNG ``bytes``, a base64/data-URL/path ``str``, or a
    ``PIL.Image.Image``. ``None`` frames (sampled before any hdmap event has
    ever arrived) cause the whole chunk to fall back to ``None`` so the denoise
    stage uses its open-loop zeros / stashed-clip path.
    """
    per_frame: list[torch.Tensor] = []
    for f in frames:
        if f is None:
            return None
        try:
            image = load_image(f)
        except Exception:
            return None
        image = resize(image, height, width)
        x = numpy_to_pt(pil_to_numpy(image))  # [1,3,H,W] in [0,1]
        x = normalize(x)  # -> [-1,1]
        per_frame.append(x.unsqueeze(2))  # [1,3,1,H,W]
    if not per_frame:
        return None
    return torch.cat(per_frame, dim=2)  # [1,3,T,H,W]


class OmniDreamsRealtimeState:
    """Per-session mutable state for the OmniDreams realtime adapter.

    Holds a :class:`ControlSignalQueue` keyed on ``"prompt"`` (max 1 queued
    entry — latest wins) and ``"hdmap"`` (max 4 queued chunks of per-frame
    pixel data, ring-buffer semantics to bound memory under a lagging client).

    ``latest_sampled_event_id`` tracks the ``event_id`` of the most recently
    consumed condition event so the downstream stage can gate deferred KV
    writes on the correct client sequence number.
    """

    def __init__(self) -> None:
        self.events = ControlSignalQueue(max_events={"prompt": 1, "hdmap": 4})
        self.latest_sampled_event_id: int | None = None

    def clear(self) -> None:
        """Reset all per-session state (called by dispose)."""
        self.events.clear()
        self.latest_sampled_event_id = None

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        """Push an updated text prompt into the event queue (replaces any pending)."""
        self.events.push("prompt", prompt, event_id=event_id)

    def has_prompt(self) -> bool:
        return self.events.has_events("prompt")

    def sample_prompt(self) -> str:
        """Pop and return the latest queued prompt; updates latest_sampled_event_id."""
        prompt = self.events.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        self.latest_sampled_event_id = self.events.last_sampled_seq_id("prompt")
        return prompt

    def receive_hdmap(self, payload: Any, *, event_id: int | None = None) -> None:
        """Push per-chunk HDMap pixel data into the event queue.

        ``payload`` is the raw transport object from the client WebSocket
        message.  See :meth:`OmniDreamsRealtimeAdapter.ingest_event` for the
        expected encoding contract.
        """
        self.events.push("hdmap", payload, event_id=event_id)

    def sample_hdmap_chunk(self, chunk_size: int) -> list[Any] | None:
        """Sample ``chunk_size`` HDMap frames for the upcoming AR block.

        Uses ``repeat_last=True`` so that if fewer than ``chunk_size`` frames
        have arrived the last received frame is tiled forward (graceful
        degradation when the client lags).  Returns ``None`` only before the
        very first hdmap event has ever been received (no prior frame to
        repeat), in which case the denoising stage falls back to the
        pre-computed zeros HDMap.
        """
        result = self.events.sample_chunk(
            "hdmap",
            ControlSignalSamplingParams(
                chunk_size=chunk_size,
                repeat_last=True,
                default_item=None,
            ),
        )
        # Track the event id of the last sampled hdmap signal so we can
        # surface it as realtime_event_id on the batch.
        hdmap_seq_id = self.events.last_sampled_seq_id("hdmap")
        if hdmap_seq_id is not None:
            self.latest_sampled_event_id = hdmap_seq_id
        return result


class OmniDreamsRealtimeAdapter(BaseRealtimeModelAdapter):
    """Realtime model adapter for the NVIDIA OmniDreams world model.

    Wires the WebSocket ``/v1/realtime_video/generate`` session loop to the
    OmniDreams AR denoising stage.  Each chunk corresponds to one AR rollout
    step driven by ``block_idx``.

    Condition inputs
    ----------------
    ``prompt``
        Free-text scene description.  The most recent queued prompt is sampled
        at chunk 0; subsequent chunks inherit the running prompt unless the
        client pushes a new one mid-stream.

    ``hdmap``
        Per-chunk HD-map raster providing the spatial driving condition for the
        denoising stage.  See :meth:`ingest_event` for the assumed encoding.

    KV window
    ---------
    ``realtime_causal_kv_cache_num_frames`` ← ``window_size_t`` (default 6)
    ``realtime_causal_sink_size``           ← ``sink_size_t``    (default 0)

    Both are read from the pipeline config at runtime via :func:`_window_size_t`
    and :func:`_sink_size_t`; request-level overrides (if present) take priority.
    """

    name = "omnidreams"

    def __init__(self) -> None:
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    def create_state(self) -> OmniDreamsRealtimeState:
        return OmniDreamsRealtimeState()

    def _state(self, session: GenerateSession) -> OmniDreamsRealtimeState:
        state = session.adapter_state
        if not isinstance(state, OmniDreamsRealtimeState):
            raise TypeError("OmniDreams realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        """Save first_frame to a temp path and stash the path back on the request."""
        if request.first_frame is None:
            return

        server_args = get_global_server_args()
        if server_args.input_save_path is not None:
            uploads_dir = server_args.input_save_path
            os.makedirs(uploads_dir, exist_ok=True)
        else:
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
            uploads_dir = session.input_temp_dir

        target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
        image_path = await save_image_to_path(request.first_frame, target_path)
        request.first_frame = image_path

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        """Return immediately (open-loop).

        OmniDreams generates continuously without waiting for the client to
        deliver the next HDMap chunk.  When the hdmap queue is empty the
        denoising stage falls back to repeating the last received frame or
        zeros (handled by :meth:`OmniDreamsRealtimeState.sample_hdmap_chunk`).

        # TODO(closed-loop): await hdmap event before returning so that each
        # AR chunk is conditioned on a fresh client-supplied HDMap, turning
        # this into a true closed-loop system (see plan Step 4 / P1).
        """
        del session

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        """Accept ``prompt`` or ``hdmap`` events from the WebSocket client.

        HDMap encoding contract (closed-loop pixel push)
        -----------------------------------------------
        ``event.payload`` is expected to be one of:

        * ``bytes`` — JPEG- or PNG-encoded image for a single HDMap frame.
        * ``list[bytes]`` — one JPEG/PNG per latent frame within the chunk
          (``len_t`` items).  The ``ControlSignalQueue`` expands the list and
          samples exactly ``len_t`` frames in :meth:`prepare_next_request`.
        * ``str`` — base64-encoded JPEG/PNG (or a data-URL / file path), for
          JSON transport.

        In :meth:`prepare_next_request` each sampled frame is decoded (via
        ``vision_utils.load_image``) and preprocessed into a single clip tensor
        ``[1, 3, len_t, H, W]`` in ``[-1, 1]`` — the exact contract the Before
        denoising stage's ``_realtime_prepare_subsequent`` expects
        (``torch.is_tensor(hdmap_in)`` at ``omnidreams.py:898``).
        """
        state = self._state(session)
        if event.kind == "prompt":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("prompt event payload must be a non-empty string")
            state.receive_prompt(event.payload, event_id=event.event_id)
            return f"kind=prompt, prompt_len={len(event.payload)}"
        if event.kind == "hdmap":
            if event.payload is None:
                raise ValueError("hdmap event payload must not be None")
            state.receive_hdmap(event.payload, event_id=event.event_id)
            return f"kind=hdmap, event_id={event.event_id}"
        raise ValueError(f"unsupported event kind: {event.kind!r}")

    def _sample_chunk_inputs(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        """Sample prompt and HDMap condition for the upcoming AR block."""
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        # Prompt: chunk 0 always uses request.prompt; later chunks use a newly
        # queued prompt if one arrived, otherwise inherit the running prompt.
        if chunk.index == 0:
            prompt = request.prompt
        elif state.has_prompt():
            prompt = state.sample_prompt()
            request.prompt = prompt
        else:
            prompt = request.prompt

        # HDMap: sample chunk_size frames and decode them into a single clip
        # tensor ``[1, 3, chunk_size, H, W]`` in ``[-1, 1]`` on CPU, mirroring
        # the Before stage's ``_preprocess_hdmap_clip``. ``None`` (no hdmap has
        # ever arrived, or a frame failed to decode) leaves the condition unset
        # so the denoise stage falls back to its stashed clip / zeros path.
        hdmap_frames = state.sample_hdmap_chunk(chunk_size)
        condition_inputs: dict[str, Any] = {}
        if hdmap_frames is not None:
            w, h = _parse_size_or_raise(request.size) if request.size else (None, None)
            if w and h:
                hdmap_tensor = _decode_hdmap_chunk(hdmap_frames, h, w)
                if hdmap_tensor is not None:
                    condition_inputs["hdmap"] = hdmap_tensor

        return RealtimeChunkInputs(prompt=prompt, condition_inputs=condition_inputs)

    def _build_sampling_params(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
        server_args: ServerArgs,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        num_frames = _get_num_frames(chunk.index, chunk_size)

        return build_sampling_params(
            chunk.request_id,
            prompt=chunk_inputs.prompt,
            size=request.size,
            num_frames=num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=chunk.request_id,
            save_output=False,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=(
                request.num_inference_steps
                or OMNIDREAMS_REALTIME_DEFAULT_NUM_INFERENCE_STEPS
            ),
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            condition_inputs=chunk_inputs.condition_inputs,
            realtime_chunk_size=chunk_size,
        )

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        """Build a Req for the next AR chunk.

        Reads ``len_t`` from the pipeline config (falls back to 2), samples
        prompt + HDMap from the per-session state, and fills all realtime batch
        fields required by the OmniDreams denoising stage.
        """
        chunk_size = _len_t(server_args)
        chunk_inputs = self._sample_chunk_inputs(session, chunk, chunk_size)
        sampling_params = self._build_sampling_params(
            session,
            chunk,
            chunk_inputs,
            chunk_size,
            server_args,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        batch.session = session.realtime_session
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = chunk.index
        batch.realtime_event_id = self._state(session).latest_sampled_event_id

        if session.request is not None:
            req = session.request
            batch.realtime_output_format = req.realtime_output_format
            batch.realtime_preview_max_width = req.realtime_preview_max_width
            batch.realtime_output_pacing = bool(req.realtime_output_pacing)
            # KV window: prefer explicit request override, fall back to pipeline defaults.
            batch.realtime_causal_kv_cache_num_frames = (
                req.realtime_causal_kv_cache_num_frames
                if req.realtime_causal_kv_cache_num_frames is not None
                else _window_size_t(server_args)
            )
            batch.realtime_causal_sink_size = (
                req.realtime_causal_sink_size
                if req.realtime_causal_sink_size is not None
                else _sink_size_t(server_args)
            )

        return batch

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        return await self.output_adapter.send(ws, session, result, batch)

    def on_chunk_complete(self, session: GenerateSession, result: OutputBatch) -> None:
        del result
        session.generate_chunk_completed()

    def dispose(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, OmniDreamsRealtimeState):
            state.clear()
        self.output_adapter.reset()
