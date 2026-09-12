# SPDX-License-Identifier: Apache-2.0
"""Realtime (tick) adapter for Cosmos-Dreams action checkpoints.

Each chunk of the ``/v1/realtime_video`` session is one autoregressive block
of ``chunk_size`` latent frames driven by one raw action row per pixel step.
Clients send ``action`` events carrying either one row (held until the next
event), a list of rows (a script consumed in order), or the generic state-mode
payload ``{"mode": "state", "transitions": [...]}``. Rows are in the
embodiment's raw units (camera: metres and rot6d columns). With no action
received, the idle action (zero translation, identity rotation) is used.
"""

from __future__ import annotations

import functools
import math
import os
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CosmosDreamsManifest,
    EmbodimentContract,
    load_cosmos_dreams_manifest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams_realtime import (
    ACTION_ROWS_CONDITION,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlScriptQueue,
    ControlStateQueue,
    ControlStateTransition,
    parse_control_event_payload,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

ACTION_EVENT_KIND = "action"
COSMOS_DREAMS_REALTIME_DEFAULT_FPS = 24
# Optional init fields forwarded to the Cosmos-Dreams sampling params.
COSMOS_DREAMS_REALTIME_EXTRA_FIELDS = ("format_prompt_as_json", "action_view_point")
ROT6D_COLUMNS = "rot6d_columns"


@functools.lru_cache(maxsize=None)
def realtime_manifest(model_path: str) -> CosmosDreamsManifest:
    """Parse the checkpoint's action contract in the HTTP process."""
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        get_diffusers_component_config,
    )

    return load_cosmos_dreams_manifest(
        get_diffusers_component_config(
            component_path=os.path.join(model_path, "transformer")
        )
    )


def idle_action_row(contract: EmbodimentContract) -> list[float]:
    """Raw row meaning "no motion": zeros, identity for rot6d rotation fields."""
    row = [0.0] * contract.raw_action_dim
    for field in contract.layout.get("fields", []):
        if field.get("representation") == ROT6D_COLUMNS:
            offset = int(field["offset"])
            row[offset] = 1.0
            row[offset + 4] = 1.0
    return row


def validate_action_row(value: Any, *, raw_action_dim: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != raw_action_dim:
        raise ValueError(
            f"Cosmos-Dreams action rows must have {raw_action_dim} numbers, got {value!r}."
        )
    row = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"Cosmos-Dreams action rows must be numeric, got {item!r}."
            )
        if not math.isfinite(item):
            raise ValueError("Cosmos-Dreams action rows must be finite.")
        row.append(float(item))
    return row


def validate_action_script(value: Any, *, raw_action_dim: int) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "Cosmos-Dreams action scripts must be a non-empty list of rows."
        )
    return [validate_action_row(row, raw_action_dim=raw_action_dim) for row in value]


class CosmosDreamsActionControlState:
    """Per-session action rows: a finite script or a held row, never both."""

    def __init__(
        self,
        *,
        idle_row: list[float],
        raw_action_dim: int,
        script_maxlen: int = 4096,
        max_transitions: int = 512,
    ) -> None:
        self.idle_row = list(idle_row)
        self.raw_action_dim = raw_action_dim
        self.script_queue = ControlScriptQueue(
            ACTION_EVENT_KIND, max_events=script_maxlen, default_item=self.idle_row
        )
        self.state_queue = ControlStateQueue(
            default_item=self.idle_row, max_transitions=max_transitions
        )
        self.latest_sampled_event_id: int | None = None

    def clear(self) -> None:
        self.script_queue.clear()
        self.state_queue.clear()
        self.latest_sampled_event_id = None

    def receive_script(
        self, rows: list[list[float]], *, event_id: int | None = None
    ) -> None:
        self.state_queue.clear()
        # A drained script settles on the idle row instead of null actions.
        self.state_queue.mark_received()
        self.script_queue.push_script(rows, event_id=event_id)

    def receive_state(
        self,
        row: list[float],
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        self.script_queue.clear()
        self.state_queue.push(
            ControlStateTransition(
                payload=row, seq_id=event_id, timestamp_ms=timestamp_ms
            )
        )

    def receive_event_payload(self, payload: Any, *, event_id: int | None) -> str:
        if (
            isinstance(payload, (list, tuple))
            and payload
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in payload
            )
        ):
            self.receive_state(
                validate_action_row(payload, raw_action_dim=self.raw_action_dim),
                event_id=event_id,
            )
            return f"kind={ACTION_EVENT_KIND}, mode=hold"
        parsed = parse_control_event_payload(
            payload,
            event_id=event_id,
            kind=ACTION_EVENT_KIND,
            normalize_state_payload=lambda row: validate_action_row(
                row, raw_action_dim=self.raw_action_dim
            ),
            validate_script_payload=lambda script: validate_action_script(
                script, raw_action_dim=self.raw_action_dim
            ),
        )
        if parsed.mode == "state":
            self.script_queue.clear()
            self.state_queue.push_many(parsed.payload)
            return f"kind={ACTION_EVENT_KIND}, mode=state, transitions={len(parsed.payload)}"
        self.receive_script(parsed.payload, event_id=event_id)
        return f"kind={ACTION_EVENT_KIND}, mode=script, rows={len(parsed.payload)}"

    def sample_rows(self, steps: int) -> list[list[float]]:
        """Exactly ``steps`` rows: the script first, then the held row, else idle."""
        if self.script_queue.has_script():
            rows = self.script_queue.sample_script(steps)
            self.latest_sampled_event_id = self.script_queue.last_sampled_seq_id()
            return rows
        rows = self.state_queue.sample_chunk(steps)
        if rows is None:
            return [list(self.idle_row) for _ in range(steps)]
        self.latest_sampled_event_id = self.state_queue.latest_sampled_seq_id()
        return rows


class CosmosDreamsRealtimeState:
    def __init__(self) -> None:
        self.manifest: CosmosDreamsManifest | None = None
        self.embodiment: str | None = None
        self.actions: CosmosDreamsActionControlState | None = None
        self.extra_sampling_fields: dict[str, Any] = {}

    def configure(
        self, manifest: CosmosDreamsManifest, *, embodiment: str
    ) -> CosmosDreamsActionControlState:
        contract = manifest.action_contract.embodiments[embodiment]
        self.manifest = manifest
        self.embodiment = embodiment
        self.actions = CosmosDreamsActionControlState(
            idle_row=idle_action_row(contract),
            raw_action_dim=contract.raw_action_dim,
        )
        return self.actions

    def clear(self) -> None:
        if self.actions is not None:
            self.actions.clear()
        self.manifest = None
        self.embodiment = None
        self.actions = None
        self.extra_sampling_fields = {}


class CosmosDreamsRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> CosmosDreamsRealtimeState:
        return CosmosDreamsRealtimeState()

    def _state(self, session: GenerateSession) -> CosmosDreamsRealtimeState:
        state = session.adapter_state
        if not isinstance(state, CosmosDreamsRealtimeState):
            raise ValueError("Cosmos-Dreams realtime session state is not initialized.")
        return state

    def _configured(
        self, session: GenerateSession
    ) -> tuple[
        CosmosDreamsRealtimeState, CosmosDreamsManifest, CosmosDreamsActionControlState
    ]:
        state = self._state(session)
        if state.manifest is None or state.actions is None or state.embodiment is None:
            raise ValueError("Cosmos-Dreams realtime session has not received init.")
        return state, state.manifest, state.actions

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        state = self._state(session)
        manifest = realtime_manifest(get_global_server_args().model_path)
        condition_inputs = request.condition_inputs or {}
        embodiment = manifest.action_contract.resolve_embodiment(
            condition_inputs.get("domain_name"), condition_inputs.get("domain_id")
        )
        actions = state.configure(manifest, embodiment=embodiment)
        script = condition_inputs.get(ACTION_ROWS_CONDITION)
        if script is not None:
            actions.receive_script(
                validate_action_script(
                    script,
                    raw_action_dim=manifest.action_contract.embodiments[
                        embodiment
                    ].raw_action_dim,
                )
            )
        extra = request.model_extra or {}
        state.extra_sampling_fields = {
            name: extra[name]
            for name in COSMOS_DREAMS_REALTIME_EXTRA_FIELDS
            if extra.get(name) is not None
        }
        await save_realtime_first_frame(
            session,
            request,
            required_error="Cosmos-Dreams realtime sessions require first_frame.",
        )

    def ingest_event(self, session: GenerateSession, event: RealtimeEvent) -> str:
        _, _, actions = self._configured(session)
        if event.kind != ACTION_EVENT_KIND:
            raise ValueError(
                f"unsupported event kind: {event.kind}; Cosmos-Dreams accepts {ACTION_EVENT_KIND!r}"
            )
        return actions.receive_event_payload(event.payload, event_id=event.event_id)

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        del server_args, chunk
        _, manifest, _ = self._configured(session)
        return manifest.chunk_size

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        del server_args, chunk
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        _, manifest, actions = self._configured(session)
        rows = actions.sample_rows(chunk_size * manifest.action_tokens_per_frame)
        return RealtimeChunkInputs(
            prompt=request.prompt, condition_inputs={ACTION_ROWS_CONDITION: rows}
        )

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        del server_args
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        state, manifest, _ = self._configured(session)
        kwargs: dict[str, Any] = dict(
            prompt=chunk_inputs.prompt,
            fps=request.fps or COSMOS_DREAMS_REALTIME_DEFAULT_FPS,
            # Frame 0 plus one block; the tick stages ignore the horizon beyond it.
            num_frames=1 + chunk_size * manifest.temporal_compression_factor,
            seed=request.seed,
            guidance_scale=1.0,
            save_output=False,
            condition_inputs=chunk_inputs.condition_inputs,
            realtime_chunk_size=chunk_size,
            domain_name=state.embodiment,
            # The conditioning image is encoded once; later ticks restore it.
            image_path=request.first_frame if chunk.index == 0 else None,
        )
        if request.width is not None or request.height is not None:
            kwargs.update(width=request.width, height=request.height)
        elif "size" in request.model_fields_set:
            kwargs["size"] = request.size
        kwargs.update(state.extra_sampling_fields)
        return build_sampling_params(chunk.request_id, **kwargs)

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        state = self._state(session)
        return None if state.actions is None else state.actions.latest_sampled_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, CosmosDreamsRealtimeState):
            state.clear()
