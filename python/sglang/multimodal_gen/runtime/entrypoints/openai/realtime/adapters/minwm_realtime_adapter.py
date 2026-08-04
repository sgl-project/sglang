# SPDX-License-Identifier: Apache-2.0
"""Realtime control adapter for MinWM's discrete primitive action labels."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_LABELS_CONDITION,
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_CHUNK_SEED_CONDITION,
    MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION,
    MINWM_CHUNK_SEEDS_INPUT,
    MINWM_CONDITION_SWITCH_CONDITION,
    MINWM_PROMPT_SCHEDULE_INPUT,
    MINWM_PROMPT_UPDATED_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
    MINWM_TOTAL_LATENT_FRAMES_CONDITION,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    key_state_to_action_label,
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlScriptQueue,
    ControlSignalQueue,
)
from sglang.multimodal_gen.runtime.realtime.states import RealtimeCameraControlState
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

MINWM_DEFAULT_DMD_STEPS = 4


class MinWMRealtimeState(RealtimeCameraControlState):
    def __init__(self) -> None:
        super().__init__(min_pulse_items=1, script_maxlen=4096, max_transitions=512)
        self.action_label_queue = ControlScriptQueue(
            "action_labels", max_events=4096, default_item=0
        )
        self.action_weight_queue = ControlScriptQueue(
            "action_weights", max_events=32768, default_item=[0.0] * 8
        )
        self.prompt_queue = ControlSignalQueue(max_events={"condition_switch": 1})
        self.seed_queue = ControlScriptQueue(
            "chunk_seeds", max_events=4096, default_item=0
        )
        self.prompt_event_id: int | None = None
        self.label_event_id: int | None = None
        self.weight_event_id: int | None = None
        self.seed_event_id: int | None = None
        self.prompt_schedule: dict[int, tuple[str, str]] = {}
        self.action_mode = "camera"

    def clear(self) -> None:
        super().clear()
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.prompt_queue.clear()
        self.seed_queue.clear()
        self.prompt_event_id = None
        self.label_event_id = None
        self.weight_event_id = None
        self.seed_event_id = None
        self.prompt_schedule.clear()
        self.action_mode = "camera"

    def receive_action_labels(
        self, labels: list[int], *, event_id: int | None = None
    ) -> None:
        self.camera_state_queue.clear()
        self.camera_script_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "labels"
        self.action_label_queue.push_script(labels, event_id=event_id)

    def receive_action_weights(
        self, weights: list[list[float]], *, event_id: int | None = None
    ) -> None:
        self.camera_state_queue.clear()
        self.camera_script_queue.clear()
        self.action_label_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "weights"
        self.action_weight_queue.push_script(weights, event_id=event_id)

    def receive_camera_action_script(
        self, camera_actions: list[list[str]], *, event_id: int | None = None
    ) -> None:
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "camera"
        super().receive_camera_action_script(camera_actions, event_id=event_id)

    def receive_camera_state_transitions(self, transitions) -> None:
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "camera"
        super().receive_camera_state_transitions(transitions)

    def receive_prompt(
        self,
        prompt: str,
        *,
        event_id: int | None = None,
        switch_kind: str = "prompt",
    ) -> None:
        if switch_kind not in {"prompt", "scene_cut"}:
            raise ValueError("MinWM condition switch must be prompt or scene_cut")
        self.prompt_queue.replace(
            "condition_switch",
            {"kind": switch_kind, "prompt": prompt},
            event_id=event_id,
        )

    def receive_chunk_seeds(
        self, seeds: list[int], *, event_id: int | None = None
    ) -> None:
        self.seed_queue.push_script(seeds, event_id=event_id)

    def receive_prompt_schedule(self, schedule: dict[int, tuple[str, str]]) -> None:
        self.prompt_schedule = dict(schedule)

    def sample_chunk_seed(self) -> int | None:
        if not self.seed_queue.has_script():
            return None
        seed = self.seed_queue.sample_script(1)[0]
        self.seed_event_id = self.seed_queue.last_sampled_seq_id()
        return int(seed)

    def sample_prompt(self) -> tuple[str, str]:
        condition_switch = self.prompt_queue.pop_latest("condition_switch")
        if not isinstance(condition_switch, dict):
            raise ValueError("MinWM condition switch payload must be an object")
        prompt = condition_switch.get("prompt")
        switch_kind = condition_switch.get("kind")
        if not isinstance(prompt, str) or switch_kind not in {"prompt", "scene_cut"}:
            raise ValueError("invalid MinWM condition switch payload")
        self.prompt_event_id = self.prompt_queue.last_sampled_seq_id("condition_switch")
        return prompt, switch_kind

    def sample_action_labels(self, chunk_size: int) -> list[int]:
        if self.action_label_queue.has_script():
            labels = self.action_label_queue.sample_script(chunk_size)
            self.label_event_id = self.action_label_queue.last_sampled_seq_id()
            return validate_action_labels(labels, expected_frames=chunk_size)
        camera_actions = self.sample_camera_actions(chunk_size)
        if camera_actions is None:
            return [0] * chunk_size
        return [key_state_to_action_label(keys) for keys in camera_actions]

    def sample_action_weights(self, frame_count: int) -> list[list[float]]:
        weights = self.action_weight_queue.sample_script(frame_count)
        self.weight_event_id = self.action_weight_queue.last_sampled_seq_id()
        return validate_action_weights(weights, expected_frames=frame_count)


class MinWMRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> MinWMRealtimeState:
        return MinWMRealtimeState()

    @staticmethod
    def _state(session: GenerateSession) -> MinWMRealtimeState:
        state = session.adapter_state
        if not isinstance(state, MinWMRealtimeState):
            raise TypeError("MinWM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError("camera_actions must be list[list[str]]")
        result = []
        for frame in payload:
            if not isinstance(frame, list):
                raise ValueError("camera_actions must be list[list[str]]")
            keys = [str(key).lower().strip() for key in frame]
            key_state_to_action_label(keys)
            result.append(keys)
        return result

    @staticmethod
    def _validate_prompt(payload: Any) -> str:
        if not isinstance(payload, str) or not payload:
            raise ValueError("prompt event payload must be a non-empty string")
        return payload

    @staticmethod
    def _validate_chunk_seeds(payload: Any) -> list[int]:
        if not isinstance(payload, list) or not payload:
            raise ValueError("chunk_seeds must be a non-empty list[int]")
        seeds = []
        for seed in payload:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("chunk_seeds must be a non-empty list[int]")
            if not 0 <= seed < 2**63:
                raise ValueError("MinWM chunk seeds must be in [0, 2**63)")
            seeds.append(seed)
        return seeds

    @staticmethod
    def _validate_prompt_schedule(payload: Any) -> dict[int, tuple[str, str]]:
        if not isinstance(payload, list):
            raise ValueError("minwm_prompt_schedule must be a list")
        schedule = {}
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("each MinWM prompt schedule item must be an object")
            target_chunk = item.get("target_chunk")
            prompt = item.get("prompt")
            switch_kind = item.get("kind", "prompt")
            if (
                isinstance(target_chunk, bool)
                or not isinstance(target_chunk, int)
                or target_chunk < 1
            ):
                raise ValueError(
                    "MinWM prompt schedule target_chunk must be a positive integer"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("MinWM prompt schedule prompt must be non-empty")
            if switch_kind not in {"prompt", "scene_cut"}:
                raise ValueError(
                    "MinWM prompt schedule kind must be prompt or scene_cut"
                )
            if target_chunk in schedule:
                raise ValueError(
                    f"duplicate MinWM prompt schedule chunk {target_chunk}"
                )
            schedule[target_chunk] = (prompt, switch_kind)
        return schedule

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        state = self._state(session)
        if request.num_inference_steps not in (None, MINWM_DEFAULT_DMD_STEPS):
            raise ValueError("MinWM DMD checkpoint requires exactly 4 inference steps")
        # The distilled student has no CFG lane. The realtime protocol defaults
        # to guidance_scale=1 for other models, so normalize it explicitly.
        request.num_inference_steps = MINWM_DEFAULT_DMD_STEPS
        request.guidance_scale = 0.0
        request.guidance_scale_2 = None
        self._normalize_generation_mode(request)
        inputs = request.condition_inputs or {}
        chunk_seeds_value = inputs.get(
            "chunk_seeds", inputs.get(MINWM_CHUNK_SEEDS_INPUT)
        )
        chunk_seeds = (
            None
            if chunk_seeds_value is None
            else self._validate_chunk_seeds(chunk_seeds_value)
        )
        if chunk_seeds is not None:
            state.receive_chunk_seeds(chunk_seeds)
        prompt_schedule = self._validate_prompt_schedule(
            inputs.get(MINWM_PROMPT_SCHEDULE_INPUT, [])
        )
        state.receive_prompt_schedule(prompt_schedule)
        label_values = [
            value
            for key in ("action_labels", MINWM_ACTION_LABELS_CONDITION)
            if (value := inputs.get(key)) is not None
        ]
        weight_values = [
            value
            for key in ("action_weights", MINWM_ACTION_WEIGHTS_CONDITION)
            if (value := inputs.get(key)) is not None
        ]
        camera_actions = inputs.get("camera_actions")
        if len(label_values) + len(weight_values) + int(camera_actions is not None) > 1:
            raise ValueError(
                "pass exactly one MinWM action form: action_labels, action_weights, or camera_actions"
            )
        if label_values:
            state.receive_action_labels(validate_action_labels(label_values[0]))
        elif weight_values:
            state.receive_action_weights(validate_action_weights(weight_values[0]))
        elif camera_actions is not None:
            state.receive_camera_action_script(
                self._validate_camera_actions(camera_actions)
            )
        await save_realtime_first_frame(
            session,
            request,
            cache_remote_urls=True,
        )
        if self._is_t2v_request(request) and request.num_frames is not None:
            total_latent_frames = self._t2v_total_latent_frames(
                request, get_global_server_args()
            )
            arch_config = (
                get_global_server_args().pipeline_config.dit_config.arch_config
            )
            first_block = int(arch_config.num_frame_first_block)
            regular_block = int(arch_config.num_frames_per_block)
            remaining = total_latent_frames - first_block
            total_chunks = 1 + max(0, remaining + regular_block - 1) // regular_block
            if request.max_chunks is not None and request.max_chunks != total_chunks:
                raise ValueError(
                    "MinWM T2V max_chunks does not match num_frames: "
                    f"expected {total_chunks}, got {request.max_chunks}"
                )
            request.max_chunks = total_chunks
        if (
            chunk_seeds is not None
            and request.max_chunks is not None
            and len(chunk_seeds) != request.max_chunks
        ):
            raise ValueError(
                "MinWM chunk_seeds length must match max_chunks: "
                f"{len(chunk_seeds)} vs {request.max_chunks}"
            )
        if request.max_chunks is not None and any(
            chunk_index >= request.max_chunks for chunk_index in prompt_schedule
        ):
            raise ValueError("MinWM prompt schedule target is outside max_chunks")

    @staticmethod
    def _normalize_generation_mode(
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        inferred_mode = "i2v" if request.first_frame is not None else "t2v"
        requested_mode = request.generation_mode
        if requested_mode == "i2v" and request.first_frame is None:
            raise ValueError("MinWM I2V requires first_frame")
        if requested_mode == "t2v" and request.first_frame is not None:
            raise ValueError("MinWM T2V does not accept first_frame")
        request.generation_mode = requested_mode or inferred_mode

    @staticmethod
    def _is_t2v_request(request: RealtimeVideoGenerationsRequest) -> bool:
        generation_mode = getattr(request, "generation_mode", None)
        if generation_mode is not None:
            return generation_mode == "t2v"
        return getattr(request, "first_frame", None) is None

    @staticmethod
    def _t2v_total_latent_frames(
        request: RealtimeVideoGenerationsRequest,
        server_args: ServerArgs,
    ) -> int | None:
        num_frames_value = getattr(request, "num_frames", None)
        if (
            not MinWMRealtimeAdapter._is_t2v_request(request)
            or num_frames_value is None
        ):
            return None
        num_frames = int(num_frames_value)
        temporal_factor = int(
            server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
        )
        if num_frames < 1 or (num_frames - 1) % temporal_factor:
            raise ValueError(
                "MinWM T2V num_frames must equal "
                f"1 + N * {temporal_factor}, got {num_frames}"
            )
        total_latent_frames = 1 + (num_frames - 1) // temporal_factor
        first_block = int(
            server_args.pipeline_config.dit_config.arch_config.num_frame_first_block
        )
        if total_latent_frames < first_block:
            raise ValueError(
                "MinWM T2V num_frames is shorter than num_frame_first_block"
            )
        return total_latent_frames

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        request = session.request
        if request is None or not self._is_t2v_request(request):
            return super().get_chunk_size(session, server_args, chunk)
        arch_config = server_args.pipeline_config.dit_config.arch_config
        first_block = int(arch_config.num_frame_first_block)
        regular_block = int(arch_config.num_frames_per_block)
        if chunk.index == 0:
            return first_block
        total_latent_frames = self._t2v_total_latent_frames(request, server_args)
        if total_latent_frames is None:
            return regular_block
        generated_before = first_block + (chunk.index - 1) * regular_block
        remaining = total_latent_frames - generated_before
        if remaining <= 0:
            raise ValueError("MinWM T2V request exceeded num_frames")
        return min(regular_block, remaining)

    def ingest_event(self, session: GenerateSession, event: RealtimeEvent) -> str:
        state = self._state(session)
        if event.kind == "action_labels":
            labels = validate_action_labels(event.payload)
            state.receive_action_labels(labels, event_id=event.event_id)
            return f"kind=action_labels, frames={len(labels)}"
        if event.kind == "action_weights":
            weights = validate_action_weights(event.payload)
            state.receive_action_weights(weights, event_id=event.event_id)
            return f"kind=action_weights, frames={len(weights)}"
        if event.kind == "camera_actions":
            return state.receive_camera_control_event_payload(
                event.payload,
                event_id=event.event_id,
                validate_camera_actions=self._validate_camera_actions,
            )
        if event.kind in {"prompt", "scene_cut"}:
            prompt = self._validate_prompt(event.payload)
            state.receive_prompt(
                prompt,
                event_id=event.event_id,
                switch_kind=event.kind,
            )
            return f"kind={event.kind}, prompt_len={len(prompt)}"
        if event.kind in {"seed", "chunk_seeds"}:
            payload = [event.payload] if event.kind == "seed" else event.payload
            seeds = self._validate_chunk_seeds(payload)
            state.receive_chunk_seeds(seeds, event_id=event.event_id)
            return f"kind={event.kind}, seeds={len(seeds)}"
        raise ValueError(f"unsupported MinWM event kind: {event.kind}")

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        prompt_updated = False
        condition_switch = None
        prompt = request.prompt
        scheduled_prompt = state.prompt_schedule.get(chunk.index)
        if scheduled_prompt is not None:
            prompt, condition_switch = scheduled_prompt
            request.prompt = prompt
            prompt_updated = True
        elif chunk.index > 0 and state.prompt_queue.has_events("condition_switch"):
            prompt, condition_switch = state.sample_prompt()
            request.prompt = prompt
            prompt_updated = True
        condition_inputs = {}
        t2v_first_block = self._is_t2v_request(request) and chunk.index == 0
        if state.action_mode == "weights":
            temporal_factor = int(
                server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
            )
            rows = (
                [[0.0] * 8] * (chunk_size * temporal_factor)
                if t2v_first_block
                else state.sample_action_weights(chunk_size * temporal_factor)
            )
            condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION] = [
                rows[start : start + temporal_factor]
                for start in range(0, len(rows), temporal_factor)
            ]
        else:
            condition_inputs[MINWM_ACTION_LABELS_CONDITION] = (
                [0] * chunk_size
                if t2v_first_block
                else state.sample_action_labels(chunk_size)
            )
        if request.max_chunks is not None:
            condition_inputs[MINWM_TOTAL_CHUNKS_CONDITION] = int(request.max_chunks)
        total_latent_frames = self._t2v_total_latent_frames(request, server_args)
        if total_latent_frames is not None:
            condition_inputs[MINWM_TOTAL_LATENT_FRAMES_CONDITION] = total_latent_frames
        if prompt_updated:
            condition_inputs[MINWM_PROMPT_UPDATED_CONDITION] = True
            condition_inputs[MINWM_CONDITION_SWITCH_CONDITION] = condition_switch
        chunk_seed = state.sample_chunk_seed()
        if chunk_seed is not None:
            condition_inputs[MINWM_CHUNK_SEED_CONDITION] = chunk_seed
            if not self._is_t2v_request(request):
                prefix_frames = 1 + chunk.index * int(
                    server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
                )
            elif chunk.index == 0:
                prefix_frames = 0
            else:
                arch_config = server_args.pipeline_config.dit_config.arch_config
                prefix_frames = int(arch_config.num_frame_first_block) + (
                    chunk.index - 1
                ) * int(arch_config.num_frames_per_block)
            condition_inputs[MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION] = prefix_frames
        return RealtimeChunkInputs(prompt=prompt, condition_inputs=condition_inputs)

    def refresh_queued_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        batch,
        event_kind: str,
    ):
        # Only keyboard state is safe to preview without consuming a scripted
        # action or prompt queue. Other controls apply at the next chunk boundary.
        if event_kind != "camera_actions":
            return None
        request = session.request
        if request is None:
            return None

        replacement = copy(batch)
        replacement.condition_inputs = dict(batch.condition_inputs or {})
        chunk_size = int(
            getattr(batch, "realtime_chunk_size", None)
            or self.get_chunk_size(session, server_args, chunk)
        )
        state = self._state(session)
        if state.action_mode != "camera":
            return None
        preview_state = deepcopy(state)
        replacement.condition_inputs.pop(MINWM_ACTION_LABELS_CONDITION, None)
        replacement.condition_inputs.pop(MINWM_ACTION_WEIGHTS_CONDITION, None)
        t2v_first_block = self._is_t2v_request(request) and chunk.index == 0
        replacement.condition_inputs[MINWM_ACTION_LABELS_CONDITION] = (
            [0] * chunk_size
            if t2v_first_block
            else preview_state.sample_action_labels(chunk_size)
        )

        replacement.realtime_action_version = session.action_version
        replacement.realtime_prompt_version = getattr(
            batch,
            "realtime_prompt_version",
            chunk.prompt_version,
        )
        replacement.realtime_event_id = self._get_state_realtime_event_id(preview_state)
        return replacement

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
        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=1,
            num_inference_steps=request.num_inference_steps or MINWM_DEFAULT_DMD_STEPS,
            chunk_size=chunk_size,
        )

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._get_state_realtime_event_id(self._state(session))

    @staticmethod
    def _get_state_realtime_event_id(state: MinWMRealtimeState) -> int | None:
        # Realtime clients issue monotonically increasing event IDs and the
        # playback cutover waits for frame.event_id >= the pending event.
        # A chunk can sample prompt and action state together, so report the
        # newest event that actually influenced it instead of prioritizing one
        # condition kind and accidentally returning an older ID.
        sampled_event_ids = (
            state.prompt_event_id,
            state.label_event_id,
            state.weight_event_id,
            state.seed_event_id,
            state.latest_sampled_event_id,
        )
        return max(
            (event_id for event_id in sampled_event_ids if event_id is not None),
            default=None,
        )

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, MinWMRealtimeState):
            state.clear()
