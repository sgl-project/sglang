# SPDX-License-Identifier: Apache-2.0

from collections import deque
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    RealtimeSession,
)


class GenerateSession:
    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.prompt_queue = deque(maxlen=1)
        self.control_queue = deque(maxlen=512)
        self.generate_chunk_cnt = 0
        self.last_control_actions: list[str] = []
        self.has_control_state = False
        self.realtime_session = RealtimeSession()

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        self.prompt_queue.clear()
        self.control_queue.clear()
        self.request = None
        self.request_id = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.last_control_actions = []
        self.has_control_state = False
        self.realtime_session.dispose()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_prompt_action(self, action: RealtimeAction):
        self.prompt_queue.append(action)

    def _append_control_frame(self, actions: list[str]):
        normalized = list(actions)
        self.control_queue.append(normalized)
        self.last_control_actions = normalized
        self.has_control_state = True

    def append_control_chunk(self, control_chunk: list[list[str]]):
        if len(control_chunk) == 0:
            self.last_control_actions = []
            self.has_control_state = True
            return
        for actions in control_chunk:
            self._append_control_frame(actions)

    def sample_prompt_action(self) -> RealtimeAction:
        return self.prompt_queue.popleft()

    def sample_control_chunk(self, chunk_size: int) -> list[list[str]] | None:
        if chunk_size <= 0:
            return None

        chunk: list[list[str]] = []
        while len(chunk) < chunk_size and len(self.control_queue) > 0:
            chunk.append(list(self.control_queue.popleft()))

        if len(chunk) == 0 and not self.has_control_state:
            # Keep emitting an explicit no-op control chunk before any user
            # control arrives.
            return [[] for _ in range(chunk_size)]

        pad_actions = list(self.last_control_actions)
        while len(chunk) < chunk_size:
            chunk.append(list(pad_actions))
        return chunk

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        elif len(self.prompt_queue) > 0:
            realtime_action = self.sample_prompt_action()
            prompt = realtime_action.action_content
            self.request.prompt = prompt
        else:
            prompt = self.request.prompt

        return build_sampling_params(
            self.request_id,
            prompt=prompt,
            size=self.request.size,
            num_frames=self.request.num_frames,
            fps=self.request.fps,
            image_path=self.request.first_frame,
            output_file_name=self.request_id,
            save_output=False,
            seed=self.request.seed,
            generator_device=self.request.generator_device,
            num_inference_steps=self.request.num_inference_steps,
            guidance_scale=self.request.guidance_scale,
            guidance_scale_2=self.request.guidance_scale_2,
            negative_prompt=self.request.negative_prompt,
            enable_teacache=self.request.enable_teacache,
            enable_frame_interpolation=self.request.enable_frame_interpolation,
            frame_interpolation_exp=self.request.frame_interpolation_exp,
            frame_interpolation_scale=self.request.frame_interpolation_scale,
            frame_interpolation_model_path=self.request.frame_interpolation_model_path,
            enable_upscaling=self.request.enable_upscaling,
            upscaling_model_path=self.request.upscaling_model_path,
            upscaling_scale=self.request.upscaling_scale,
            diffusers_kwargs=self.request.diffusers_kwargs,
            profile=self.request.profile,
            num_profiled_timesteps=self.request.num_profiled_timesteps,
            profile_all_stages=self.request.profile_all_stages,
            perf_dump_path=self.request.perf_dump_path,
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )
