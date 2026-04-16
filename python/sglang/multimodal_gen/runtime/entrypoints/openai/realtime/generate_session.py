from collections import deque
from enum import Enum
from uuid import uuid4

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    RealtimeSession,
)


class RealtimeVideoMode(str, Enum):
    T2V = "t2v"
    V2V = "v2v"


class GenerateSession:
    _FIRST_BLOCK_ENCODE_FRAMES = 9
    _NEXT_BLOCK_ENCODE_FRAMES = 12

    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request = None
        self.mode: RealtimeVideoMode | None = None
        self.action_queue = deque(maxlen=1)
        self.video_frame_queue = deque(maxlen=256)
        self.generate_chunk_cnt = 0
        self.realtime_session = RealtimeSession()

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def set_mode(self, mode: RealtimeVideoMode | None):
        self.mode = mode

    def dispose(self):
        self.action_queue.clear()
        self.video_frame_queue.clear()
        self.mode = None
        self.request = None
        self.request_id = None
        self.generate_chunk_cnt = 0
        self.realtime_session.dispose()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_action(self, action: RealtimeAction):
        self.action_queue.append(action)

    def append_video_frames(self, frames: list):
        if len(frames) > 0:
            self.video_frame_queue.extend(frames)

    def has_pending_video_frames(self) -> bool:
        return len(self.video_frame_queue) >= self.required_video_frames()

    def is_v2v_enabled(self) -> bool:
        if self.request is None:
            return False
        if self.mode is not None:
            return self.mode == RealtimeVideoMode.V2V
        # Auto mode only checks first_frame.
        return self.request.first_frame is not None

    def required_video_frames(self) -> int:
        # todo make _FIRST_BLOCK_ENCODE_FRAMES and _NEXT_BLOCK_ENCODE_FRAMES config
        if self.generate_chunk_cnt == 0:
            return self._FIRST_BLOCK_ENCODE_FRAMES
        return self._NEXT_BLOCK_ENCODE_FRAMES

    def sample_action(self) -> RealtimeAction:
        return self.action_queue.popleft()

    def sample_video_frames(self):
        required = self.required_video_frames()
        if len(self.video_frame_queue) < required:
            return None

        pending_frames = []
        while len(self.video_frame_queue) > 0:
            pending_frames.append(self.video_frame_queue.popleft())
        if len(pending_frames) < required:
            return None
        if len(pending_frames) == required:
            return pending_frames

        # TODO more sampling strategy.
        indices = np.round(np.linspace(0, len(pending_frames) - 1, required)).astype(
            int
        )
        return [pending_frames[i] for i in indices]

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        elif len(self.action_queue) > 0:
            realtime_action = self.sample_action()
            # TODO more sampling strategy.
            # only support prompt action now
            if realtime_action.type == "prompt":
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
            seed=self.request.seed,
            generator_device=self.request.generator_device,
            num_inference_steps=self.request.num_inference_steps,
            guidance_scale=self.request.guidance_scale,
            guidance_scale_2=self.request.guidance_scale_2,
            negative_prompt=self.request.negative_prompt,
            enable_teacache=self.request.enable_teacache,
            profile=self.request.profile,
            num_profiled_timesteps=self.request.num_profiled_timesteps,
            profile_all_stages=self.request.profile_all_stages,
            perf_dump_path=self.request.perf_dump_path,
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )
