# SPDX-License-Identifier: Apache-2.0

import itertools
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from sglang.omni.core.coordinator import OmniCoordinator
from sglang.omni.core.interleaved import TEXT_ROLE_METADATA_KEY, TEXT_ROLE_THINK
from sglang.omni.core.protocol import (
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniContextRef,
    OmniInputSegment,
    OmniRequest,
)


@dataclass(slots=True)
class _InMemoryContextOps:
    metadata: dict[str, Any] = field(default_factory=dict)
    generation_kind: str | None = None
    session_id: str | None = None

    def get_role(self, name: str, default: str):
        return default

    def get_model(self):
        return None

    def get_position_count(self, *, condition_path_role: str | None = None):
        return 0

    def run_temporary_forward(self, *, prepared, forward):
        raise RuntimeError("no temporary forward backend")


class _ScriptedARBackend:
    def __init__(self, boundaries: list[OmniBoundary]):
        self._boundaries = list(boundaries)
        self._counter = itertools.count()
        self.appended_segments: list[GeneratedSegment] = []
        self.appended_inputs: list[OmniRequest] = []
        self.released_contexts: list[OmniContextBundle] = []

    def begin_request_context(
        self,
        request: OmniRequest,
        *,
        stream_sink=None,
    ) -> OmniContextBundle:
        context_id = f"scripted-{next(self._counter)}"
        return OmniContextBundle(
            full=OmniContextRef(
                context_id=context_id,
                metadata={
                    "messages": [message.to_dict() for message in request.messages]
                },
            )
        )

    def append_input_segments(self, context, request, *, stream_sink=None):
        self.appended_inputs.append(request)
        context.full.version += 1
        return context

    def decode_until_boundary(self, context, *, request, stream_sink=None):
        if not self._boundaries:
            return OmniBoundary(type="done")
        return self._boundaries.pop(0)

    def append_generated_segment(self, context, segment, *, request):
        self.appended_segments.append(segment)
        context.full.version += 1
        context.full.metadata["last_generated_type"] = segment.type
        return context

    def get_context_ops(self, context):
        return _InMemoryContextOps(metadata=dict(context.full.metadata))

    def release(self, context):
        self.released_contexts.append(context)


class _ImageBackend:
    def __init__(self):
        self.calls = 0

    def generate_segment(self, request, context_ops):
        self.calls += 1
        return GeneratedSegment(
            type="image",
            image={"b64_json": "abc"},
            metadata={"position_count": context_ops.get_position_count()},
        )


class _AlwaysImageARBackend:
    def __init__(self):
        self.released_contexts: list[OmniContextBundle] = []

    def begin_request_context(
        self,
        request: OmniRequest,
        *,
        stream_sink=None,
    ) -> OmniContextBundle:
        return OmniContextBundle(
            full=OmniContextRef(
                context_id=f"concurrent-{id(request)}",
                metadata={"decode_count": 0},
            )
        )

    def append_input_segments(self, context, request, *, stream_sink=None):
        return context

    def decode_until_boundary(self, context, *, request, stream_sink=None):
        decode_count = int(context.full.metadata.get("decode_count", 0))
        context.full.metadata["decode_count"] = decode_count + 1
        if decode_count == 0:
            return OmniBoundary(type="image")
        return OmniBoundary(type="done")

    def append_generated_segment(self, context, segment, *, request):
        return context

    def get_context_ops(self, context):
        return _InMemoryContextOps(metadata=dict(context.full.metadata))

    def release(self, context):
        self.released_contexts.append(context)


class _SlowImageBackend:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def generate_segment(self, request, context_ops):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return GeneratedSegment(type="image", image={"b64_json": "abc"})
        finally:
            with self.lock:
                self.active -= 1


class TestOmniCoordinator(unittest.TestCase):
    def test_commits_image_before_continuing_text(self):
        ar_backend = _ScriptedARBackend(
            [
                OmniBoundary(type="text", text="before", token_ids=(1,)),
                OmniBoundary(type="image"),
                OmniBoundary(type="text", text="after", token_ids=(2,)),
                OmniBoundary(type="done"),
            ]
        )
        gen_backend = _ImageBackend()
        coordinator = OmniCoordinator(ar_backend, gen_backend)
        request = OmniRequest(messages=(OmniInputSegment(type="text", text="draw"),))

        response = coordinator.generate(request)

        self.assertEqual(["text", "image", "text"], [s.type for s in response.segments])
        self.assertEqual(1, gen_backend.calls)
        self.assertEqual(1, len(ar_backend.appended_segments))
        self.assertEqual(1, len(ar_backend.released_contexts))
        self.assertEqual(
            {
                "num_segments": 3,
                "num_text_segments": 2,
                "num_image_segments": 1,
                "num_audio_segments": 0,
                "num_video_segments": 0,
            },
            response.stats,
        )

    def test_session_turn_keeps_context_and_continues_text_after_image(self):
        ar_backend = _ScriptedARBackend(
            [
                OmniBoundary(type="image"),
                OmniBoundary(type="text", text="after"),
                OmniBoundary(type="image"),
            ]
        )
        gen_backend = _ImageBackend()
        coordinator = OmniCoordinator(ar_backend, gen_backend)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            max_images=1,
        )

        response, context = coordinator.generate_with_context(
            request,
            release_context=False,
            stop_after_generation_limit=True,
        )

        self.assertEqual(["image", "text"], [s.type for s in response.segments])
        self.assertEqual(1, gen_backend.calls)
        self.assertEqual(0, len(ar_backend.released_contexts))

        ar_backend._boundaries = [OmniBoundary(type="text", text="next")]
        response, _ = coordinator.generate_with_context(
            OmniRequest(messages=(OmniInputSegment(type="text", text="again"),)),
            context=context,
            release_context=False,
        )

        self.assertEqual("again", ar_backend.appended_inputs[0].messages[0].text)
        self.assertEqual(["text"], [s.type for s in response.segments])

    def test_generation_admission_limits_media_concurrency(self):
        gen_backend = _SlowImageBackend()
        coordinator = OmniCoordinator(
            _AlwaysImageARBackend(),
            gen_backend,
            max_concurrent_generations=1,
        )
        request = OmniRequest(messages=(OmniInputSegment(type="text", text="draw"),))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(coordinator.generate, request) for _ in range(2)]
            for future in futures:
                future.result()

        self.assertEqual(1, gen_backend.max_active)

    def test_think_text_does_not_count_against_user_visible_text_limit(self):
        ar_backend = _ScriptedARBackend(
            [
                OmniBoundary(
                    type="text",
                    text="internal prompt",
                    metadata={TEXT_ROLE_METADATA_KEY: TEXT_ROLE_THINK},
                ),
                OmniBoundary(type="image"),
                OmniBoundary(type="done"),
            ]
        )
        gen_backend = _ImageBackend()
        coordinator = OmniCoordinator(ar_backend, gen_backend)
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            max_text_segments=0,
        )

        response = coordinator.generate(request)

        self.assertEqual(["image"], [segment.type for segment in response.segments])
        self.assertEqual(1, gen_backend.calls)


if __name__ == "__main__":
    unittest.main()
