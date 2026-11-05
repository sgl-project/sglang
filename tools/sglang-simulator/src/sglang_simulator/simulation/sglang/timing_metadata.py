"""Constant-size virtual timing snapshots carried by native response metadata."""

from dataclasses import dataclass
from functools import wraps

import msgspec
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.coordinator import Coordinator, ProtocolError
from sglang_simulator.simulation.sglang.token_events import TokenEventRecorder

METADATA_KEY = "sglang_simulator"
_REQUEST_ATTRIBUTE = "_simulator_timing"


@dataclass
class TimingMetadataOutput:
    """Host auxiliary output, consumed after native visible-token commits."""

    clock: Coordinator
    commit_ns: int
    previous: object = None
    token_events: TokenEventRecorder | None = None

    def consume(self, batch, commits) -> None:
        if self.previous is not None:
            self.previous.consume(batch, commits)
        output_events = []
        for req, commit in zip(batch.reqs, commits, strict=True):
            prior = getattr(req, _REQUEST_ATTRIBUTE, None)
            count = prior["completion_tokens"] if prior is not None else 0
            first_ns = prior["first_token_ns"] if prior is not None else None
            last_ns = prior["last_token_ns"] if prior is not None else None
            # Native abort cleanup can append a running batch's result after
            # the protocol cancellation. That discarded tail is not output.
            if commit is not None and not self.clock.is_cancelled(req.rid):
                if commit.output_index != count:
                    raise ProtocolError(f"non-contiguous token commit for {req.rid}")
                count += len(commit.token_ids)
                if commit.token_ids:
                    if self.token_events is not None:
                        output_events.append(
                            (req.rid, commit.output_index, len(commit.token_ids))
                        )
                    if first_ns is None:
                        first_ns = self.commit_ns
                    last_ns = self.commit_ns
            snapshot = {
                "version": 1,
                "rid": req.rid,
                "arrival_ns": self.clock.request_arrival_ns(req.rid),
                "first_token_ns": first_ns,
                "last_token_ns": last_ns,
                "commit_ns": self.commit_ns,
                "terminal_ns": self.commit_ns if req.finished() else None,
                "prompt_tokens": len(req.origin_input_ids),
                "completion_tokens": count,
                "cached_tokens": req.cached_tokens,
            }
            # Replace instead of mutating: an earlier IPC output may still hold it.
            setattr(req, _REQUEST_ATTRIBUTE, snapshot)
        if self.token_events is not None and output_events:
            self.token_events.record(self.commit_ns, output_events)


def attach_timing_output(
    result, clock: Coordinator, token_events: TokenEventRecorder | None = None
) -> None:
    result.auxiliary_host_output = TimingMetadataOutput(
        clock, clock.now_ns, result.auxiliary_host_output, token_events
    )


class C_TimingOutputStreamerHook(BaseHook):
    HOOK_CLASS_NAME = "SchedulerOutputStreamer"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.output_streamer"

    @classmethod
    def hook(cls, target):
        original = target.build_additional_customized_info

        @wraps(original)
        def build(self, reqs):
            info = original(self, reqs)
            if METADATA_KEY in info:
                raise ProtocolError(f"reserved response metadata key: {METADATA_KEY}")
            info[METADATA_KEY] = [
                getattr(req, _REQUEST_ATTRIBUTE, None) for req in reqs
            ]
            return info

        target.has_additional_customized_info = True
        target.build_additional_customized_info = build


class C_TimingTokenizerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"
    # Tokenizer and OpenAI serving classes are absent from scheduler workers.
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original = target.update_request_meta_info

        @wraps(original)
        def update(self, meta_info, state, customized_info, index, finish_reason):
            original(
                self,
                meta_info,
                state,
                {
                    key: value
                    for key, value in customized_info.items()
                    if key != METADATA_KEY
                },
                index,
                finish_reason,
            )
            snapshots = customized_info.get(METADATA_KEY)
            if snapshots is not None and snapshots[index] is not None:
                # Native token metadata is accumulated and sliced. A snapshot is
                # request-level data and must survive coalescing as a whole.
                meta_info[METADATA_KEY] = snapshots[index]

        target.update_request_meta_info = update


class C_TimingChatHook(BaseHook):
    HOOK_CLASS_NAME = "OpenAIServingChat"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.openai.serving_chat"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original = target._generate_stream_content

        @wraps(original)
        async def generate(self, content, *args, **kwargs):
            snapshot = content["meta_info"].get(METADATA_KEY)
            finish_reason = content["meta_info"].get("finish_reason") or {}
            if snapshot is None and finish_reason.get("type") != "abort":
                raise ProtocolError("coordinated chat response lacks timing metadata")
            if snapshot is not None:
                yield (
                    f"event: {METADATA_KEY}\n"
                    f"data: {msgspec.json.encode(snapshot).decode()}\n\n"
                )
            async for chunk in original(self, content, *args, **kwargs):
                yield chunk

        target._generate_stream_content = generate
