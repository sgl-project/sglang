"""Pydantic event schemas mirroring openai-python's types/realtime/.

Schema only. Business logic lives in session.py.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

# Wire codes for client matching. Multi-use only; single-use codes are
# inlined at call sites.
ERROR_TYPE_INVALID_REQUEST = "invalid_request_error"
ERROR_TYPE_SERVER = "server_error"

CODE_NOT_SUPPORTED = "not_supported"
CODE_INVALID_PAYLOAD = "invalid_payload"
CODE_INVALID_STATE = "invalid_state"
CODE_INVALID_VALUE = "invalid_value"
CODE_INFERENCE_FAILED = "inference_failed"


# OpenAI canonical ID format is {prefix}_{24-char hex}. The prefixes
# (event, item, sess) are what the OpenAI Realtime server emits
def new_event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:24]}"


# OpenAI's format is {type, rate?} where type is audio/pcm, audio/pcmu,
# or audio/pcma. We accept only audio/pcm. The rate whitelist (16k/24k/48k)
# is an sglang extension and is checked in the handler, not here.


class AudioInputFormatPCM(BaseModel):
    type: Literal["audio/pcm"]
    rate: Optional[int] = None


class AudioInputFormatPCMU(BaseModel):
    type: Literal["audio/pcmu"]


class AudioInputFormatPCMA(BaseModel):
    type: Literal["audio/pcma"]


AudioInputFormat = Annotated[
    Union[AudioInputFormatPCM, AudioInputFormatPCMU, AudioInputFormatPCMA],
    Field(discriminator="type"),
]


class TranscriptionConfig(BaseModel):
    model: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None  # rejected non-null in handler


class AudioInputConfig(BaseModel):
    """Sample rate lives inside `format.rate`, not as a sibling field."""

    format: Optional[AudioInputFormat] = None
    transcription: Optional[TranscriptionConfig] = None
    noise_reduction: Optional[Any] = None  # rejected non-null in handler
    turn_detection: Optional[Any] = None  # rejected non-null in handler


class AudioConfig(BaseModel):
    input: Optional[AudioInputConfig] = None


class TranscriptionSessionConfig(BaseModel):
    type: Literal["transcription"]
    audio: Optional[AudioConfig] = None
    include: Optional[List[str]] = None  # logged and dropped in handler

    model_config = ConfigDict(extra="allow")


# `extra="allow"` on every client event so OpenAI can add fields on existing
# event types without us 400-ing the client.


class SessionUpdate(BaseModel):
    type: Literal["session.update"]
    event_id: Optional[str] = None
    session: TranscriptionSessionConfig

    model_config = ConfigDict(extra="allow")


class InputAudioBufferAppend(BaseModel):
    type: Literal["input_audio_buffer.append"]
    event_id: Optional[str] = None
    audio: str  # base64 PCM16 LE

    model_config = ConfigDict(extra="allow")


class InputAudioBufferCommit(BaseModel):
    type: Literal["input_audio_buffer.commit"]
    event_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class InputAudioBufferClear(BaseModel):
    type: Literal["input_audio_buffer.clear"]
    event_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class _ServerEventBase(BaseModel):
    event_id: str = Field(default_factory=new_event_id)


class SessionInfo(BaseModel):
    """Stainless types omit `id` on this Session union, but the real OpenAI
    server probably sends `id` and `object`. Emit them anyway; `extra="allow"`
    makes false positives free.
    """

    id: str
    object: Literal["realtime.transcription_session"] = "realtime.transcription_session"
    type: Literal["transcription"] = "transcription"
    audio: Dict[str, Any]


class SessionCreated(_ServerEventBase):
    type: Literal["session.created"] = "session.created"
    session: SessionInfo


class SessionUpdated(_ServerEventBase):
    type: Literal["session.updated"] = "session.updated"
    session: SessionInfo


class InputAudioBufferCommitted(_ServerEventBase):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    item_id: str
    previous_item_id: Optional[str] = None  # explicit null on first commit, not omitted


class InputAudioBufferCleared(_ServerEventBase):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class InputAudioContent(BaseModel):
    type: Literal["input_audio"] = "input_audio"
    transcript: str


class Item(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["user"] = "user"
    status: Literal["completed"] = "completed"
    content: List[InputAudioContent]


class ConversationItemCreated(_ServerEventBase):
    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: Optional[str] = None
    item: Item


class TranscriptionDelta(_ServerEventBase):
    """One event per word, not per chunk."""

    type: Literal["conversation.item.input_audio_transcription.delta"] = (
        "conversation.item.input_audio_transcription.delta"
    )
    item_id: str
    content_index: int = 0
    delta: str


class UsageDuration(BaseModel):
    """sglang emits the duration variant; OpenAI's Usage union also has a
    tokens variant we don't track."""

    type: Literal["duration"] = "duration"
    seconds: float


class TranscriptionCompleted(_ServerEventBase):
    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str
    content_index: int = 0
    transcript: str
    usage: UsageDuration  # capture pcm duration before `_roll_item()` clears the buffer


class TranscriptionFailedError(BaseModel):
    type: Literal["server_error"] = "server_error"
    code: Literal["inference_failed"] = "inference_failed"
    message: str


class TranscriptionFailed(_ServerEventBase):
    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    item_id: str
    content_index: int = 0
    error: TranscriptionFailedError


class ErrorDetails(BaseModel):
    type: str  # ERROR_TYPE_*
    code: str  # CODE_*
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None  # echoed client event_id when known


class ErrorEvent(_ServerEventBase):
    type: Literal["error"] = "error"
    error: ErrorDetails


def format_error_envelope(
    code: str,
    message: str,
    *,
    error_type: str = ERROR_TYPE_INVALID_REQUEST,
    param: Optional[str] = None,
    client_event_id: Optional[str] = None,
) -> str:
    """Used by `_send_error` (echoes client event_id) and handler.py's
    pre-session reject path (e.g. too_many_sessions)."""
    return ErrorEvent(
        error=ErrorDetails(
            type=error_type,
            code=code,
            message=message,
            param=param,
            event_id=client_event_id,
        )
    ).model_dump_json()
