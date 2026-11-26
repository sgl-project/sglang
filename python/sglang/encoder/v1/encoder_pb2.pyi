from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EncodeRequest(_message.Message):
    __slots__ = ("mm_items", "req_id", "num_parts", "part_idx", "prefill_host", "embedding_port")
    MM_ITEMS_FIELD_NUMBER: _ClassVar[int]
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTS_FIELD_NUMBER: _ClassVar[int]
    PART_IDX_FIELD_NUMBER: _ClassVar[int]
    PREFILL_HOST_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_PORT_FIELD_NUMBER: _ClassVar[int]
    mm_items: _containers.RepeatedScalarFieldContainer[str]
    req_id: str
    num_parts: int
    part_idx: int
    prefill_host: str
    embedding_port: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, mm_items: _Optional[_Iterable[str]] = ..., req_id: _Optional[str] = ..., num_parts: _Optional[int] = ..., part_idx: _Optional[int] = ..., prefill_host: _Optional[str] = ..., embedding_port: _Optional[_Iterable[int]] = ...) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ("embedding_size", "embedding_len", "embedding_dim")
    EMBEDDING_SIZE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_LEN_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    embedding_size: int
    embedding_len: int
    embedding_dim: int
    def __init__(self, embedding_size: _Optional[int] = ..., embedding_len: _Optional[int] = ..., embedding_dim: _Optional[int] = ...) -> None: ...

class SendRequest(_message.Message):
    __slots__ = ("req_id", "prefill_host", "embedding_port", "session_id", "buffer_address")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    PREFILL_HOST_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_PORT_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    BUFFER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    prefill_host: str
    embedding_port: int
    session_id: str
    buffer_address: int
    def __init__(self, req_id: _Optional[str] = ..., prefill_host: _Optional[str] = ..., embedding_port: _Optional[int] = ..., session_id: _Optional[str] = ..., buffer_address: _Optional[int] = ...) -> None: ...

class SendResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SchedulerReceiveUrlRequest(_message.Message):
    __slots__ = ("req_id", "receive_url", "receive_count")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVE_URL_FIELD_NUMBER: _ClassVar[int]
    RECEIVE_COUNT_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    receive_url: str
    receive_count: int
    def __init__(self, req_id: _Optional[str] = ..., receive_url: _Optional[str] = ..., receive_count: _Optional[int] = ...) -> None: ...

class SchedulerReceiveUrlResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
