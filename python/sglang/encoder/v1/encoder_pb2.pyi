from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetEncoderInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetEncoderInfoResponse(_message.Message):
    __slots__ = ("model_path", "tokenizer_path", "served_model_name", "supports_staged_video", "staged_media_policy")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_PATH_FIELD_NUMBER: _ClassVar[int]
    SERVED_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STAGED_VIDEO_FIELD_NUMBER: _ClassVar[int]
    STAGED_MEDIA_POLICY_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    tokenizer_path: str
    served_model_name: str
    supports_staged_video: bool
    staged_media_policy: StagedMediaPolicy
    def __init__(self, model_path: _Optional[str] = ..., tokenizer_path: _Optional[str] = ..., served_model_name: _Optional[str] = ..., supports_staged_video: _Optional[bool] = ..., staged_media_policy: _Optional[_Union[StagedMediaPolicy, _Mapping]] = ...) -> None: ...

class StagedMediaPolicy(_message.Message):
    __slots__ = ("allowed_domains", "max_bytes", "timeout_secs")
    ALLOWED_DOMAINS_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECS_FIELD_NUMBER: _ClassVar[int]
    allowed_domains: _containers.RepeatedScalarFieldContainer[str]
    max_bytes: int
    timeout_secs: float
    def __init__(self, allowed_domains: _Optional[_Iterable[str]] = ..., max_bytes: _Optional[int] = ..., timeout_secs: _Optional[float] = ...) -> None: ...

class StageRequest(_message.Message):
    __slots__ = ("request_id", "mm_items", "video_urls")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MM_ITEMS_FIELD_NUMBER: _ClassVar[int]
    VIDEO_URLS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    mm_items: _containers.RepeatedScalarFieldContainer[str]
    video_urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_id: _Optional[str] = ..., mm_items: _Optional[_Iterable[str]] = ..., video_urls: _Optional[_Iterable[str]] = ...) -> None: ...

class ImageLayout(_message.Message):
    __slots__ = ("t", "h", "w", "identity")
    T_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    t: int
    h: int
    w: int
    identity: str
    def __init__(self, t: _Optional[int] = ..., h: _Optional[int] = ..., w: _Optional[int] = ..., identity: _Optional[str] = ...) -> None: ...

class VideoLayout(_message.Message):
    __slots__ = ("t", "h", "w", "identities", "timestamps", "second_per_grid_ts")
    T_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    IDENTITIES_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    SECOND_PER_GRID_TS_FIELD_NUMBER: _ClassVar[int]
    t: int
    h: int
    w: int
    identities: _containers.RepeatedScalarFieldContainer[str]
    timestamps: _containers.RepeatedScalarFieldContainer[float]
    second_per_grid_ts: float
    def __init__(self, t: _Optional[int] = ..., h: _Optional[int] = ..., w: _Optional[int] = ..., identities: _Optional[_Iterable[str]] = ..., timestamps: _Optional[_Iterable[float]] = ..., second_per_grid_ts: _Optional[float] = ...) -> None: ...

class StagedImageTransfer(_message.Message):
    __slots__ = ("version", "request_id", "transfer_id", "part_id", "part_idx", "num_parts", "image_count", "images", "videos")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_ID_FIELD_NUMBER: _ClassVar[int]
    PART_ID_FIELD_NUMBER: _ClassVar[int]
    PART_IDX_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    VIDEOS_FIELD_NUMBER: _ClassVar[int]
    version: int
    request_id: str
    transfer_id: str
    part_id: str
    part_idx: int
    num_parts: int
    image_count: int
    images: _containers.RepeatedCompositeFieldContainer[ImageLayout]
    videos: _containers.RepeatedCompositeFieldContainer[VideoLayout]
    def __init__(self, version: _Optional[int] = ..., request_id: _Optional[str] = ..., transfer_id: _Optional[str] = ..., part_id: _Optional[str] = ..., part_idx: _Optional[int] = ..., num_parts: _Optional[int] = ..., image_count: _Optional[int] = ..., images: _Optional[_Iterable[_Union[ImageLayout, _Mapping]]] = ..., videos: _Optional[_Iterable[_Union[VideoLayout, _Mapping]]] = ...) -> None: ...

class StageResponse(_message.Message):
    __slots__ = ("transfer",)
    TRANSFER_FIELD_NUMBER: _ClassVar[int]
    transfer: StagedImageTransfer
    def __init__(self, transfer: _Optional[_Union[StagedImageTransfer, _Mapping]] = ...) -> None: ...

class DescribeRequest(_message.Message):
    __slots__ = ("transfer",)
    TRANSFER_FIELD_NUMBER: _ClassVar[int]
    transfer: StagedImageTransfer
    def __init__(self, transfer: _Optional[_Union[StagedImageTransfer, _Mapping]] = ...) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("shape", "dtype")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class DeliverRequest(_message.Message):
    __slots__ = ("transfer", "receive_url", "nixl_target")
    TRANSFER_FIELD_NUMBER: _ClassVar[int]
    RECEIVE_URL_FIELD_NUMBER: _ClassVar[int]
    NIXL_TARGET_FIELD_NUMBER: _ClassVar[int]
    transfer: StagedImageTransfer
    receive_url: str
    nixl_target: bytes
    def __init__(self, transfer: _Optional[_Union[StagedImageTransfer, _Mapping]] = ..., receive_url: _Optional[str] = ..., nixl_target: _Optional[bytes] = ...) -> None: ...

class DeliverResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReleaseRequest(_message.Message):
    __slots__ = ("transfer_id",)
    TRANSFER_ID_FIELD_NUMBER: _ClassVar[int]
    transfer_id: str
    def __init__(self, transfer_id: _Optional[str] = ...) -> None: ...

class ReleaseResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

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
