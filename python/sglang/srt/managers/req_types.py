"""Request data types shared by io_struct and the scheduler.

Kept free of scheduler, memory-pool, and model imports, so request and IPC code
can use them without loading the runtime.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import (
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.multimodal.transport.cuda_ipc import (
    CUDA_IPC_FEATURE_COPY_EVENT_KEY,
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    RETAINED_CUDA_IPC_FEATURE_PROXY_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.utils import flatten_nested_list

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000
_MM_HASH_MASK = (1 << 64) - 1
_MM_IDENTITY_BYTES = 32
_MM_WIDE_PAD_TAG = b"sglang-mm-wide-pad-v1"
# Keep MM_PAD_SHIFT_VALUE + sentinel below 2**63 so it fits an int64 token ID.
_MM_WIDE_PAD_MASK = (1 << 62) - 1

logger = logging.getLogger(__name__)


ReturnHiddenStatesMode = Union[bool, Literal["last"]]
SamplingLogprobsMode = Literal["selected", "support"]


def get_return_hidden_states_mode(
    return_hidden_states: ReturnHiddenStatesMode,
) -> CaptureHiddenMode:
    if return_hidden_states is True:
        return CaptureHiddenMode.FULL
    if return_hidden_states == "last":
        return CaptureHiddenMode.LAST
    if return_hidden_states is False:
        return CaptureHiddenMode.NULL
    raise ValueError(
        "return_hidden_states must be a boolean or the string literal 'last'."
    )


def get_request_return_hidden_states_mode(
    return_hidden_states: Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode],
) -> CaptureHiddenMode:
    if isinstance(return_hidden_states, list):
        return max(
            (get_return_hidden_states_mode(mode) for mode in return_hidden_states),
            default=CaptureHiddenMode.NULL,
        )
    return get_return_hidden_states_mode(return_hidden_states)


def _compute_pad_values(identity: bytes) -> Tuple[int, ...]:
    """derive shared wide sentinels from an item hash or full identity digest"""
    digest = hashlib.sha256(_MM_WIDE_PAD_TAG + identity).digest()
    # prefix matching can stop after one sentinel, so each must retain wide bits
    return tuple(
        MM_PAD_SHIFT_VALUE
        + (int.from_bytes(digest[i : i + 8], byteorder="big") & _MM_WIDE_PAD_MASK)
        for i in range(0, _MM_IDENTITY_BYTES, 8)
    )


# Any processor-owned value. Msgpack IPC pickles the ones it would not decode
# back unchanged (tuples, dataclasses, transport proxies); see map_object_fields.
MultimodalDataValue: TypeAlias = object


class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has a full identity for embedding caching and batch deduplication.
    KV caching additionally scopes the request by its ordered media identities.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: Optional[int] = None
    pad_value: Optional[int] = None
    offsets: Optional[List[Tuple[int, int]]] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Optional[MultimodalDataValue] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[MultimodalDataValue] = None
    # Keep precomputed_embeddings on GPU after use (EPD pool/GPU receive path)
    keep_device_embedding: bool = False

    # Processor-owned tensors/arrays/scalars/transports. msgspec rejects a
    # precise union with multiple custom types, but accepts Ext-decoded values
    # under object.
    model_specific_data: Dict[str, MultimodalDataValue] = msgspec.field(
        default_factory=dict
    )

    # Full identity is authoritative for every cache; hash is legacy wire metadata.
    # pad_value remains the first sentinel for scalar-padding consumers
    identity: Optional[bytes] = None
    pad_values: Optional[Tuple[int, ...]] = None
    processor_fingerprint: Optional[str] = None
    # Caller routing label (routing metadata only; never part of identity).
    routing_label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.hash is not None and not 0 <= self.hash <= _MM_HASH_MASK:
            # retain full artifact fingerprints before narrowing the wire field
            self.set_hash(self.hash)

    def __getattr__(self, name: str) -> MultimodalDataValue:
        if name in self.model_specific_data:
            return self.model_specific_data[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: MultimodalDataValue) -> None:
        if name in self.__struct_fields__:
            if name == "hash" and isinstance(value, int):
                value &= _MM_HASH_MASK
            msgspec.Struct.__setattr__(self, name, value)
        else:
            self.model_specific_data[name] = value

    def __setitem__(self, key: str, value: MultimodalDataValue) -> None:
        setattr(self, key, value)

    def set(self, key: str, value: MultimodalDataValue) -> None:
        self.__setitem__(key, value)

    def set_hash(self, hash_value: int) -> None:
        self.hash = hash_value
        # Preserve the full incoming value before the legacy field masks it.
        self.set_identity(self._input_identity(hash_value))
        self.hash = hash_value

    def set_identity(self, identity: bytes) -> None:
        """Adopt a full 32-byte digest and derive four 62-bit pad sentinels.

        ``hash`` keeps a compatibility scalar, never the embedding-cache key.
        The radix key also binds the full identity in the request namespace.
        """
        if len(identity) != _MM_IDENTITY_BYTES:
            raise ValueError(
                f"multimodal identity must be {_MM_IDENTITY_BYTES} bytes, "
                f"got {len(identity)}"
            )
        self.identity = bytes(identity)
        self.hash = int.from_bytes(identity[:8], byteorder="big", signed=False)
        self.pad_values = _compute_pad_values(self.identity)
        self.pad_value = self.pad_values[0]

    @property
    def cache_key(self) -> bytes:
        """The full identity, shared by embedding caching and batch dedup."""
        self.set_pad_value()
        return self.identity

    def clear_identity(self) -> None:
        """Invalidate all derived state after slicing or replacing a feature."""
        self.hash = self.identity = self.pad_value = self.pad_values = None

    def bind_processor_fingerprint(self, fingerprint: str) -> bool:
        """Scope caches to the serving processor, once per request-owned item."""
        from sglang.srt.multimodal.cache.digest import structured_digest

        if self.processor_fingerprint == fingerprint:
            return False
        if self.processor_fingerprint is not None:
            raise ValueError("Multimodal item belongs to a different processor")
        self.processor_fingerprint = fingerprint
        if self.identity is not None:
            self.set_identity(
                structured_digest(("processor", fingerprint, self.identity))
            )
        return True

    def padding_values(self) -> Tuple[int, ...]:
        return self.pad_values or (self.pad_value,)

    def padding_sequence(self, length: int) -> List[int]:
        values = self.padding_values()
        repeats, remainder = divmod(length, len(values))
        return list(values) * repeats + list(values[:remainder])

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        if self.identity is not None:
            if self.pad_value is None:
                self.set_identity(self.identity)
            return

        hash_hint = self.hash
        self.set_identity(self._input_identity(hash_hint))
        if hash_hint is not None:
            self.hash = hash_hint

    def _input_identity(self, hash_hint: Optional[int]) -> bytes:
        from sglang.srt.multimodal.cache.identity import (
            resolve_multimodal_item_identity,
        )

        return resolve_multimodal_item_identity(
            existing_hash=hash_hint,
            feature=self.feature,
            precomputed_embeddings=self.precomputed_embeddings,
            modality=self.modality.name,
            input_format=self.format.name,
            metadata=self.model_specific_data,
            namespace=self.processor_fingerprint,
        )

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int, ipc_consumer_count: int = 1):
        """materialize cuda ipc proxy tensors in-place on target_device"""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, ipc_consumer_count
            )
            if consumer_count == 1:
                self.feature = self.feature.reconstruct_on_target_device(target_device)
            else:
                self.feature = self.feature.reconstruct_on_target_device(
                    target_device, consumer_count=consumer_count
                )
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if extra_key == RETAINED_CUDA_IPC_FEATURE_PROXY_KEY:
                continue
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data

    def can_defer_cuda_ipc_feature_reconstruction(self) -> bool:
        """Whether a DP-aware model will materialize this feature lazily.

        Hashing and pad-value generation must already have completed on the
        tokenizer worker.  Any additional IPC proxy would still need eager
        reconstruction, so keep the narrow fast path feature-only.
        """
        return (
            self.model_specific_data.get(
                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
            )
            and self.hash is not None
            and self.pad_value is not None
            and isinstance(self.feature, CudaIpcTensorTransportProxy)
            and not isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            and not any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def acknowledge_deferred_cuda_ipc_feature(self, consumer_count: int = 1):
        """Release a lazy IPC feature when an embedding-cache hit skips ViT."""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, consumer_count
            )
            self.feature.acknowledge_consumption(consumer_count)

    def release_transport_proxies(self, consumer_count: int = 1) -> None:
        """Best-effort release of proxies left by an abandoned request."""
        retained_proxy = self.model_specific_data.pop(
            RETAINED_CUDA_IPC_FEATURE_PROXY_KEY, None
        )
        self.model_specific_data.pop(CUDA_IPC_FEATURE_COPY_EVENT_KEY, None)
        values = [self.feature, self.precomputed_embeddings, retained_proxy]
        values.extend(self.model_specific_data.values())
        for value in values:
            if not isinstance(value, CudaIpcTensorTransportProxy):
                continue
            count = self._resolve_transport_consumer_count(value, consumer_count)
            try:
                value.release_without_reconstruction(count)
            except Exception:
                logger.warning(
                    "Failed to release an abandoned multimodal transport proxy",
                    exc_info=True,
                )

    @staticmethod
    def _resolve_transport_consumer_count(proxy, requested_count: int) -> int:
        """Clamp a group acknowledgement to the proxy's actual consumer set."""
        proxy_count = getattr(
            proxy,
            "total_consumer_count",
            getattr(proxy, "consumer_count", requested_count),
        )
        return min(requested_count, proxy_count)


class MultimodalProcessorOutput(
    msgspec.Struct, kw_only=True, dict=True, array_like=True, weakref=True
):
    """Raw output from multimodal processors before scheduler-side preparation (pad, hash).

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Preprocessed inputs may
    already carry ``pad_value`` and ``hash`` to avoid hashing the same tensor once
    per scheduler TP rank.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    # A ClassVar, so it never goes on the wire. A processor's subclass sets it to
    # have wrap_shm_features also move large model_specific_data tensors to SHM.
    shm_model_specific_data: ClassVar[bool] = False

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    def map_object_fields(
        self, transform: Callable[[list], list]
    ) -> MultimodalProcessorOutput:
        """Copy with transform applied to every item's object-typed values.

        transform sees the values of all items in one list, in a fixed order,
        and returning that list unchanged returns self.
        """
        values = [
            value
            for item in self.mm_items
            for value in (
                item.feature,
                item.precomputed_embeddings,
                *item.model_specific_data.values(),
            )
        ]
        new_values = transform(values)
        if new_values is values:
            return self
        new_values = iter(new_values)
        mm_items = [
            msgspec.structs.replace(
                item,
                feature=next(new_values),
                precomputed_embeddings=next(new_values),
                model_specific_data={
                    key: next(new_values) for key in item.model_specific_data
                },
            )
            for item in self.mm_items
        ]
        return msgspec.structs.replace(self, mm_items=mm_items)

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        """pad the input_ids with mm_items if it's not already padded"""
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = item.padding_sequence(
                    end - start + 1
                )
        return padded_input_ids

    @staticmethod
    def build_token_modalities(
        input_ids, mm_items: List[MultimodalDataItem]
    ) -> Optional[List[int]]:
        """Build the pre-padding token modality map from item offsets."""
        if input_ids is None or not mm_items:
            return None
        if isinstance(input_ids, torch.Tensor):
            num_tokens = input_ids.numel()
        else:
            num_tokens = len(flatten_nested_list(input_ids))
        token_modalities = [0] * num_tokens
        for item in mm_items:
            if not item.offsets:
                continue
            modality = item.modality.value
            for start, end in item.offsets:
                if start < 0 or end < start or end >= num_tokens:
                    raise ValueError(
                        "Invalid multimodal token offsets: "
                        f"offset=({start}, {end}), num_tokens={num_tokens}"
                    )
                if any(token_modalities[index] for index in range(start, end + 1)):
                    raise ValueError(
                        f"Overlapping multimodal token offsets at ({start}, {end})"
                    )
                token_modalities[start : end + 1] = [modality] * (end - start + 1)
        return token_modalities


class BaseFinishReason:
    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    padded_input_ids: Optional[List[int]] = None
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None
    token_modalities: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None
    mrope_position_delta_device_cache: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def get_mrope_position_delta_on_device(self, device) -> torch.Tensor:
        # Rebuilding this from Python ints every decode step costs a blocking H2D copy.
        if self.mrope_position_delta_device_cache is None:
            self.mrope_position_delta_device_cache = self.mrope_position_delta.reshape(
                1
            ).to(device=device, dtype=torch.int64)
        return self.mrope_position_delta_device_cache

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        for item in self.mm_items:
            try:
                # Release both deferred features that were never used and
                # borrowed features retained for possible re-prefill.
                item.release_transport_proxies()
            except Exception:
                logger.warning(
                    "Failed to release an unused multimodal feature transport",
                    exc_info=True,
                )
            finally:
                item.feature = None

    @staticmethod
    def from_processor_output(
        obj: MultimodalProcessorOutput, *, requires_mm_token_modalities: bool = False
    ):
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]

        # try reconstructing from cuda-ipc
        reconstruct_device = None
        try:
            for mm_item in mm_items:
                if (
                    mm_item.has_cuda_ipc_proxy()
                    and not mm_item.can_defer_cuda_ipc_feature_reconstruction()
                ):
                    if reconstruct_device is None:
                        reconstruct_device = torch.cuda.current_device()
                    mm_item.reconstruct(reconstruct_device)
        except BaseException:
            for mm_item in mm_items:
                mm_item.release_transport_proxies()
            raise

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        token_modalities = (
            MultimodalProcessorOutput.build_token_modalities(obj.input_ids, mm_items)
            if requires_mm_token_modalities
            else None
        )

        for item in mm_items:
            item.set_pad_value()

        from sglang.srt.mem_cache.events import register_mm_routing_labels

        register_mm_routing_labels(mm_items)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            padded_input_ids=obj.padded_input_ids,
            token_modalities=token_modalities,
        )
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(mm_inputs, arg, val)

        return mm_inputs

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def compute_mm_token_counts(self) -> Tuple[int, int, int]:
        """Count prompt tokens consumed by each modality (image, audio, video).

        A modality's token count is the total span covered by its items'
        offsets. Returns a (image_tokens, audio_tokens, video_tokens) tuple.
        """
        image_tokens = audio_tokens = video_tokens = 0
        for item in self.mm_items:
            if not item.offsets:
                continue
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            if item.is_image():
                image_tokens += num_tokens
            elif item.is_audio():
                audio_tokens += num_tokens
            elif item.is_video():
                video_tokens += num_tokens
        return image_tokens, audio_tokens, video_tokens

    def merge(self, other: MultimodalInputs):

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        if other.token_modalities is not None:
            if self.token_modalities is None:
                self.token_modalities = list(other.token_modalities)
            else:
                self.token_modalities += other.token_modalities

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact
