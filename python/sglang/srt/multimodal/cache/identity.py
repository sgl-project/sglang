"""Stable identities for multimodal inputs and processor artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

import numpy as np
import torch
import transformers
from PIL import Image

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

CONTENT_HASH_PREFIX = "sha256:"
_SHA256_HEX_LENGTH = 64
_MEDIA_ENVELOPE_FIELDS = frozenset(
    {"type", "format", "url", "image", "video", "audio", "content_hash"}
)


@runtime_checkable
class PreprocessFingerprintProvider(Protocol):
    """Explicit source for settings that can change processor artifacts."""

    def preprocess_fingerprint_payload(self) -> Any: ...


def parse_content_hash(value: Optional[str]) -> Optional[str]:
    """Validate and normalize a public content digest."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith(CONTENT_HASH_PREFIX):
        raise ValueError("content_hash must use the form 'sha256:<64 hex digits>'")
    digest = value[len(CONTENT_HASH_PREFIX) :]
    if len(digest) != _SHA256_HEX_LENGTH:
        raise ValueError("content_hash must contain exactly 64 SHA-256 hex digits")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("content_hash contains non-hexadecimal characters") from exc
    return CONTENT_HASH_PREFIX + digest.lower()


def _digest_bytes(payload: bytes) -> str:
    return CONTENT_HASH_PREFIX + hashlib.sha256(payload).hexdigest()


def _hash_parts(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return CONTENT_HASH_PREFIX + hasher.hexdigest()


@dataclass(frozen=True)
class MediaSnapshot:
    """An immutable-enough media snapshot paired with its strict identity."""

    data: Any
    content_digest: str
    size_bytes: int
    source: str


def _snapshot_pil(image: Image.Image) -> MediaSnapshot:
    snapshot = image.copy()
    snapshot.load()
    payload = snapshot.tobytes()
    palette = snapshot.palette.tobytes() if snapshot.palette is not None else b""
    palette_mode = (
        snapshot.palette.mode.encode() if snapshot.palette is not None else b""
    )
    transparency = snapshot.info.get("transparency")
    if transparency is None:
        transparency_payload = b"none"
    elif isinstance(transparency, bytes):
        transparency_payload = b"bytes:" + transparency
    else:
        transparency_payload = (
            f"{type(transparency).__name__}:{transparency!r}".encode()
        )
    digest = _hash_parts(
        b"pil",
        snapshot.mode.encode(),
        json.dumps(snapshot.size).encode(),
        palette_mode,
        palette,
        transparency_payload,
        payload,
    )
    return MediaSnapshot(snapshot, digest, len(payload), "pil")


def _snapshot_tensor(tensor: torch.Tensor) -> MediaSnapshot:
    snapshot = tensor.detach().to("cpu").contiguous().clone()
    payload = snapshot.view(torch.uint8).numpy().tobytes()
    digest = _hash_parts(
        b"torch",
        str(snapshot.dtype).encode(),
        json.dumps(list(snapshot.shape)).encode(),
        payload,
    )
    return MediaSnapshot(snapshot, digest, len(payload), "tensor")


def _snapshot_ndarray(array: np.ndarray) -> MediaSnapshot:
    snapshot = np.ascontiguousarray(array).copy()
    payload = snapshot.view(np.uint8).tobytes()
    digest = _hash_parts(
        b"numpy",
        snapshot.dtype.str.encode(),
        json.dumps(list(snapshot.shape)).encode(),
        payload,
    )
    return MediaSnapshot(snapshot, digest, len(payload), "ndarray")


def _read_media_bytes(media: str | bytes) -> bytes:
    if isinstance(media, bytes):
        return bytes(media)

    from sglang.srt.utils import get_image_bytes, image_extension_names

    if media.startswith("file://"):
        media = unquote(urlparse(media).path)
    elif media.startswith(("http://", "https://", "data:")):
        return get_image_bytes(media)
    # ``load_image`` accepts relative local paths only by image extension.
    # Match that contract instead of probing arbitrary base64 as a filename.
    if media.lower().endswith(image_extension_names) and Path(media).is_file():
        return Path(media).read_bytes()
    return get_image_bytes(media)


def snapshot_media(media: Any) -> MediaSnapshot:
    """Snapshot media and hash exactly what will be handed to the decoder.

    Paths and URLs are deliberately not identities. They are resolved to bytes
    on every untrusted lookup, so changing their contents produces a cache miss.
    """
    from sglang.srt.utils import ImageData

    if isinstance(media, ImageData):
        media = media.url
    elif isinstance(media, Mapping) and "format" not in media:
        if "url" in media:
            media = media["url"]
        elif "image" in media:
            media = media["image"]

    if isinstance(media, (str, bytes)):
        payload = _read_media_bytes(media)
        return MediaSnapshot(payload, _digest_bytes(payload), len(payload), "bytes")
    if isinstance(media, Image.Image):
        return _snapshot_pil(media)
    if isinstance(media, torch.Tensor):
        return _snapshot_tensor(media)
    if isinstance(media, np.ndarray):
        return _snapshot_ndarray(media)
    raise TypeError(f"Unsupported media identity input: {type(media).__name__}")


def media_preprocess_kwargs(
    source: Any, *, defaults: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Conservatively capture per-request options that can affect an artifact.

    Unknown options are included instead of allow-listed. This may create a safe
    false miss for a metadata-only option, but it prevents a new model option
    from silently creating a false cache hit.
    """
    defaults = defaults or {}
    if dataclasses.is_dataclass(source):
        values = {
            field.name: value
            for field, value in zip(
                dataclasses.fields(source), dataclasses.astuple(source)
            )
            if field.name not in _MEDIA_ENVELOPE_FIELDS
        }
    elif isinstance(source, Mapping):
        values = {
            key: value
            for key, value in source.items()
            if key not in _MEDIA_ENVELOPE_FIELDS
        }
    else:
        return {}

    result = {}
    for key, value in values.items():
        if value is None or (isinstance(value, Mapping) and not value):
            continue
        if key in defaults and _canonicalize(value) == _canonicalize(defaults[key]):
            continue
        result[key] = value
    return result


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonicalize(value: Any) -> Any:
    """Encode cache-key inputs without collapsing distinct Python values."""
    if dataclasses.is_dataclass(value):
        return {
            "type": "dataclass",
            "class": _qualified_type_name(value),
            "fields": [
                [field.name, _canonicalize(field_value)]
                for field, field_value in zip(
                    dataclasses.fields(value), dataclasses.astuple(value)
                )
            ],
        }
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": _qualified_type_name(value),
            "value": _canonicalize(value.value),
        }
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float64", "bits": struct.pack("!d", value).hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "value": bytes(value).hex()}
    if isinstance(value, torch.dtype):
        return {"type": "torch_dtype", "value": str(value)}
    if isinstance(value, torch.Tensor):
        snapshot = value.detach().to("cpu").contiguous()
        payload = snapshot.view(torch.uint8).numpy().tobytes()
        return {
            "type": "torch_tensor",
            "dtype": str(snapshot.dtype),
            "shape": list(snapshot.shape),
            "digest": _digest_bytes(payload),
        }
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return {
            "type": "numpy_scalar",
            "dtype": scalar.dtype.str,
            "value": scalar.tobytes().hex(),
        }
    if isinstance(value, np.ndarray):
        snapshot = np.ascontiguousarray(value)
        return {
            "type": "numpy_array",
            "dtype": snapshot.dtype.str,
            "shape": list(snapshot.shape),
            "digest": _digest_bytes(snapshot.view(np.uint8).tobytes()),
        }
    if isinstance(value, Mapping):
        items = [
            [_canonicalize(key), _canonicalize(item)] for key, item in value.items()
        ]
        items.sort(key=lambda pair: _canonical_sort_key(pair[0]))
        return {
            "type": "mapping",
            "items": items,
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        items.sort(key=_canonical_sort_key)
        return {
            "type": "frozenset" if isinstance(value, frozenset) else "set",
            "items": items,
        }
    raise ValueError(
        "Unsupported value in multimodal cache identity: "
        f"{_qualified_type_name(value)}"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":")
    ).encode()


def build_artifact_key(
    content_digest: str,
    *,
    modality: str,
    processor_fingerprint: str,
    preprocess_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the cache key for a processor artifact."""
    content_digest = parse_content_hash(content_digest)
    payload = {
        "content_digest": content_digest,
        "modality": modality,
        "processor_fingerprint": processor_fingerprint,
        "preprocess_kwargs": preprocess_kwargs or {},
    }
    return _digest_bytes(_canonical_json(payload))


def resolve_multimodal_item_hash(
    *,
    existing_hash: Optional[int] = None,
    feature: Any = None,
    precomputed_embeddings: Any = None,
    namespace: Optional[str] = None,
) -> int:
    """Unified helper for resolving a hash for MultimodalDataItem cache, optionally scoped to an artifact identity.

    Args:
        namespace: Optional SHA-256 identity covering every input that can change the preprocessing result.
            It scopes the feature hash so downstream caches cannot reuse embeddings across different preprocessing settings.
    """
    from sglang.srt.environ import envs

    if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
        import uuid

        item_hash = uuid.uuid4().int
    elif existing_hash is not None:
        # if exists, reuse
        item_hash = existing_hash
    else:
        # hash from feature
        from sglang.srt.managers.mm_utils import hash_feature

        value = feature if feature is not None else precomputed_embeddings
        item_hash = hash_feature(value)

    if namespace is None:
        return item_hash

    if isinstance(item_hash, bool) or not isinstance(item_hash, int) or item_hash < 0:
        raise ValueError("item hash must be a non-negative integer")
    namespace = parse_content_hash(namespace)
    assert namespace is not None
    hash_bytes = item_hash.to_bytes(
        max(1, (item_hash.bit_length() + 7) // 8), byteorder="big", signed=False
    )
    digest = _hash_parts(
        b"multimodal-feature-v1",
        bytes.fromhex(namespace[len(CONTENT_HASH_PREFIX) :]),
        hash_bytes,
    )
    return int.from_bytes(
        bytes.fromhex(digest[len(CONTENT_HASH_PREFIX) :])[:8],
        byteorder="big",
        signed=False,
    )


def build_processor_fingerprint(
    processor: Any,
    hf_config: Any,
    server_args: ServerArgs,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fingerprint preprocessing choices that can change processor output."""
    processor_payload = (
        processor.preprocess_fingerprint_payload()
        if isinstance(processor, PreprocessFingerprintProvider)
        else {}
    )
    hf_payload = hf_config.to_dict()
    payload = {
        "transformers": transformers.__version__,
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "model_type": hf_payload.get("model_type"),
        "architectures": hf_payload.get("architectures"),
        "model_revision": server_args.revision,
        "processor_revision": server_args.revision,
        "disable_fast_image_processor": server_args.disable_fast_image_processor,
        "mm_process_config": server_args.mm_process_config or {},
        "processor": processor_payload,
        "extra": extra or {},
    }
    return _digest_bytes(_canonical_json(payload))
