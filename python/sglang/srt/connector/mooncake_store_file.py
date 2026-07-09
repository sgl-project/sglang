# SPDX-License-Identifier: Apache-2.0

import fnmatch
import itertools
import json
import logging
import math
import os
import struct
from typing import Generator, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector import BaseFileConnector

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "~/.config/mooncake/weight_store.json"
DEFAULT_METADATA_DIR = "/tmp/mooncake-sglang-models"
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"]
_CLIENT_COUNTER = itertools.count(1)
SAFETENSORS_DTYPES = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
}


def _matches(path: str, patterns: Optional[list[str]]) -> bool:
    return bool(patterns) and any(
        fnmatch.fnmatch(path, pattern) for pattern in patterns
    )


def _parse_size(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    units = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    upper = text.upper()
    for suffix, multiplier in units.items():
        if upper.endswith(suffix):
            return int(float(upper[: -len(suffix)].strip()) * multiplier)
    raise ValueError(f"invalid size: {value!r}")


def _checkpoint_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "mooncake":
        raise ValueError(f"invalid Mooncake Store model URL: {url!r}")
    checkpoint_id = parsed.netloc + parsed.path
    checkpoint_id = checkpoint_id.strip("/")
    if not checkpoint_id:
        raise ValueError(f"Mooncake Store model URL has no checkpoint id: {url!r}")
    return checkpoint_id


def _load_store_config(path: Optional[str]) -> dict:
    config_path = (
        path or os.environ.get("MOONCAKE_WEIGHT_STORE_CONFIG") or DEFAULT_CONFIG_PATH
    )
    with open(os.path.expanduser(config_path), encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = config_path
    return config


def _unique_local_hostname(local_hostname: str) -> str:
    try:
        name, port_text = local_hostname.rsplit(":", 1)
        port = int(port_text)
    except ValueError:
        name = local_hostname
        port = 12346

    counter = next(_CLIENT_COUNTER)
    unique_port = port + counter
    if unique_port > 65535:
        unique_port = 10000 + (unique_port % 50000)
    return f"{name}-{os.getpid()}-{counter}:{unique_port}"


def _create_store(config: dict):
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    configured_local_hostname = config.get("local_hostname", "sglang-client:12346")
    local_hostname = _unique_local_hostname(configured_local_hostname)
    metadata_server = config["metadata_server"]
    global_segment_size = _parse_size(config.get("global_segment_size", 0))
    local_buffer_size = _parse_size(config.get("local_buffer_size", "512MB"))
    protocol = config.get("protocol", "tcp")
    rdma_devices = config.get("rdma_devices", "")
    master_server_addr = config.get(
        "master_server_addr", config.get("master_server_address", "")
    )

    setup_config = {
        "local_hostname": local_hostname,
        "metadata_server": metadata_server,
        "global_segment_size": global_segment_size,
        "local_buffer_size": local_buffer_size,
        "protocol": protocol,
        "rdma_devices": rdma_devices,
        "master_server_addr": master_server_addr,
    }

    try:
        result = store.setup(
            local_hostname,
            metadata_server,
            global_segment_size,
            local_buffer_size,
            protocol,
            rdma_devices,
            master_server_addr,
        )
    except TypeError:
        result = store.setup(setup_config)

    if result != 0:
        raise RuntimeError(f"failed to setup MooncakeDistributedStore: {result}")
    return store, local_hostname


class MooncakeStoreFileConnector(BaseFileConnector):
    """File-level Mooncake Store connector.

    Metadata files are materialized locally for HuggingFace/SGLang config
    loading. Weight files are not materialized: each safetensors shard is read
    one tensor at a time from file-level Mooncake chunks, coalesced into a
    reused registered buffer, and streamed straight to the target device.
    """

    def __init__(self, url: str, device=None, **kwargs) -> None:
        super().__init__(url)
        self.checkpoint_id = _checkpoint_id_from_url(url)
        self.device = device

        config = _load_store_config(kwargs.get("mooncake_weight_store_config"))
        self.store, self.local_hostname = _create_store(config)

        from mooncake.weight_store import READY, ModelFileCacheClient

        self.ready_status = READY
        self.client = ModelFileCacheClient(
            self.store,
            replica_num=int(config.get("replica_num", 1)),
            file_chunk_size=_parse_size(config.get("file_chunk_size", "64MB")),
            progress=False,
        )
        self.manifest = self.client.inspect_model(self.checkpoint_id)
        if self.manifest.status != self.ready_status:
            raise RuntimeError(
                f"Mooncake checkpoint {self.checkpoint_id!r} is not READY: "
                f"{self.manifest.status}"
            )

        root = kwargs.get("metadata_dir") or kwargs.get("materialize_dir")
        root = root or os.environ.get("MOONCAKE_MODEL_MATERIALIZE_DIR")
        root = root or DEFAULT_METADATA_DIR
        self.local_dir = os.path.join(root, self.checkpoint_id)
        os.makedirs(self.local_dir, exist_ok=True)

        self.file_chunk_size = _parse_size(config.get("file_chunk_size", "64MB"))
        self.tensor_batch_buffer_size = _parse_size(
            kwargs.get(
                "tensor_batch_buffer_size",
                config.get("tensor_batch_buffer_size", "512MB"),
            )
        )
        if self.tensor_batch_buffer_size <= 0:
            raise ValueError("tensor_batch_buffer_size must be positive")
        batch_device = kwargs.get(
            "tensor_batch_device", config.get("tensor_batch_device", "auto")
        )
        tensor_batch_device = (
            self.device
            if batch_device == "auto" and self.device is not None
            else ("cpu" if batch_device == "auto" else batch_device)
        )
        try:
            self.tensor_batch_device = torch.device(tensor_batch_device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"invalid tensor_batch_device: {tensor_batch_device!r}"
            ) from error
        if (
            self.tensor_batch_device.type == "cuda"
            and self.tensor_batch_device.index is None
        ):
            self.tensor_batch_device = torch.device("cuda", torch.cuda.current_device())
        self._range_read_disabled = False
        self._batch_buffer = {"buffer": None, "ptr": None, "size": 0}
        logger.info(
            "Initialized MooncakeStoreFileConnector checkpoint=%s local_dir=%s "
            "local_hostname=%s batch_device=%s batch_size=%d",
            self.checkpoint_id,
            self.local_dir,
            self.local_hostname,
            self.tensor_batch_device,
            self.tensor_batch_buffer_size,
        )

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        paths = [record.path for record in self.manifest.files]
        if allow_pattern is not None:
            paths = [path for path in paths if _matches(path, allow_pattern)]
        return [f"{self.url.rstrip('/')}/{path}" for path in sorted(paths)]

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        for record in self.manifest.files:
            path = record.path
            if allow_pattern is not None and not _matches(path, allow_pattern):
                continue
            if ignore_pattern is not None and _matches(path, ignore_pattern):
                continue
            if ignore_pattern is None and _matches(path, WEIGHT_PATTERNS):
                continue

            output_path = os.path.join(self.local_dir, path)
            if self._local_file_matches(output_path, record.size):
                continue
            logger.info(
                "Materializing Mooncake metadata file %s to %s",
                path,
                output_path,
            )
            self.client.materialize_file(self.checkpoint_id, path, output_path)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        del rank
        for record in self._weight_records():
            logger.info(
                "Streaming Mooncake safetensors shard %s (%d bytes) by tensor",
                record.path,
                record.size,
            )
            reader = _SequentialChunkReader(self.store, record, self.file_chunk_size)
            header_size = struct.unpack("<Q", reader.read(0, 8))[0]
            if header_size > record.size - 8:
                raise RuntimeError(
                    f"invalid safetensors header size for {record.path}: {header_size}"
                )
            header = json.loads(reader.read(8, header_size).decode("utf-8"))
            data_offset = 8 + header_size

            tensor_infos = self._parse_tensor_infos(header, data_offset, record.size)
            batches = self._tensor_batches(tensor_infos)
            for batch in batches:
                tensors = self._read_tensor_batch(reader, record, batch)
                for name, tensor in tensors:
                    yield name, tensor

    def close(self):
        if self.closed:
            return
        self.closed = True
        slot = getattr(self, "_batch_buffer", None)
        if slot is not None and slot["ptr"] is not None:
            result = self.store.unregister_buffer(slot["ptr"])
            if result != 0:
                logger.warning(
                    "Failed to unregister Mooncake tensor batch buffer: %s", result
                )
            slot["ptr"] = None
            slot["buffer"] = None
            slot["size"] = 0
        closer = getattr(getattr(self, "store", None), "close", None)
        if closer is not None:
            closer()
        # Do not remove local_dir: metadata files must remain available for the
        # lifetime of SGLang config/tokenizer users and are useful for debugging.

    def _weight_records(self):
        records = [
            record
            for record in self.manifest.files
            if record.path.endswith(".safetensors")
        ]
        if not records:
            raise RuntimeError(
                f"Mooncake checkpoint {self.checkpoint_id!r} has no safetensors files"
            )
        return sorted(records, key=lambda record: record.path)

    @staticmethod
    def _local_file_matches(path: str, size: int) -> bool:
        try:
            return os.path.getsize(path) == size
        except OSError:
            return False

    def _read_tensor_batch(self, reader, record, batch):
        batch_size = sum(info["size"] for info in batch)
        if batch_size == 0:
            return [
                (info["name"], torch.empty(info["shape"], dtype=info["dtype"]))
                for info in batch
            ]

        if not self._ensure_batch_buffer(batch_size):
            return [self._read_tensor_fallback(reader, record, info) for info in batch]

        slot = self._batch_buffer

        keys = []
        dst_offsets = []
        src_offsets = []
        sizes = []
        tensor_offset = 0
        for info in batch:
            fragments = self._range_fragments(
                record,
                info["offset"],
                info["size"],
                destination_offset=tensor_offset,
            )
            for key, dst, src, fragment_size in zip(*fragments):
                if (
                    keys
                    and keys[-1] == key
                    and dst_offsets[-1][0] + sizes[-1][0] == dst[0]
                    and src_offsets[-1][0] + sizes[-1][0] == src[0]
                ):
                    sizes[-1][0] += fragment_size[0]
                else:
                    keys.append(key)
                    dst_offsets.append(dst)
                    src_offsets.append(src)
                    sizes.append(fragment_size)
            info["buffer_offset"] = tensor_offset
            tensor_offset += info["size"]

        try:
            results = self.store.get_into_ranges(
                [slot["ptr"]],
                [keys],
                [dst_offsets],
                [src_offsets],
                [sizes],
            )
            if len(results) != 1 or len(results[0]) != len(keys):
                raise RuntimeError(
                    f"get_into_ranges returned an invalid result for {record.path}"
                )
            for key, expected_sizes, actual_sizes in zip(keys, sizes, results[0]):
                if actual_sizes != expected_sizes:
                    raise RuntimeError(
                        f"get_into_ranges failed for {key}: "
                        f"expected {expected_sizes}, got {actual_sizes}"
                    )
        except Exception as error:
            self._range_read_disabled = True
            logger.warning(
                "Mooncake batched range read failed for %s; falling back to "
                "sequential get for this connector: %s",
                record.path,
                error,
            )
            return [self._read_tensor_fallback(reader, record, info) for info in batch]

        tensors = []
        for info in batch:
            start = info["buffer_offset"]
            raw = slot["buffer"][start : start + info["size"]]
            tensor = raw.view(info["dtype"]).reshape(info["shape"])
            tensors.append((info["name"], tensor))
        return tensors

    def _ensure_batch_buffer(self, required_size: int) -> bool:
        register_buffer = getattr(self.store, "register_buffer", None)
        unregister_buffer = getattr(self.store, "unregister_buffer", None)
        get_into_ranges = getattr(self.store, "get_into_ranges", None)
        if self._range_read_disabled or not all(
            callable(method)
            for method in (register_buffer, unregister_buffer, get_into_ranges)
        ):
            return False

        slot = self._batch_buffer
        if slot["size"] >= required_size:
            return True

        if slot["ptr"] is not None:
            result = unregister_buffer(slot["ptr"])
            if result != 0:
                raise RuntimeError(
                    f"failed to resize Mooncake tensor batch buffer: {result}"
                )
            slot["buffer"] = None
            slot["ptr"] = None
            slot["size"] = 0

        size = max(required_size, self.tensor_batch_buffer_size)
        buffer = torch.empty(size, dtype=torch.uint8, device=self.tensor_batch_device)
        buffer_ptr = buffer.data_ptr()
        result = register_buffer(buffer_ptr, size)
        if result != 0:
            logger.warning(
                "Failed to register tensor buffer for %s, falling back to get: %s",
                "tensor batch buffer",
                result,
            )
            self._range_read_disabled = True
            return False

        slot["buffer"] = buffer
        slot["ptr"] = buffer_ptr
        slot["size"] = size
        return True

    @staticmethod
    def _read_tensor_fallback(reader, record, info):
        payload = reader.read(info["offset"], info["size"])
        tensor = torch.frombuffer(payload, dtype=info["dtype"]).reshape(info["shape"])
        return info["name"], tensor

    def _range_fragments(
        self, record, offset: int, size: int, destination_offset: int = 0
    ):
        keys = []
        dst_offsets = []
        src_offsets = []
        sizes = []
        copied = 0
        while copied < size:
            absolute_offset = offset + copied
            chunk_index = absolute_offset // self.file_chunk_size
            source_offset = absolute_offset % self.file_chunk_size
            fragment_size = min(size - copied, self.file_chunk_size - source_offset)
            keys.append(record.chunks[chunk_index])
            dst_offsets.append([destination_offset + copied])
            src_offsets.append([source_offset])
            sizes.append([fragment_size])
            copied += fragment_size
        return keys, dst_offsets, src_offsets, sizes

    @staticmethod
    def _tensor_records(header: dict):
        records = [
            (name, metadata)
            for name, metadata in header.items()
            if name != "__metadata__"
        ]
        return sorted(records, key=lambda item: item[1]["data_offsets"][0])

    def _parse_tensor_infos(self, header: dict, data_offset: int, file_size: int):
        infos = []
        for name, metadata in self._tensor_records(header):
            dtype_name = metadata["dtype"]
            if dtype_name not in SAFETENSORS_DTYPES:
                raise ValueError(
                    f"unsupported safetensors dtype {dtype_name!r} for {name}"
                )
            dtype, element_size = SAFETENSORS_DTYPES[dtype_name]
            shape = tuple(metadata["shape"])
            start, end = metadata["data_offsets"]
            if start < 0 or end < start or data_offset + end > file_size:
                raise RuntimeError(
                    f"invalid safetensors data offsets for {name}: "
                    f"[{start}, {end}] in a {file_size}-byte file"
                )
            tensor_size = end - start
            expected_size = math.prod(shape) * element_size
            if tensor_size != expected_size:
                raise RuntimeError(
                    f"invalid safetensors tensor size for {name}: "
                    f"expected {expected_size}, got {tensor_size}"
                )
            infos.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "offset": data_offset + start,
                    "size": tensor_size,
                }
            )
        return infos

    def _tensor_batches(self, infos):
        batches = []
        current = []
        current_size = 0
        for info in infos:
            if current and current_size + info["size"] > self.tensor_batch_buffer_size:
                batches.append(current)
                current = []
                current_size = 0
            current.append(info)
            current_size += info["size"]
        if current:
            batches.append(current)
        return batches


class _SequentialChunkReader:
    def __init__(self, store, record, chunk_size: int) -> None:
        self.store = store
        self.record = record
        self.chunk_size = chunk_size
        self.chunk_index = -1
        self.chunk = b""

    def read(self, offset: int, size: int) -> bytearray:
        if offset < 0 or size < 0 or offset + size > self.record.size:
            raise ValueError(
                f"invalid range for {self.record.path}: offset={offset}, size={size}"
            )
        output = bytearray(size)
        output_offset = 0
        while output_offset < size:
            absolute_offset = offset + output_offset
            chunk_index = absolute_offset // self.chunk_size
            if chunk_index != self.chunk_index:
                chunk_key = self.record.chunks[chunk_index]
                chunk = self.store.get(chunk_key)
                if chunk is None:
                    raise KeyError(chunk_key)
                self.chunk = bytes(chunk)
                self.chunk_index = chunk_index

            chunk_offset = absolute_offset - chunk_index * self.chunk_size
            copy_size = min(size - output_offset, len(self.chunk) - chunk_offset)
            if copy_size <= 0:
                raise RuntimeError(
                    f"invalid chunk size for {self.record.path} chunk {chunk_index}"
                )
            output[output_offset : output_offset + copy_size] = memoryview(self.chunk)[
                chunk_offset : chunk_offset + copy_size
            ]
            output_offset += copy_size
        return output
