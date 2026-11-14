#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structural inventory and adapter manifest for Kimi K3 GGUF expert packs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

FORMAT = "SGLANG-KIMI-GGMLMOEPACK-ADAPTER-v1"
PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_VERSION = 1
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
PACK_ALIGNMENT = 4096
ROLE_ORDER = ("up", "gate", "down")
EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<role>up|gate|down)_exps\.weight$"
)
SHARD_RE = re.compile(r"-(?P<number>\d{5})-of-(?P<count>\d{5})\.gguf$")
COPY_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class KimiK3Spec:
    num_hidden_layers: int
    first_k_dense_replace: int
    num_experts: int
    top_k: int
    num_shared_experts: int
    hidden_size: int
    routed_expert_hidden_size: int
    moe_intermediate_size: int
    hidden_act: str
    active_moe_layer_ids: tuple[int, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KimiK3Spec:
        text_config = config.get("text_config", config)
        num_hidden_layers = int(text_config["num_hidden_layers"])
        first_dense = int(text_config["first_k_dense_replace"])
        active_layers = tuple(range(first_dense, num_hidden_layers))
        result = cls(
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=first_dense,
            num_experts=int(text_config["num_experts"]),
            top_k=int(text_config["num_experts_per_token"]),
            num_shared_experts=int(text_config["num_shared_experts"]),
            hidden_size=int(text_config["hidden_size"]),
            routed_expert_hidden_size=int(text_config["routed_expert_hidden_size"]),
            moe_intermediate_size=int(text_config["moe_intermediate_size"]),
            hidden_act=str(text_config["hidden_act"]),
            active_moe_layer_ids=active_layers,
        )
        result.validate_kimi_k3()
        return result

    def validate_kimi_k3(self) -> None:
        expected = {
            "num_hidden_layers": 93,
            "first_k_dense_replace": 1,
            "num_experts": 896,
            "top_k": 16,
            "num_shared_experts": 2,
            "hidden_size": 7168,
            "routed_expert_hidden_size": 3584,
            "moe_intermediate_size": 3072,
            "hidden_act": "situ",
        }
        actual = {
            "num_hidden_layers": self.num_hidden_layers,
            "first_k_dense_replace": self.first_k_dense_replace,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_shared_experts": self.num_shared_experts,
            "hidden_size": self.hidden_size,
            "routed_expert_hidden_size": self.routed_expert_hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "hidden_act": self.hidden_act,
        }
        if actual != expected:
            raise ValueError(
                "Kimi K3 model invariants do not match the audited model: "
                f"expected={expected}, actual={actual}"
            )
        expected_layers = tuple(range(1, 93))
        if self.active_moe_layer_ids != expected_layers:
            raise ValueError("Kimi K3 active MoE layers must be exactly 1..92")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    dtype_id: int
    shard_index: int
    shard_path: str
    data_offset: int
    nbytes: int

    @property
    def expert_key(self) -> tuple[int, str] | None:
        match = EXPERT_RE.fullmatch(self.name)
        if match is None:
            return None
        return int(match.group("layer")), match.group("role")


@dataclass(frozen=True)
class PackEntryRecord:
    tensor_name: str
    expert: int
    offset: int
    nbytes: int


def _field_value(reader: Any, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        raise ValueError(f"GGUF metadata is missing required field {name!r}")
    return field.contents()


def _sha256_range(stream: BinaryIO, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = stream.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise EOFError(f"short read at offset {offset}; {remaining} bytes remain")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb", buffering=0) as stream:
        return _sha256_range(stream, 0, path.stat().st_size)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def discover_gguf_shards(directory: Path) -> list[Path]:
    directory = directory.resolve(strict=True)
    candidates = sorted(directory.glob("*.gguf"))
    if not candidates:
        raise ValueError(f"no GGUF files found in {directory}")
    numbered: list[tuple[int, int, Path]] = []
    for path in candidates:
        match = SHARD_RE.search(path.name)
        if match is None:
            raise ValueError(f"GGUF shard name does not contain NNNNN-of-NNNNN: {path}")
        numbered.append((int(match.group("number")), int(match.group("count")), path))
    counts = {count for _, count, _ in numbered}
    if len(counts) != 1:
        raise ValueError(f"GGUF shard filenames disagree on split count: {counts}")
    count = counts.pop()
    numbers = [number for number, _, _ in numbered]
    if count != len(numbered) or sorted(numbers) != list(range(1, count + 1)):
        raise ValueError(
            f"GGUF shard set is incomplete: count={count}, numbers={sorted(numbers)}"
        )
    return [path for _, _, path in sorted(numbered)]


def _git_sha(repo: Path | None) -> str:
    if repo is None:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def scan_gguf_shards(
    paths: list[Path], *, full_source_hashes: bool = False
) -> tuple[list[dict[str, Any]], list[TensorRecord], dict[str, Any]]:
    import gguf

    shard_records: list[dict[str, Any]] = []
    tensors: list[TensorRecord] = []
    seen_names: set[str] = set()
    split_count: int | None = None
    split_tensor_count: int | None = None
    architecture: str | None = None
    for shard_index, path in enumerate(paths):
        reader = gguf.GGUFReader(str(path), mode="r")
        shard_split_count = int(_field_value(reader, "split.count"))
        shard_split_no = int(_field_value(reader, "split.no"))
        shard_tensor_count = int(_field_value(reader, "split.tensors.count"))
        shard_architecture = str(_field_value(reader, "general.architecture"))
        if shard_split_no != shard_index:
            raise ValueError(
                f"GGUF split.no mismatch for {path}: {shard_split_no} != {shard_index}"
            )
        if split_count is None:
            split_count = shard_split_count
            split_tensor_count = shard_tensor_count
            architecture = shard_architecture
        elif (
            split_count != shard_split_count
            or split_tensor_count != shard_tensor_count
            or architecture != shard_architecture
        ):
            raise ValueError(f"GGUF split metadata mismatch at {path}")

        metadata_nbytes = int(reader.data_offset)
        with path.open("rb", buffering=0) as stream:
            metadata_sha256 = _sha256_range(stream, 0, metadata_nbytes)
        shard_record: dict[str, Any] = {
            "index": shard_index,
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "metadata_nbytes": metadata_nbytes,
            "metadata_sha256": metadata_sha256,
            "tensor_count": len(reader.tensors),
        }
        if full_source_hashes:
            shard_record["sha256"] = sha256_file(path)
        shard_records.append(shard_record)

        for tensor in reader.tensors:
            if tensor.name in seen_names:
                raise ValueError(f"duplicate GGUF tensor across shards: {tensor.name}")
            seen_names.add(tensor.name)
            tensors.append(
                TensorRecord(
                    name=tensor.name,
                    shape=tuple(int(value) for value in tensor.shape.tolist()),
                    dtype=tensor.tensor_type.name,
                    dtype_id=int(tensor.tensor_type),
                    shard_index=shard_index,
                    shard_path=str(path.resolve()),
                    data_offset=int(tensor.data_offset),
                    nbytes=int(tensor.n_bytes),
                )
            )
        del reader

    if split_count != len(paths):
        raise ValueError(f"GGUF split.count={split_count}, found {len(paths)} files")
    if split_tensor_count != len(tensors):
        raise ValueError(
            f"GGUF split.tensors.count={split_tensor_count}, found {len(tensors)}"
        )
    if architecture != "kimi-k3":
        raise ValueError(f"expected GGUF architecture 'kimi-k3', got {architecture!r}")
    summary = {
        "architecture": architecture,
        "shard_count": len(paths),
        "tensor_count": len(tensors),
        "total_bytes": sum(item["size"] for item in shard_records),
        "full_source_hashes": full_source_hashes,
    }
    return shard_records, tensors, summary


def validate_expert_tensors(
    tensors: Iterable[TensorRecord], spec: KimiK3Spec
) -> dict[tuple[int, str], TensorRecord]:
    expert_tensors: dict[tuple[int, str], TensorRecord] = {}
    for tensor in tensors:
        key = tensor.expert_key
        if key is None:
            continue
        if key in expert_tensors:
            raise ValueError(f"duplicate routed expert tensor {key}")
        expert_tensors[key] = tensor
    expected_keys = {
        (layer, role) for layer in spec.active_moe_layer_ids for role in ROLE_ORDER
    }
    if set(expert_tensors) != expected_keys:
        missing = sorted(expected_keys - set(expert_tensors))
        extra = sorted(set(expert_tensors) - expected_keys)
        raise ValueError(
            f"routed expert tensor coverage mismatch: missing={missing[:8]}, "
            f"extra={extra[:8]}"
        )
    expected_layout = {
        "up": ((spec.routed_expert_hidden_size, spec.moe_intermediate_size), "Q2_K"),
        "gate": (
            (spec.routed_expert_hidden_size, spec.moe_intermediate_size),
            "Q2_K",
        ),
        "down": (
            (spec.moe_intermediate_size, spec.routed_expert_hidden_size),
            "Q3_K",
        ),
    }
    for (layer, role), tensor in expert_tensors.items():
        expected_shape, expected_dtype = expected_layout[role]
        if tensor.shape != (*expected_shape, spec.num_experts):
            raise ValueError(
                f"unexpected expert shape for {(layer, role)}: {tensor.shape}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"unexpected expert dtype for {(layer, role)}: {tensor.dtype}"
            )
        if tensor.nbytes % spec.num_experts:
            raise ValueError(f"expert tensor is not evenly sliceable: {tensor.name}")
    return expert_tensors


def _decode_name(raw: bytes) -> str:
    try:
        return raw.split(b"\0", 1)[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("expert-pack tensor name is not valid UTF-8") from exc


def _read_pack_entry(stream: BinaryIO, digest: hashlib._Hash) -> PackEntryRecord:
    raw = stream.read(PACK_ENTRY.size)
    if len(raw) != PACK_ENTRY.size:
        raise ValueError("GGML expert-pack index is truncated")
    digest.update(raw)
    name, expert, reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
    if reserved != 0:
        raise ValueError("GGML expert-pack reserved entry field must be zero")
    return PackEntryRecord(_decode_name(name), expert, offset, nbytes)


def _compare_ranges(
    pack_stream: BinaryIO,
    pack_entry: PackEntryRecord,
    tensor: TensorRecord,
    expert_bytes: int,
) -> None:
    source_offset = tensor.data_offset + pack_entry.expert * expert_bytes
    with Path(tensor.shard_path).open("rb", buffering=0) as source_stream:
        source_stream.seek(source_offset)
        pack_stream.seek(pack_entry.offset)
        remaining = expert_bytes
        while remaining:
            size = min(COPY_CHUNK_BYTES, remaining)
            source_chunk = source_stream.read(size)
            pack_chunk = pack_stream.read(size)
            if source_chunk != pack_chunk or len(source_chunk) != size:
                raise ValueError(
                    "expert-pack payload does not match GGUF source for "
                    f"{pack_entry.tensor_name} expert {pack_entry.expert}"
                )
            remaining -= size


def validate_ggml_moe_pack(
    path: Path,
    expert_tensors: dict[tuple[int, str], TensorRecord],
    spec: KimiK3Spec,
    *,
    payload_samples: int = 6,
    full_pack_hash: bool = False,
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    expected_count = len(spec.active_moe_layer_ids) * spec.num_experts * len(ROLE_ORDER)
    file_size = path.stat().st_size
    index_digest = hashlib.sha256()
    role_summary: dict[str, dict[str, Any]] = {}
    sample_indices = set()
    if payload_samples > 0:
        if payload_samples == 1:
            sample_indices.add(0)
        else:
            sample_indices.update(
                round(index * (expected_count - 1) / (payload_samples - 1))
                for index in range(payload_samples)
            )
    sampled_entries: list[tuple[PackEntryRecord, TensorRecord, int]] = []
    object_bytes: int | None = None
    object_start: int | None = None
    previous_end = 0

    with path.open("rb", buffering=0) as stream:
        raw_header = stream.read(PACK_HEADER.size)
        if len(raw_header) != PACK_HEADER.size:
            raise ValueError("GGML expert-pack header is truncated")
        index_digest.update(raw_header)
        magic, version, header_size, index_count, data_start = PACK_HEADER.unpack(
            raw_header
        )
        if magic != PACK_MAGIC or version != PACK_VERSION:
            raise ValueError("GGML expert-pack magic or version does not match")
        if header_size != PACK_HEADER.size:
            raise ValueError("GGML expert-pack header size does not match")
        if index_count != expected_count:
            raise ValueError(
                f"GGML expert-pack has {index_count} entries; expected {expected_count}"
            )
        minimum_data_start = PACK_HEADER.size + index_count * PACK_ENTRY.size
        if data_start < minimum_data_start or data_start % PACK_ALIGNMENT:
            raise ValueError("GGML expert-pack data_start is invalid or unaligned")
        previous_end = data_start

        for index in range(index_count):
            entry = _read_pack_entry(stream, index_digest)
            object_index, physical_role_id = divmod(index, len(ROLE_ORDER))
            active_layer_index, expected_expert = divmod(object_index, spec.num_experts)
            expected_layer = spec.active_moe_layer_ids[active_layer_index]
            expected_role = ROLE_ORDER[physical_role_id]
            match = EXPERT_RE.fullmatch(entry.tensor_name)
            if match is None:
                raise ValueError(f"invalid expert tensor name: {entry.tensor_name!r}")
            actual_key = (int(match.group("layer")), match.group("role"))
            expected_key = (expected_layer, expected_role)
            if actual_key != expected_key or entry.expert != expected_expert:
                raise ValueError(
                    "GGML expert-pack is not complete expert-major up/gate/down "
                    f"layout at index {index}: expected={(*expected_key, expected_expert)}, "
                    f"actual={(*actual_key, entry.expert)}"
                )
            tensor = expert_tensors[expected_key]
            expert_bytes = tensor.nbytes // spec.num_experts
            if entry.nbytes != expert_bytes:
                raise ValueError(
                    f"expert-pack byte size mismatch for index {index}: "
                    f"{entry.nbytes} != {expert_bytes}"
                )
            if entry.offset % PACK_ALIGNMENT:
                raise ValueError(f"expert-pack entry {index} is not 4 KiB aligned")
            if entry.offset < previous_end or entry.offset + entry.nbytes > file_size:
                raise ValueError(
                    f"expert-pack entry {index} overlaps or is out of range"
                )
            previous_end = entry.offset + entry.nbytes
            summary = role_summary.setdefault(
                expected_role,
                {
                    "dtype": tensor.dtype,
                    "dtype_id": tensor.dtype_id,
                    "logical_shape": list(tensor.shape[:2]),
                    "expert_bytes": expert_bytes,
                    "entry_count": 0,
                    "payload_bytes": 0,
                },
            )
            if summary["expert_bytes"] != expert_bytes:
                raise ValueError(f"variable expert bytes for role {expected_role}")
            summary["entry_count"] += 1
            summary["payload_bytes"] += entry.nbytes

            if physical_role_id == 0:
                object_start = entry.offset
            elif physical_role_id == len(ROLE_ORDER) - 1:
                assert object_start is not None
                span = entry.offset + entry.nbytes - object_start
                if object_bytes is None:
                    object_bytes = span
                elif object_bytes != span:
                    raise ValueError("expert-pack object spans are not fixed size")
            if index in sample_indices:
                sampled_entries.append((entry, tensor, expert_bytes))

        if previous_end != file_size:
            raise ValueError(
                f"expert-pack has unexplained trailing bytes: {file_size - previous_end}"
            )
        for entry, tensor, expert_bytes in sampled_entries:
            _compare_ranges(stream, entry, tensor, expert_bytes)

    assert object_bytes is not None
    result: dict[str, Any] = {
        "path": str(path),
        "size": file_size,
        "magic": PACK_MAGIC.rstrip(b"\0").decode("ascii"),
        "version": PACK_VERSION,
        "header_bytes": PACK_HEADER.size,
        "entry_bytes": PACK_ENTRY.size,
        "index_count": expected_count,
        "data_start": data_start,
        "alignment": PACK_ALIGNMENT,
        "index_sha256": index_digest.hexdigest(),
        "physical_role_order": list(ROLE_ORDER),
        "active_moe_layer_ids": list(spec.active_moe_layer_ids),
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "object_bytes": object_bytes,
        "roles": role_summary,
        "payload_samples_verified": len(sampled_entries),
        "full_pack_hash": full_pack_hash,
    }
    if full_pack_hash:
        result["sha256"] = sha256_file(path)
    return result


def _align_up(value: int, alignment: int = PACK_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _pack_layout(
    expert_tensors: dict[tuple[int, str], TensorRecord], spec: KimiK3Spec
) -> tuple[list[tuple[PackEntryRecord, TensorRecord]], int, int]:
    index_count = len(spec.active_moe_layer_ids) * spec.num_experts * len(ROLE_ORDER)
    data_start = _align_up(PACK_HEADER.size + index_count * PACK_ENTRY.size)
    offset = data_start
    entries: list[tuple[PackEntryRecord, TensorRecord]] = []
    for layer in spec.active_moe_layer_ids:
        for expert in range(spec.num_experts):
            for role in ROLE_ORDER:
                tensor = expert_tensors[(layer, role)]
                expert_bytes = tensor.nbytes // spec.num_experts
                offset = _align_up(offset)
                entries.append(
                    (
                        PackEntryRecord(tensor.name, expert, offset, expert_bytes),
                        tensor,
                    )
                )
                offset += expert_bytes
    return entries, data_start, offset


def estimate_ggml_moe_pack_size(
    expert_tensors: dict[tuple[int, str], TensorRecord], spec: KimiK3Spec
) -> int:
    return _pack_layout(expert_tensors, spec)[2]


def _copy_tensor_slice(
    source: BinaryIO, output: BinaryIO, offset: int, nbytes: int
) -> None:
    source.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise EOFError(
                f"short GGUF read at offset {offset}; {remaining} bytes remain"
            )
        output.write(chunk)
        remaining -= len(chunk)


def write_ggml_moe_pack(
    path: Path,
    expert_tensors: dict[tuple[int, str], TensorRecord],
    spec: KimiK3Spec,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    entries, data_start, final_size = _pack_layout(expert_tensors, spec)
    path = path.resolve()
    partial = path.with_name(path.name + ".partial")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Expert Pack: {path}")
    if partial.exists():
        raise FileExistsError(f"partial Expert Pack already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        sources = {
            source_path: stack.enter_context(Path(source_path).open("rb", buffering=0))
            for source_path in {tensor.shard_path for _, tensor in entries}
        }
        output = stack.enter_context(partial.open("xb", buffering=0))
        output.write(
            PACK_HEADER.pack(
                PACK_MAGIC,
                PACK_VERSION,
                PACK_HEADER.size,
                len(entries),
                data_start,
            )
        )
        for entry, _ in entries:
            encoded_name = entry.tensor_name.encode("utf-8")
            if len(encoded_name) >= 128:
                raise ValueError(
                    f"expert tensor name is too long for the Pack index: {entry.tensor_name}"
                )
            output.write(
                PACK_ENTRY.pack(
                    encoded_name.ljust(128, b"\0"),
                    entry.expert,
                    0,
                    entry.offset,
                    entry.nbytes,
                )
            )
        output.write(bytes(data_start - output.tell()))

        total = len(entries)
        for index, (entry, tensor) in enumerate(entries, start=1):
            padding = entry.offset - output.tell()
            if padding < 0:
                raise RuntimeError("Expert Pack layout moved backwards")
            if padding:
                output.write(bytes(padding))
            source_offset = tensor.data_offset + entry.expert * (
                tensor.nbytes // spec.num_experts
            )
            _copy_tensor_slice(
                sources[tensor.shard_path], output, source_offset, entry.nbytes
            )
            if progress is not None and (index % 1024 == 0 or index == total):
                progress(index, total)
        output.flush()
        os.fsync(output.fileno())

    if partial.stat().st_size != final_size:
        raise RuntimeError(
            f"generated Expert Pack size {partial.stat().st_size} != {final_size}"
        )
    os.replace(partial, path)
    return final_size


def create_manifest(
    *,
    gguf_dir: Path,
    expert_pack: Path,
    model_config: Path,
    tokenizer_dir: Path,
    payload_samples: int = 6,
    full_source_hashes: bool = False,
    full_pack_hash: bool = False,
    repo: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(model_config.read_text(encoding="utf-8"))
    spec = KimiK3Spec.from_config(config)
    shard_paths = discover_gguf_shards(gguf_dir)
    shard_records, tensors, source_summary = scan_gguf_shards(
        shard_paths, full_source_hashes=full_source_hashes
    )
    expert_tensors = validate_expert_tensors(tensors, spec)
    pack = validate_ggml_moe_pack(
        expert_pack,
        expert_tensors,
        spec,
        payload_samples=payload_samples,
        full_pack_hash=full_pack_hash,
    )

    tokenizer_files = []
    for path in sorted(tokenizer_dir.resolve(strict=True).iterdir()):
        if path.is_file():
            tokenizer_files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    tensor_records = [
        {
            "name": tensor.name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "dtype_id": tensor.dtype_id,
            "shard_index": tensor.shard_index,
            "data_offset": tensor.data_offset,
            "nbytes": tensor.nbytes,
        }
        for tensor in sorted(tensors, key=lambda item: item.name)
    ]
    source_inventory = {
        "summary": source_summary,
        "shards": shard_records,
        "tensors": tensor_records,
    }
    model = {
        "config_path": str(model_config.resolve()),
        "config_sha256": sha256_file(model_config),
        "architecture": "KimiLinearForCausalLM",
        "num_hidden_layers": spec.num_hidden_layers,
        "active_moe_layer_ids": list(spec.active_moe_layer_ids),
        "num_experts": spec.num_experts,
        "num_experts_per_token": spec.top_k,
        "num_shared_experts": spec.num_shared_experts,
        "hidden_size": spec.hidden_size,
        "routed_expert_hidden_size": spec.routed_expert_hidden_size,
        "moe_intermediate_size": spec.moe_intermediate_size,
        "hidden_act": spec.hidden_act,
    }
    return {
        "complete": True,
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sglang_git_sha": _git_sha(repo),
        "hard_constraints": {
            "top_k": 16,
            "top_k_is_immutable": True,
            "all_selected_experts_must_execute": True,
            "expert_pruning_allowed": False,
            "requantization_allowed": False,
        },
        "model": model,
        "source": {
            **source_inventory,
            "inventory_sha256": canonical_sha256(source_inventory),
        },
        "expert_pack": pack,
        "tokenizer": {
            "path": str(tokenizer_dir.resolve()),
            "files": tokenizer_files,
            "inventory_sha256": canonical_sha256(tokenizer_files),
        },
        "verification": {
            "structure": "complete",
            "payload_samples": payload_samples,
            "full_source_hashes": full_source_hashes,
            "full_pack_hash": full_pack_hash,
        },
    }
