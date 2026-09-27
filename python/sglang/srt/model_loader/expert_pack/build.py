#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gguf

try:
    from .format import (
        ENTRY_STRUCT,
        FLAG_IDENTITY_PAYLOAD,
        FLAG_TRIPLET_OBJECTS,
        HEADER_STRUCT,
        ROLE_NAMES,
        IndexEntry,
        PackHeader,
        align_up,
        inspect_pack,
        sha256_file,
        write_index,
    )
except ImportError:
    from format import (  # type: ignore[no-redef]
        ENTRY_STRUCT,
        FLAG_IDENTITY_PAYLOAD,
        FLAG_TRIPLET_OBJECTS,
        HEADER_STRUCT,
        ROLE_NAMES,
        IndexEntry,
        PackHeader,
        align_up,
        inspect_pack,
        sha256_file,
        write_index,
    )


FORMAT = "SGLANG-EXPERTPACK-v1"
EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<role>gate|up|down)_exps\.weight$"
)
COPY_CHUNK_BYTES = 16 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def hash_range(stream, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = stream.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise EOFError(
                f"short read at source offset {offset}, {remaining} bytes remain"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def copy_range(source_fd: int, output, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    copied = 0
    while copied < nbytes:
        chunk = os.pread(
            source_fd, min(COPY_CHUNK_BYTES, nbytes - copied), offset + copied
        )
        if not chunk:
            raise EOFError(f"short read at source offset {offset + copied}")
        output.write(chunk)
        digest.update(chunk)
        copied += len(chunk)
    return digest.hexdigest()


def load_inventory(
    path: Path, source: Path, expected_sha256: str
) -> tuple[dict, list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    headers = [row for row in rows if row.get("kind") == "source"]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    tensors = [row for row in rows if row.get("kind") == "tensor"]
    if len(headers) != 1 or len(summaries) != 1:
        raise ValueError(
            "inventory must contain exactly one source and one summary record"
        )
    header = headers[0]
    if Path(header["path"]).resolve() != source.resolve():
        raise ValueError("inventory source path does not match --source")
    if header.get("source_sha256") != expected_sha256 or not header.get(
        "payload_hashes"
    ):
        raise ValueError(
            "inventory is not a full-hash inventory for the requested source"
        )
    if int(header["size"]) != source.stat().st_size:
        raise ValueError("inventory source size does not match the source file")
    if len(tensors) != int(header["tensor_count"]):
        raise ValueError("inventory tensor count does not match its header")
    if any(not row.get("sha256") for row in tensors):
        raise ValueError("inventory contains a tensor without a payload hash")
    return header, tensors, summaries[0]


def create_inventory(source: Path, source_sha256: str) -> tuple[dict, list[dict], dict]:
    reader = gguf.GGUFReader(source, "r")
    tensors = []
    inventory_digest = hashlib.sha256()
    with source.open("rb", buffering=0) as stream:
        for tensor in sorted(reader.tensors, key=lambda item: item.name):
            record = {
                "kind": "tensor",
                "name": tensor.name,
                "shape": [int(value) for value in tensor.shape.tolist()],
                "type": tensor.tensor_type.name,
                "type_id": int(tensor.tensor_type),
                "offset": int(tensor.data_offset),
                "nbytes": int(tensor.n_bytes),
                "sha256": hash_range(
                    stream, int(tensor.data_offset), int(tensor.n_bytes)
                ),
            }
            encoded = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
            inventory_digest.update(encoded)
            tensors.append(record)
    header = {
        "kind": "source",
        "path": str(source.resolve()),
        "size": source.stat().st_size,
        "source_sha256": source_sha256,
        "gguf_data_offset": int(reader.data_offset),
        "gguf_alignment": int(reader.alignment),
        "tensor_count": len(tensors),
        "metadata_count": len(reader.fields),
        "payload_hashes": True,
    }
    summary = {
        "kind": "summary",
        "tensor_count": len(tensors),
        "inventory_sha256": inventory_digest.hexdigest(),
    }
    return header, tensors, summary


def validate_inventory_against_reader(source: Path, tensors: list[dict]) -> None:
    reader = gguf.GGUFReader(source, "r")
    actual = {
        tensor.name: {
            "shape": [int(value) for value in tensor.shape.tolist()],
            "type": tensor.tensor_type.name,
            "type_id": int(tensor.tensor_type),
            "offset": int(tensor.data_offset),
            "nbytes": int(tensor.n_bytes),
        }
        for tensor in reader.tensors
    }
    recorded = {row["name"]: row for row in tensors}
    if set(actual) != set(recorded):
        raise ValueError("inventory tensor names do not match the GGUF reader")
    for name, value in actual.items():
        if any(value[field] != recorded[name].get(field) for field in value):
            raise ValueError(f"inventory metadata mismatch for tensor {name}")
    ordered = sorted(tensors, key=lambda row: int(row["offset"]))
    previous_end = 0
    for row in ordered:
        offset = int(row["offset"])
        end = offset + int(row["nbytes"])
        if offset < previous_end or end > source.stat().st_size:
            raise ValueError(f"invalid or overlapping source range for {row['name']}")
        previous_end = end


def generation(model_digest: str, source_digest: str, layer: int, expert: int) -> int:
    value = hashlib.sha256(
        f"{model_digest}:{source_digest}:{layer}:{expert}".encode("ascii")
    ).digest()
    return int.from_bytes(value[:8], "little") or 1


def tool_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("format.py")):
        digest.update(path.name.encode("ascii") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(args: argparse.Namespace) -> dict[str, object]:
    started_at = now()
    started_monotonic = time.monotonic()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    partial_path = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        raise ValueError(f"completed output already exists: {output}")
    if args.config_blob is not None:
        if sha256_file(args.config_blob.resolve(strict=True)) != args.config_sha256:
            raise ValueError("DeepSeek config hash does not match its digest")

    actual_source_sha256 = sha256_file(source)
    if actual_source_sha256 != args.source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {args.source_sha256}, got {actual_source_sha256}"
        )

    if args.inventory is None:
        inventory_header, tensors, inventory_summary = create_inventory(
            source, actual_source_sha256
        )
    else:
        inventory_header, tensors, inventory_summary = load_inventory(
            args.inventory.resolve(strict=True), source, actual_source_sha256
        )
    validate_inventory_against_reader(source, tensors)

    expert_tensors: dict[tuple[int, str], dict] = {}
    for row in tensors:
        match = EXPERT_RE.fullmatch(row["name"])
        if match is not None:
            key = int(match.group("layer")), match.group("role")
            if key in expert_tensors:
                raise ValueError(f"duplicate routed-expert tensor {key}")
            expert_tensors[key] = row
    expected_tensor_keys = {
        (layer, role) for layer in range(args.num_layers) for role in ROLE_NAMES
    }
    if set(expert_tensors) != expected_tensor_keys:
        missing = sorted(expected_tensor_keys - set(expert_tensors))
        extra = sorted(set(expert_tensors) - expected_tensor_keys)
        raise ValueError(
            f"routed-expert tensor coverage mismatch: missing={missing[:8]} extra={extra[:8]}"
        )

    role_bytes = set()
    for (layer, role), row in expert_tensors.items():
        shape = [int(value) for value in row["shape"]]
        if len(shape) != 3 or shape[-1] != args.num_experts:
            raise ValueError(
                f"unexpected expert shape for layer={layer} role={role}: {shape}"
            )
        if int(row["nbytes"]) % args.num_experts:
            raise ValueError(f"expert tensor is not evenly sliceable: {row['name']}")
        slice_bytes = int(row["nbytes"]) // args.num_experts
        block_size, type_size = gguf.GGML_QUANT_SIZES[
            gguf.GGMLQuantizationType(int(row["type_id"]))
        ]
        logical_elements = 1
        for dimension in shape[:-1]:
            logical_elements *= dimension
        expected_bytes = logical_elements // block_size * type_size
        if logical_elements % block_size or slice_bytes != expected_bytes:
            raise ValueError(
                f"quantized slice size mismatch for {row['name']}: {slice_bytes} != {expected_bytes}"
            )
        role_bytes.add(slice_bytes)
    if len(role_bytes) != 1:
        raise ValueError(
            f"triplet v1 requires uniform role sizes, got {sorted(role_bytes)}"
        )
    role_nbytes = role_bytes.pop()

    object_count = args.num_layers * args.num_experts
    object_payload_bytes = role_nbytes * len(ROLE_NAMES)
    object_stride = align_up(object_payload_bytes, args.alignment)
    index_count = object_count * len(ROLE_NAMES)
    data_start = align_up(
        HEADER_STRUCT.size + index_count * ENTRY_STRUCT.size, args.alignment
    )
    expected_pack_bytes = data_start + object_count * object_stride
    header = PackHeader(
        flags=FLAG_IDENTITY_PAYLOAD | FLAG_TRIPLET_OBJECTS,
        index_count=index_count,
        data_start=data_start,
        alignment=args.alignment,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        role_count=len(ROLE_NAMES),
        model_identity_sha256=args.model_identity_sha256,
        source_blob_sha256=actual_source_sha256,
        config_sha256=args.config_sha256,
    )
    header_raw = header.pack()

    existing_bytes = (
        partial_path.stat().st_size if args.resume and partial_path.exists() else 0
    )
    remaining_bytes = max(expected_pack_bytes - existing_bytes, 0)
    free_bytes = shutil.disk_usage(output.parent).free
    safety_bytes = int(args.safety_margin_gib * 1024**3)
    if free_bytes < remaining_bytes + safety_bytes:
        raise OSError(
            f"insufficient free space: free={free_bytes}, remaining_pack={remaining_bytes}, "
            f"safety={safety_bytes}"
        )

    if args.resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "in_progress":
            raise ValueError("resume checkpoint is not in progress")
        for field, expected in (
            ("source_sha256", actual_source_sha256),
            ("model_identity_sha256", args.model_identity_sha256),
            ("config_sha256", args.config_sha256),
            ("tool_sha256", tool_sha256()),
            ("expected_pack_bytes", expected_pack_bytes),
        ):
            if checkpoint.get(field) != expected:
                raise ValueError(f"resume checkpoint {field} mismatch")
        completed_layers = [int(value) for value in checkpoint["completed_layers"]]
        if completed_layers != list(range(len(completed_layers))):
            raise ValueError("completed layers in checkpoint are not a prefix")
        entries = [IndexEntry.from_dict(value) for value in checkpoint["entries"]]
        stream = partial_path.open("r+b", buffering=0)
        if stream.read(len(header_raw)) != header_raw:
            raise ValueError("partial pack header does not match the requested build")
        pack_end = int(checkpoint["pack_end"])
        if partial_path.stat().st_size < pack_end:
            raise ValueError("partial pack is shorter than its checkpoint")
        stream.truncate(pack_end)
        stream.seek(pack_end)
    else:
        if partial_path.exists() or checkpoint_path.exists() or manifest_path.exists():
            raise ValueError(
                "build outputs already exist; use --resume for an in-progress build"
            )
        completed_layers = []
        entries: list[IndexEntry] = []
        stream = partial_path.open("x+b", buffering=0)
        stream.write(header_raw)
        stream.truncate(data_start)
        stream.seek(data_start)
        checkpoint = {
            "format": FORMAT + "-checkpoint",
            "version": 1,
            "status": "in_progress",
            "started_at": started_at,
            "source_sha256": actual_source_sha256,
            "model_identity_sha256": args.model_identity_sha256,
            "config_sha256": args.config_sha256,
            "tool_sha256": tool_sha256(),
            "expected_pack_bytes": expected_pack_bytes,
            "completed_layers": [],
            "pack_end": data_start,
            "entries": [],
        }
        write_json_atomic(checkpoint_path, checkpoint)

    source_fd = os.open(source, os.O_RDONLY)
    try:
        for layer in range(len(completed_layers), args.num_layers):
            layer_entries = []
            for expert in range(args.num_experts):
                object_ordinal = layer * args.num_experts + expert
                object_offset = data_start + object_ordinal * object_stride
                if stream.tell() != object_offset:
                    raise ValueError(
                        f"pack cursor mismatch: {stream.tell()} != {object_offset}"
                    )
                object_generation = generation(
                    args.model_identity_sha256, actual_source_sha256, layer, expert
                )
                for role in ROLE_NAMES:
                    tensor = expert_tensors[(layer, role)]
                    source_slice_offset = int(tensor["offset"]) + expert * role_nbytes
                    pack_offset = stream.tell()
                    slice_sha256 = copy_range(
                        source_fd, stream, source_slice_offset, role_nbytes
                    )
                    dtype_id = int(tensor["type_id"])
                    block_size = int(
                        gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(dtype_id)][0]
                    )
                    entry = IndexEntry(
                        layer=layer,
                        expert=expert,
                        role=role,
                        dtype_id=dtype_id,
                        dtype=str(tensor["type"]),
                        tensor_name=str(tensor["name"]),
                        source_tensor_offset=int(tensor["offset"]),
                        source_tensor_nbytes=int(tensor["nbytes"]),
                        source_slice_offset=source_slice_offset,
                        source_slice_nbytes=role_nbytes,
                        pack_offset=pack_offset,
                        pack_nbytes=role_nbytes,
                        source_tensor_sha256=str(tensor["sha256"]),
                        source_slice_sha256=slice_sha256,
                        checksum=slice_sha256,
                        shape=tuple(int(value) for value in tensor["shape"][:-1]),
                        quant_scheme=str(tensor["type"]),
                        transform_id="identity-v1",
                        block_size=block_size,
                        generation=object_generation,
                    )
                    entry.pack()
                    layer_entries.append(entry)
                padding = object_stride - object_payload_bytes
                if padding:
                    stream.write(bytes(padding))
            stream.flush()
            os.fsync(stream.fileno())
            entries.extend(layer_entries)
            checkpoint["completed_layers"].append(layer)
            checkpoint["pack_end"] = stream.tell()
            checkpoint["entries"] = [entry.to_dict() for entry in entries]
            write_json_atomic(checkpoint_path, checkpoint)
            print(
                f"completed layer {layer}/{args.num_layers - 1}: pack_end={stream.tell()}",
                file=sys.stderr,
                flush=True,
            )

        if stream.tell() != expected_pack_bytes:
            raise ValueError(
                f"final pack size mismatch: {stream.tell()} != {expected_pack_bytes}"
            )
        index_sha256 = write_index(stream, header, entries)
        stream.flush()
        os.fsync(stream.fileno())
    finally:
        os.close(source_fd)
        stream.close()

    pack_sha256 = sha256_file(partial_path)
    reader = gguf.GGUFReader(source, "r")
    with source.open("rb", buffering=0) as source_stream:
        source_metadata_sha256 = hash_range(source_stream, 0, int(reader.data_offset))

    routed_names = {row["name"] for row in expert_tensors.values()}
    tensor_manifest = []
    for tensor in sorted(tensors, key=lambda row: row["name"]):
        routed = tensor["name"] in routed_names
        tensor_manifest.append(
            {
                "name": tensor["name"],
                "shape": tensor["shape"],
                "type": tensor["type"],
                "type_id": tensor["type_id"],
                "source_offset": tensor["offset"],
                "source_nbytes": tensor["nbytes"],
                "source_payload_sha256": tensor["sha256"],
                "category": "routed_expert" if routed else "non_routed",
                "mapping": "expert_pack_identity" if routed else "gguf_direct_identity",
                "scale_storage": (
                    "inline_quant_block"
                    if tensor["type"] == "MXFP4"
                    else "tensor_native"
                ),
            }
        )

    manifest = {
        "format": FORMAT,
        "version": 1,
        "complete": True,
        "created_at": now(),
        "layout": "triplet_identity",
        "role_order": list(ROLE_NAMES),
        "alignment": args.alignment,
        "pack_path": str(output),
        "pack_size": expected_pack_bytes,
        "pack_sha256": pack_sha256,
        "header_bytes": header.header_bytes,
        "index_count": index_count,
        "index_entry_bytes": header.entry_bytes,
        "index_sha256": index_sha256,
        "data_start": data_start,
        "object_count": object_count,
        "object_payload_bytes": object_payload_bytes,
        "object_stride": object_stride,
        "role_bytes": role_nbytes,
        "payload_bytes": index_count * role_nbytes,
        "padding_bytes": expected_pack_bytes - data_start - index_count * role_nbytes,
        "model": {
            "ref": args.model_ref,
            "model_identity_sha256": args.model_identity_sha256,
            "config_sha256": args.config_sha256,
            "num_layers": args.num_layers,
            "num_routed_experts": args.num_experts,
            "top_k": args.top_k,
            "single_gpu": True,
        },
        "source": {
            "path": str(source),
            "size": source.stat().st_size,
            "sha256": actual_source_sha256,
            "gguf_data_offset": int(reader.data_offset),
            "gguf_metadata_sha256": source_metadata_sha256,
            "inventory_path": str(args.inventory.resolve()) if args.inventory else None,
            "inventory_sha256": inventory_summary.get("inventory_sha256"),
            "tensor_count": len(tensors),
        },
        "coverage": {
            "layers": list(range(args.num_layers)),
            "experts_per_layer": args.num_experts,
            "roles": list(ROLE_NAMES),
            "routed_tensor_count": len(routed_names),
            "non_routed_tensor_count": len(tensors) - len(routed_names),
        },
        "transform": {
            "id": "identity-v1",
            "description": "Contiguous source bytes; no dequantization, requantization, or value transform",
            "reversible": True,
            "tool_sha256": tool_sha256(),
        },
        "builder": {
            "git_sha": git_sha(),
            "python": sys.version,
            "command": " ".join(sys.argv),
            "started_at": started_at,
            "elapsed_s": time.monotonic() - started_monotonic,
        },
        "tensors": tensor_manifest,
    }

    os.replace(partial_path, output)
    write_json_atomic(manifest_path, manifest)
    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = now()
    checkpoint["manifest_sha256"] = sha256_file(manifest_path)
    checkpoint["pack_sha256"] = pack_sha256
    write_json_atomic(checkpoint_path, checkpoint)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an auditable SGLang expert pack from GGUF"
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-ref", default="deepseek-v4-flash")
    parser.add_argument("--model-identity-sha256")
    parser.add_argument("--config-blob", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--safety-margin-gib", type=float, default=16.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.inspect:
        return args
    for name in ("source", "source_sha256", "model_identity_sha256", "config_sha256"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required when building")
    args.manifest = args.manifest or args.output.with_name(
        args.output.name + ".manifest.json"
    )
    args.checkpoint = args.checkpoint or args.output.with_name(
        args.output.name + ".checkpoint.json"
    )
    if args.num_layers <= 0 or args.num_experts <= 0 or args.top_k <= 0:
        parser.error("model dimensions and top-k must be positive")
    align_up(0, args.alignment)
    return args


def main() -> int:
    args = parse_args()
    if args.inspect:
        inspect_pack(args.output, args.limit)
        return 0
    manifest = build(args)
    print(
        json.dumps(
            {
                "pack": manifest["pack_path"],
                "pack_sha256": manifest["pack_sha256"],
                "manifest": str(args.manifest.resolve()),
                "objects": manifest["object_count"],
                "entries": manifest["index_count"],
                "pack_size": manifest["pack_size"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
