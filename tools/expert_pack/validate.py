#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

try:
    from .format import (
        ROLE_NAMES,
        IndexEntry,
        read_header,
        read_index,
        sha256_file,
    )
except ImportError:
    from format import (  # type: ignore[no-redef]
        ROLE_NAMES,
        IndexEntry,
        read_header,
        read_index,
        sha256_file,
    )


CHUNK_BYTES = 16 * 1024 * 1024


def hash_range(stream, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = stream.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise EOFError(f"short read at offset {offset}; {remaining} bytes remain")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def compare_ranges(source, pack, entry: IndexEntry) -> None:
    remaining = entry.pack_nbytes
    source_offset = entry.source_slice_offset
    pack_offset = entry.pack_offset
    while remaining:
        length = min(remaining, CHUNK_BYTES)
        source.seek(source_offset)
        pack.seek(pack_offset)
        source_data = source.read(length)
        pack_data = pack.read(length)
        if len(source_data) != length or len(pack_data) != length:
            raise EOFError(f"short source/pack read for entry {entry.key}")
        if source_data != pack_data:
            raise ValueError(f"source/pack bytes differ for entry {entry.key}")
        source_offset += length
        pack_offset += length
        remaining -= length


def validate(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    pack_path = args.pack.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "SGLANG-EXPERTPACK-v1" or not manifest.get("complete"):
        raise ValueError("manifest is not a complete SGLANG-EXPERTPACK-v1 manifest")
    if Path(manifest["pack_path"]).resolve() != pack_path:
        raise ValueError("manifest pack path does not match --pack")
    if pack_path.stat().st_size != int(manifest["pack_size"]):
        raise ValueError("pack size does not match manifest")

    source_path = (
        args.source.resolve(strict=True)
        if args.source is not None
        else Path(manifest["source"]["path"]).resolve(strict=True)
    )
    if source_path.stat().st_size != int(manifest["source"]["size"]):
        raise ValueError("source size does not match manifest")

    with pack_path.open("rb", buffering=0) as pack:
        header = read_header(pack)
        index_start = header.header_bytes
        pack.seek(index_start)
        raw_index = pack.read(header.index_count * header.entry_bytes)
        if len(raw_index) != header.index_count * header.entry_bytes:
            raise ValueError("pack index is truncated")
        if hashlib.sha256(raw_index).hexdigest() != manifest["index_sha256"]:
            raise ValueError("pack index SHA-256 does not match manifest")
        entries = read_index(pack, header)

    model = manifest["model"]
    source = manifest["source"]
    for actual, expected, field in (
        (
            header.model_identity_sha256,
            model["model_identity_sha256"],
            "model identity digest",
        ),
        (header.source_blob_sha256, source["sha256"], "source digest"),
        (header.config_sha256, model["config_sha256"], "config digest"),
        (header.num_layers, model["num_layers"], "layer count"),
        (header.num_experts, model["num_routed_experts"], "expert count"),
        (header.top_k, model["top_k"], "top-k"),
        (header.index_count, manifest["index_count"], "index count"),
        (header.data_start, manifest["data_start"], "data start"),
        (header.alignment, manifest["alignment"], "alignment"),
    ):
        if actual != expected:
            raise ValueError(f"pack header {field} does not match manifest")

    expected_keys = {
        (layer, expert, role)
        for layer in range(header.num_layers)
        for expert in range(header.num_experts)
        for role in ROLE_NAMES
    }
    by_key = {(entry.layer, entry.expert, entry.role): entry for entry in entries}
    if len(by_key) != len(entries) or set(by_key) != expected_keys:
        raise ValueError("pack index does not have exact layer/expert/role coverage")

    tensor_map = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    non_routed = [
        tensor for tensor in manifest["tensors"] if tensor["category"] == "non_routed"
    ]
    routed = [
        tensor
        for tensor in manifest["tensors"]
        if tensor["category"] == "routed_expert"
    ]
    if len(non_routed) != manifest["coverage"]["non_routed_tensor_count"]:
        raise ValueError("non-routed tensor coverage does not match manifest summary")
    if len(routed) != manifest["coverage"]["routed_tensor_count"]:
        raise ValueError("routed tensor coverage does not match manifest summary")

    ranges = []
    object_stride = int(manifest["object_stride"])
    for layer in range(header.num_layers):
        for expert in range(header.num_experts):
            object_entries = [by_key[(layer, expert, role)] for role in ROLE_NAMES]
            expected_object_start = (
                header.data_start
                + (layer * header.num_experts + expert) * object_stride
            )
            if object_entries[0].pack_offset != expected_object_start:
                raise ValueError(
                    f"object {(layer, expert)} is not at its expected aligned offset"
                )
            if expected_object_start % header.alignment:
                raise ValueError(f"object {(layer, expert)} is not aligned")
            cursor = expected_object_start
            generations = set()
            for entry in object_entries:
                entry.pack()
                tensor = tensor_map.get(entry.tensor_name)
                if tensor is None or tensor["category"] != "routed_expert":
                    raise ValueError(
                        f"entry {entry.key} does not map to a routed tensor"
                    )
                if (
                    entry.pack_offset != cursor
                    or entry.pack_nbytes != entry.source_slice_nbytes
                ):
                    raise ValueError(
                        f"entry {entry.key} breaks identity triplet layout"
                    )
                if (
                    entry.transform_id != "identity-v1"
                    or entry.checksum != entry.source_slice_sha256
                ):
                    raise ValueError(
                        f"entry {entry.key} is not an auditable identity transform"
                    )
                if entry.source_tensor_offset != tensor["source_offset"]:
                    raise ValueError(f"entry {entry.key} source tensor offset mismatch")
                if entry.source_tensor_nbytes != tensor["source_nbytes"]:
                    raise ValueError(f"entry {entry.key} source tensor size mismatch")
                if entry.source_tensor_sha256 != tensor["source_payload_sha256"]:
                    raise ValueError(f"entry {entry.key} source tensor hash mismatch")
                expected_slice_offset = (
                    entry.source_tensor_offset + expert * entry.source_slice_nbytes
                )
                if entry.source_slice_offset != expected_slice_offset:
                    raise ValueError(f"entry {entry.key} source slice offset mismatch")
                if entry.source_slice_offset + entry.source_slice_nbytes > (
                    entry.source_tensor_offset + entry.source_tensor_nbytes
                ):
                    raise ValueError(f"entry {entry.key} source slice is out of bounds")
                ranges.append(
                    (
                        entry.pack_offset,
                        entry.pack_offset + entry.pack_nbytes,
                        entry.key,
                    )
                )
                generations.add(entry.generation)
                cursor += entry.pack_nbytes
            if len(generations) != 1:
                raise ValueError(
                    f"object {(layer, expert)} has inconsistent generations"
                )
            if cursor > expected_object_start + object_stride:
                raise ValueError(f"object {(layer, expert)} exceeds its stride")

    ranges.sort()
    previous_end = header.data_start
    for start, end, key in ranges:
        if start < previous_end or end > pack_path.stat().st_size:
            raise ValueError(f"overlapping or out-of-range pack entry {key}")
        previous_end = end

    bytes_hashed = 0
    pack_hash_ok = None
    if args.full_pack_hash:
        pack_hash_ok = sha256_file(pack_path) == manifest["pack_sha256"]
        bytes_hashed += pack_path.stat().st_size
        if not pack_hash_ok:
            raise ValueError("full pack SHA-256 does not match manifest")

    entry_hash_count = 0
    if args.full_pack_entry_hashes:
        with pack_path.open("rb", buffering=0) as pack:
            for entry in sorted(entries, key=lambda value: value.pack_offset):
                if (
                    hash_range(pack, entry.pack_offset, entry.pack_nbytes)
                    != entry.checksum
                ):
                    raise ValueError(
                        f"pack payload checksum mismatch for entry {entry.key}"
                    )
                bytes_hashed += entry.pack_nbytes
                entry_hash_count += 1

    source_tensor_hash_count = 0
    if args.full_source_tensor_hashes:
        with source_path.open("rb", buffering=0) as source_stream:
            for tensor in sorted(
                manifest["tensors"], key=lambda value: value["source_offset"]
            ):
                digest = hash_range(
                    source_stream,
                    int(tensor["source_offset"]),
                    int(tensor["source_nbytes"]),
                )
                if digest != tensor["source_payload_sha256"]:
                    raise ValueError(
                        f"source tensor hash mismatch for {tensor['name']}"
                    )
                bytes_hashed += int(tensor["source_nbytes"])
                source_tensor_hash_count += 1

    source_file_hash_ok = None
    if args.full_source_file_hash:
        source_file_hash_ok = sha256_file(source_path) == source["sha256"]
        bytes_hashed += source_path.stat().st_size
        if not source_file_hash_ok:
            raise ValueError("full source file SHA-256 does not match manifest")

    sample_count = min(args.source_range_samples, len(entries))
    sampled_entries = []
    if sample_count:
        seed = int(source["sha256"][:16], 16)
        sampled_entries = random.Random(seed).sample(entries, sample_count)
        with (
            source_path.open("rb", buffering=0) as source_stream,
            pack_path.open("rb", buffering=0) as pack_stream,
        ):
            for entry in sampled_entries:
                compare_ranges(source_stream, pack_stream, entry)
                bytes_hashed += entry.pack_nbytes * 2

    elapsed_s = time.monotonic() - started
    result = {
        "status": "PASS",
        "pack": str(pack_path),
        "manifest": str(manifest_path),
        "source": str(source_path),
        "layers": header.num_layers,
        "experts_per_layer": header.num_experts,
        "top_k": header.top_k,
        "index_count": len(entries),
        "object_count": header.num_layers * header.num_experts,
        "non_routed_tensor_count": len(non_routed),
        "routed_tensor_count": len(routed),
        "full_pack_hash": pack_hash_ok,
        "full_pack_entry_hash_count": entry_hash_count,
        "full_source_tensor_hash_count": source_tensor_hash_count,
        "full_source_file_hash": source_file_hash_ok,
        "source_range_compare_count": len(sampled_entries),
        "bytes_verified": bytes_hashed,
        "elapsed_s": elapsed_s,
        "verified_mib_s": bytes_hashed / 1024**2 / elapsed_s if bytes_hashed else None,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SGLANG-EXPERTPACK-v1")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-range-samples", type=int, default=96)
    parser.add_argument("--full-pack-hash", action="store_true")
    parser.add_argument("--full-pack-entry-hashes", action="store_true")
    parser.add_argument("--full-source-tensor-hashes", action="store_true")
    parser.add_argument("--full-source-file-hash", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.source_range_samples < 0:
        parser.error("--source-range-samples must be non-negative")
    if args.full:
        args.full_pack_hash = True
        args.full_pack_entry_hashes = True
        args.full_source_tensor_hashes = True
        args.full_source_file_hash = True
    return args


def main() -> int:
    args = parse_args()
    print(json.dumps(validate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
