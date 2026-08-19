from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "expert_pack"
sys.path.insert(0, str(TOOLS))

from format import ROLE_NAMES, read_header, read_index  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_synthetic_gguf(path: Path, layers: int, experts: int) -> None:
    writer = GGUFWriter(path, "llama")
    writer.add_block_count(layers)
    writer.add_tensor(
        "token_embd.weight", np.arange(24, dtype=np.float32).reshape(6, 4)
    )
    for layer in range(layers):
        for role_index, role in enumerate(ROLE_NAMES):
            values = np.arange(experts * 8, dtype=np.float32).reshape(experts, 2, 4)
            values = values + layer * 1000 + role_index * 100
            writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_synthetic_gguf_round_trip_and_corruption_detection(tmp_path: Path) -> None:
    layers = 2
    experts = 4
    source = tmp_path / "synthetic.gguf"
    ollama_manifest = tmp_path / "ollama-manifest.json"
    config_blob = tmp_path / "config.json"
    pack = tmp_path / "synthetic.expert-pack"
    manifest = tmp_path / "synthetic.expert-pack.manifest.json"
    report = tmp_path / "validation.json"
    create_synthetic_gguf(source, layers, experts)
    ollama_manifest.write_text('{"synthetic":true}\n', encoding="utf-8")
    config_blob.write_text('{"architecture":"synthetic"}\n', encoding="utf-8")

    build_command = [
        sys.executable,
        str(TOOLS / "build.py"),
        "--source",
        str(source),
        "--source-sha256",
        sha256(source),
        "--output",
        str(pack),
        "--manifest",
        str(manifest),
        "--ollama-manifest",
        str(ollama_manifest),
        "--ollama-manifest-digest",
        sha256(ollama_manifest),
        "--config-blob",
        str(config_blob),
        "--config-sha256",
        sha256(config_blob),
        "--model-ref",
        "synthetic/deepseek-v4",
        "--num-layers",
        str(layers),
        "--num-experts",
        str(experts),
        "--top-k",
        "2",
        "--safety-margin-gib",
        "0",
    ]
    subprocess.run(build_command, check=True, capture_output=True, text=True)

    validate_command = [
        sys.executable,
        str(TOOLS / "validate.py"),
        "--pack",
        str(pack),
        "--manifest",
        str(manifest),
        "--source",
        str(source),
        "--source-range-samples",
        str(layers * experts * len(ROLE_NAMES)),
        "--full",
        "--report",
        str(report),
    ]
    subprocess.run(validate_command, check=True, capture_output=True, text=True)
    validation = json.loads(report.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["index_count"] == layers * experts * len(ROLE_NAMES)
    assert validation["full_pack_entry_hash_count"] == validation["index_count"]
    assert validation["source_range_compare_count"] == validation["index_count"]

    with pack.open("rb", buffering=0) as pack_stream, source.open(
        "rb", buffering=0
    ) as source_stream:
        header = read_header(pack_stream)
        entries = read_index(pack_stream, header)
        assert header.num_layers == layers
        assert header.num_experts == experts
        for entry in entries:
            source_stream.seek(entry.source_slice_offset)
            pack_stream.seek(entry.pack_offset)
            assert source_stream.read(entry.source_slice_nbytes) == pack_stream.read(
                entry.pack_nbytes
            )

    subprocess.run(
        [sys.executable, str(TOOLS / "build.py"), "--output", str(pack), "--inspect"],
        check=True,
        capture_output=True,
        text=True,
    )

    with pack.open("r+b", buffering=0) as stream:
        stream.seek(entries[0].pack_offset)
        original = stream.read(1)
        stream.seek(entries[0].pack_offset)
        stream.write(bytes([original[0] ^ 0xFF]))
        stream.flush()
        os.fsync(stream.fileno())
    corrupted = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "validate.py"),
            "--pack",
            str(pack),
            "--manifest",
            str(manifest),
            "--source-range-samples",
            "0",
            "--full-pack-entry-hashes",
        ],
        capture_output=True,
        text=True,
    )
    assert corrupted.returncode != 0
    assert "pack payload checksum mismatch" in corrupted.stderr
