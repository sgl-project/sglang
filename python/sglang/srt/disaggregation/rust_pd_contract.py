"""PD-CANON-V1 artifact and layout fingerprints used during pair bootstrap."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MODEL_DOMAIN = "SGLANG-PD-MODEL-MANIFEST-V1"
TOKENIZER_DOMAIN = "SGLANG-PD-TOKENIZER-MANIFEST-V1"
LAYOUT_DOMAIN = "SGLANG-PD-LAYOUT-V1"
NATIVE_DOMAIN = "SGLANG-PD-NATIVE-ABI-V1"

MODEL_REQUIRED = ("config.json", "generation_config.json")
TOKENIZER_REQUIRED = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
FROZEN_ARTIFACT = {
    "config.json": (
        727,
        "4505bef6e4f2328440d54e7149815cc8663d4baae02ee0246ec5d51ae64ead5c",
    ),
    "generation_config.json": (
        239,
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    ),
    "model.safetensors": (
        1_503_300_328,
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    ),
    "tokenizer.json": (
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    "merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
}
MODEL_DIGEST = "a9a9ba010b3a083e0e8485267bef35e6b9d482ab2386701dda94aba302ee0855"
TOKENIZER_DIGEST = "35feb1c5fa61e27b74c321872c805301e35b92008b578af0c1251147b696719e"
LAYOUT_DIGEST = "f7640ee0e4c00a3d019dd0635636a585fbbc90de1509983ba6e0633d04d5e7f3"
NATIVE_DIGEST = "09b2dcde40291c67b3f5876417ac613663b017f88450eae31c8aaec097b8dca2"


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    digest: bytes


def contract_digests(
    model_root: str | Path,
    tokenizer_root: str | Path,
) -> tuple[str, str, str, str]:
    model = _frozen_manifest_digest(MODEL_DOMAIN, _model_files(Path(model_root)))
    tokenizer = _frozen_manifest_digest(
        TOKENIZER_DOMAIN,
        _required_files(Path(tokenizer_root), TOKENIZER_REQUIRED),
    )
    layout = layout_fingerprint()
    native = native_abi_digest()
    if (
        model != MODEL_DIGEST
        or tokenizer != TOKENIZER_DIGEST
        or layout != LAYOUT_DIGEST
        or native != NATIVE_DIGEST
    ):
        raise ValueError("PD_UNSUPPORTED")
    return model, tokenizer, layout, native


def manifest_digest(domain: str, files: Iterable[Path]) -> str:
    entries = [_manifest_entry(path) for path in files]
    entries.sort(key=lambda entry: entry.path.encode("utf-8"))
    if not entries or len({entry.path for entry in entries}) != len(entries):
        raise ValueError("PD_UNSUPPORTED")
    body = bytearray(_domain(domain))
    body.extend(_u32(len(entries)))
    for entry in entries:
        body.extend(_string(entry.path))
        body.extend(_u64(entry.size))
        body.extend(entry.digest)
    return hashlib.sha256(body).hexdigest()


def _frozen_manifest_digest(domain: str, files: Iterable[Path]) -> str:
    files = list(files)
    entries = [_manifest_entry(path) for path in files]
    expected_names = (
        set(MODEL_REQUIRED) | {"model.safetensors"}
        if domain == MODEL_DOMAIN
        else set(TOKENIZER_REQUIRED)
    )
    if {entry.path for entry in entries} != expected_names:
        raise ValueError("PD_UNSUPPORTED")
    for entry in entries:
        expected = FROZEN_ARTIFACT.get(entry.path)
        if expected != (entry.size, entry.digest.hex()):
            raise ValueError("PD_UNSUPPORTED")
    return manifest_digest(domain, files)


def layout_fingerprint() -> str:
    body = bytearray(_domain(LAYOUT_DOMAIN))
    for value in ("bf16", "NHD", "flashinfer", "aux-v1", "completion-v1"):
        body.extend(_string(value))
    for value in (28, 8, 128, 2048, 64, 1, 1, 1, 58):
        body.extend(_u32(value))
    for region_id in range(58):
        body.extend(_u16(region_id))
        if region_id < 28:
            body.extend(_string("key"))
            body.extend(_u16(region_id))
        elif region_id < 56:
            body.extend(_string("value"))
            body.extend(_u16(region_id - 28))
        elif region_id == 56:
            body.extend(_string("aux"))
            body.extend(_u16(0))
        else:
            body.extend(_string("completion"))
            body.extend(_u16(0))
    return hashlib.sha256(body).hexdigest()


def native_abi_digest() -> str:
    body = bytearray(_domain(NATIVE_DOMAIN))
    for value in (
        "0.3.11.post1",
        "e9c61075720039bcfc5fffd19f847608402be3d0",
        "1c128925bc63839fca0fce3cfacd84a400f10a7891bdf9fa86840261ee6e299d",
    ):
        body.extend(_string(value))
    return hashlib.sha256(body).hexdigest()


def _model_files(root: Path) -> list[Path]:
    required = _required_files(root, MODEL_REQUIRED)
    weights = sorted(root.glob("*.safetensors"))
    indexes = sorted(root.glob("*.safetensors.index.json"))
    remote_code = sorted(root.glob("*.py"))
    if not weights:
        raise ValueError("PD_UNSUPPORTED")
    return [*required, *weights, *indexes, *remote_code]


def _required_files(root: Path, names: Iterable[str]) -> list[Path]:
    if not root.is_dir():
        raise ValueError("PD_UNSUPPORTED")
    files = [root / name for name in names]
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("PD_UNSUPPORTED")
    return files


def _manifest_entry(path: Path) -> ManifestEntry:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return ManifestEntry(path.name, size, digest.digest())


def _domain(value: str) -> bytes:
    return value.encode("ascii") + b"\0"


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _u32(len(encoded)) + encoded


def _u16(value: int) -> bytes:
    return struct.pack(">H", value)


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)
