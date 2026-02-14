"""
Model File Verifier - Verify model file integrity using SHA256 checksums.

Example commands:
    # Verify using HuggingFace model online metadata
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum Qwen/Qwen3-0.6B

    # Verify using locally generated checksum
    python -m sglang.srt.utils.model_file_verifier generate --model-path <hf-id-or-model-path> --model-checksum checksums.json
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum checksums.json
"""

import argparse
import fnmatch
import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======== Data Format ========


@dataclass
class FileInfo:
    sha256: str
    size: int


@dataclass
class Manifest:
    files: Dict[str, FileInfo]

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if "checksums" in data:
            warnings.warn(
                "The 'checksums' format is deprecated. "
                "Please regenerate with the latest version to use the new 'files' format.",
                DeprecationWarning,
                stacklevel=3,
            )
            return cls(
                files={
                    k: FileInfo(sha256=v, size=-1) for k, v in data["checksums"].items()
                }
            )
        return cls(files={k: FileInfo(**v) for k, v in data["files"].items()})

    def to_dict(self) -> dict:
        return asdict(self)


# ======== Constants ========


IGNORE_PATTERNS = [
    ".DS_Store",
    "*.lock",
    ".gitattributes",
    "LICENSE",
    "LICENSE.*",
    "README.md",
    "README.*",
    "NOTICE",
]


# ======== Verify ========


def verify(*, model_path: str, checksums_source: str, max_workers: int = 4) -> None:
    model_path = Path(model_path).resolve()
    expected = _load_checksums(checksums_source)
    actual = _compute_manifest_from_folder(
        model_path=model_path,
        filenames=list(expected.files.keys()),
        max_workers=max_workers,
    )
    _compare_manifests(expected=expected, actual=actual)
    print(f"[ModelFileVerifier] All {len(expected.files)} files verified successfully.")


def _compare_manifests(*, expected: Manifest, actual: Manifest) -> None:
    errors = []
    for filename, exp in expected.files.items():
        if filename not in actual.files:
            errors.append(f"{filename}: missing (expected size={exp.size})")
        elif actual.files[filename].sha256 != exp.sha256:
            act = actual.files[filename]
            errors.append(
                f"{filename}: mismatch (expected={exp.sha256[:16]}... size={exp.size}, actual={act.sha256[:16]}... size={act.size})"
            )

    if errors:
        raise IntegrityError("Integrity check failed: " + "; ".join(errors))


# ======== Generate ========


def generate_checksums(
    *, source: str, output_path: str, max_workers: int = 4
) -> Manifest:
    if Path(source).is_dir():
        model_path = Path(source).resolve()
        files = _discover_files(model_path)
        if not files:
            raise IntegrityError(f"No model files found in {model_path}")
        manifest = _compute_manifest_from_folder(
            model_path=model_path, filenames=files, max_workers=max_workers
        )
    else:
        manifest = Manifest(files=_load_file_infos_from_hf(repo_id=source))

    Path(output_path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )

    print(
        f"[ModelFileVerifier] Generated checksums for {len(manifest.files)} files -> {output_path}"
    )
    return manifest


def _discover_files(model_path: Path) -> List[str]:
    return sorted(
        e.name
        for e in model_path.iterdir()
        if e.is_file()
        and not e.name.startswith(".")
        and not any(fnmatch.fnmatch(e.name, p) for p in IGNORE_PATTERNS)
    )


# ======== Load Checksums ========


def _load_checksums(source: str) -> Manifest:
    if Path(source).is_file():
        data = json.loads(Path(source).read_text())
        return Manifest.from_dict(data)
    return Manifest(files=_load_file_infos_from_hf(repo_id=source))


def _load_file_infos_from_hf(*, repo_id: str) -> Dict[str, FileInfo]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.ls(repo_id, detail=True)

    file_infos = dict(
        r for r in map(lambda f: _get_filename_and_info_from_hf_file(fs, f), files) if r
    )
    if not file_infos:
        raise IntegrityError(f"No files found in HF repo {repo_id}.")

    return file_infos


def _get_filename_and_info_from_hf_file(
    fs, file_info
) -> Optional[Tuple[str, FileInfo]]:
    if file_info.get("type") != "file":
        return None

    filename = Path(file_info.get("name", "")).name
    if any(fnmatch.fnmatch(filename, pat) for pat in IGNORE_PATTERNS):
        return None

    size = file_info.get("size", -1)
    lfs_info = file_info.get("lfs")
    if lfs_info and "sha256" in lfs_info:
        return filename, FileInfo(sha256=lfs_info["sha256"], size=size)

    if "sha256" in file_info:
        return filename, FileInfo(sha256=file_info["sha256"], size=size)

    content = fs.read_bytes(file_info.get("name", ""))
    return filename, FileInfo(
        sha256=hashlib.sha256(content).hexdigest(), size=len(content)
    )


# ======== Compute Checksums ========


def _compute_manifest_from_folder(
    *, model_path: Path, filenames: List[str], max_workers: int
) -> Manifest:
    from tqdm import tqdm

    def compute_one(filename: str) -> Tuple[str, Optional[FileInfo]]:
        full_path = model_path / filename
        if not full_path.exists():
            return filename, None
        sha256 = compute_sha256(file_path=full_path)
        size = full_path.stat().st_size
        return filename, FileInfo(sha256=sha256, size=size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(compute_one, filenames),
                total=len(filenames),
                desc="Computing checksums",
            )
        )

    return Manifest(files={k: v for k, v in results if v is not None})


def compute_sha256(*, file_path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


# ======== Exceptions ========


class IntegrityError(Exception):
    pass


# ======== CLI ========


def _add_common_args(parser):
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model directory or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model-checksum",
        required=True,
        help="Checksums JSON file path",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Model File Verifier - Verify model file integrity using checksums"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "generate", help="Generate checksums.json for a model"
    )
    _add_common_args(gen_parser)
    gen_parser.set_defaults(
        func=lambda args: generate_checksums(
            source=args.model_path,
            output_path=args.model_checksum,
            max_workers=args.workers,
        )
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify model files against checksums"
    )
    _add_common_args(verify_parser)
    verify_parser.set_defaults(
        func=lambda args: verify(
            model_path=args.model_path,
            checksums_source=args.model_checksum,
            max_workers=args.workers,
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
