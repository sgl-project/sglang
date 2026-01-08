"""
Model File Verifier - Verify model file integrity using SHA256 checksums.

Standalone usage:
    python -m sglang.srt.utils.model_file_verifier generate --model-path /path/to/model --output checksums.json
    python -m sglang.srt.utils.model_file_verifier generate --model-path Qwen/Qwen3-0.6B --output checksums.json
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum checksums.json

As a module:
    from sglang.srt.utils.model_file_verifier import verify, generate_checksums
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

IGNORE_PATTERNS = [
    "checksums.json",
    ".DS_Store",
    "*.lock",
]


# ======== Verify ========


def verify(model_path: str, checksums_source: str, max_workers: int = 4) -> None:
    model_path = Path(model_path).resolve()
    expected = _load_checksums(checksums_source)
    actual = _compute_checksums(model_path, list(expected.keys()), max_workers)
    _compare_checksums(expected, actual)
    print(f"[ModelFileVerifier] All {len(expected)} files verified successfully.")


def _compare_checksums(expected: Dict[str, str], actual: Dict[str, str]) -> None:
    mismatches = []
    missing = []

    for filename, expected_hash in expected.items():
        if filename not in actual:
            missing.append(filename)
        elif actual[filename] != expected_hash:
            mismatches.append(
                {
                    "file": filename,
                    "expected": expected_hash,
                    "actual": actual[filename],
                }
            )

    if missing or mismatches:
        error_parts = []
        if missing:
            error_parts.append(f"Missing files: {missing}")
        if mismatches:
            mismatch_details = "; ".join(
                f"{m['file']} (expected={m['expected'][:16]}..., actual={m['actual'][:16]}...)"
                for m in mismatches
            )
            error_parts.append(f"Checksum mismatches: {mismatch_details}")
        raise IntegrityError(" | ".join(error_parts))


# ======== Generate ========


def generate_checksums(
    source: str, output_path: str, max_workers: int = 4
) -> Dict[str, str]:
    if Path(source).is_dir():
        checksums = _generate_checksums_from_local(source, max_workers)
    else:
        checksums = _load_checksums_from_hf(source)

    output = {"checksums": checksums}
    Path(output_path).write_text(json.dumps(output, indent=2, sort_keys=True))

    print(
        f"[ModelFileVerifier] Generated checksums for {len(checksums)} files -> {output_path}"
    )
    return checksums


def _generate_checksums_from_local(
    model_path: str, max_workers: int
) -> Dict[str, str]:
    model_path = Path(model_path).resolve()
    files = _discover_files(model_path)

    if not files:
        raise IntegrityError(f"No model files found in {model_path}")

    return _compute_checksums(model_path, files, max_workers)


def _discover_files(model_path: Path) -> List[str]:
    import fnmatch

    files = []
    for entry in model_path.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_file():
            continue
        if any(fnmatch.fnmatch(entry.name, pat) for pat in IGNORE_PATTERNS):
            continue
        files.append(entry.name)
    return sorted(files)


# ======== Load Checksums ========


def _load_checksums(source: str) -> Dict[str, str]:
    if Path(source).is_file():
        data = json.loads(Path(source).read_text())
        return data["checksums"]
    return _load_checksums_from_hf(source)


def _load_checksums_from_hf(repo_id: str) -> Dict[str, str]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError:
        raise IntegrityError(
            "huggingface_hub not installed. Install it or provide a local checksums file."
        )

    fs = HfFileSystem()
    checksums = {}
    files_without_checksum = []

    try:
        files = fs.ls(repo_id, detail=True)
    except Exception as e:
        raise IntegrityError(f"Failed to list files from HF repo {repo_id}: {e}")

    for file_info in files:
        if file_info.get("type") != "file":
            continue
        filename = Path(file_info.get("name", "")).name
        lfs_info = file_info.get("lfs")
        if lfs_info and "sha256" in lfs_info:
            checksums[filename] = lfs_info["sha256"]
        elif "sha256" in file_info:
            checksums[filename] = file_info["sha256"]
        else:
            files_without_checksum.append(filename)

    if files_without_checksum:
        raise IntegrityError(
            f"Files without SHA256 checksum in HF repo {repo_id}: {files_without_checksum}. "
            "Generate checksums from local directory instead."
        )

    return checksums


# ======== Compute Checksums ========


def _compute_checksums(
    model_path: Path, filenames: List[str], max_workers: int
) -> Dict[str, str]:
    from tqdm import tqdm

    def compute_one(filename: str) -> Tuple[str, str]:
        full_path = model_path / filename
        sha256 = compute_sha256(full_path)
        return filename, sha256

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(compute_one, filenames),
                total=len(filenames),
                desc="Computing checksums",
            )
        )

    return dict(results)


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


# ======== Exceptions ========


class IntegrityError(Exception):
    pass


# ======== CLI ========


def _cli_generate(args):
    generate_checksums(args.source, args.output, args.workers)


def _cli_verify(args):
    verify(args.model_path, args.model_checksum, args.workers)


def main():
    parser = argparse.ArgumentParser(
        description="Model File Verifier - Verify model file integrity using checksums"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "generate", help="Generate checksums.json for a model"
    )
    gen_parser.add_argument(
        "--source",
        required=True,
        help="Local model directory or HuggingFace repo ID",
    )
    gen_parser.add_argument(
        "--output", required=True, help="Output path for checksums.json"
    )
    gen_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (local only)"
    )
    gen_parser.set_defaults(func=_cli_generate)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify model files against checksums"
    )
    verify_parser.add_argument(
        "--model-path", required=True, help="Path to model directory"
    )
    verify_parser.add_argument(
        "--model-checksum",
        required=True,
        help="Checksums source: JSON file path or HuggingFace repo ID",
    )
    verify_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    verify_parser.set_defaults(func=_cli_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
