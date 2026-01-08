"""
Model File Verifier - Verify model file integrity using SHA256 checksums.

Example command:
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum Qwen/Qwen3-0.6B
"""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IGNORE_PATTERNS = [
    "checksums.json",
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
    expected = _load_checksums_from_hf(repo_id=checksums_source)
    actual = _compute_checksums_from_folder(
        model_path=model_path, filenames=list(expected.keys()), max_workers=max_workers
    )
    _compare_checksums(expected=expected, actual=actual)
    print(f"[ModelFileVerifier] All {len(expected)} files verified successfully.")


def _compare_checksums(*, expected: Dict[str, str], actual: Dict[str, str]) -> None:
    errors = []
    for filename, expected_hash in expected.items():
        if filename not in actual:
            errors.append(f"{filename}: missing")
        elif actual[filename] != expected_hash:
            errors.append(
                f"{filename}: mismatch (expected={expected_hash[:16]}..., actual={actual[filename][:16]}...)"
            )

    if errors:
        raise IntegrityError("Integrity check failed: " + "; ".join(errors))


# ======== Load Checksums ========


def _load_checksums_from_hf(*, repo_id: str) -> Dict[str, str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.ls(repo_id, detail=True)

    checksums = dict(
        r
        for r in map(lambda f: _get_filename_and_checksum_from_hf_file(fs, f), files)
        if r
    )
    if not checksums:
        raise IntegrityError(f"No files found in HF repo {repo_id}.")

    return checksums


def _get_filename_and_checksum_from_hf_file(fs, file_info):
    import fnmatch

    if file_info.get("type") != "file":
        return None

    filename = Path(file_info.get("name", "")).name
    if any(fnmatch.fnmatch(filename, pat) for pat in IGNORE_PATTERNS):
        return None

    lfs_info = file_info.get("lfs")
    if lfs_info and "sha256" in lfs_info:
        return filename, lfs_info["sha256"]

    if "sha256" in file_info:
        return filename, file_info["sha256"]

    content = fs.read_bytes(file_info.get("name", ""))
    return filename, hashlib.sha256(content).hexdigest()


# ======== Compute Checksums ========


def _compute_checksums_from_folder(
    *, model_path: Path, filenames: List[str], max_workers: int
) -> Dict[str, str]:
    from tqdm import tqdm

    def compute_one(filename: str) -> Tuple[str, Optional[str]]:
        full_path = model_path / filename
        if not full_path.exists():
            return filename, None
        sha256 = compute_sha256(file_path=full_path)
        return filename, sha256

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(compute_one, filenames),
                total=len(filenames),
                desc="Computing checksums",
            )
        )

    return {k: v for k, v in results if v is not None}


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


def main():
    parser = argparse.ArgumentParser(
        description="Model File Verifier - Verify model file integrity using checksums"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model directory",
    )
    parser.add_argument(
        "--model-checksum",
        required=True,
        help="HuggingFace repo ID for checksums",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()
    verify(
        model_path=args.model_path,
        checksums_source=args.model_checksum,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
