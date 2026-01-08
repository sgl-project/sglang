"""
Model File Verifier - Verify model file integrity using SHA256 checksums.

Standalone usage:
    python model_file_verifier.py generate --model-path /path/to/model
    python model_file_verifier.py verify --model-path /path/to/model --checksums checksums.json

As a module:
    from sglang.srt.utils.model_file_verifier import ModelFileVerifier
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ======== Exceptions ========


class IntegrityError(Exception):
    pass


# ======== Core Implementation ========


class ModelFileVerifier:
    CHECKSUM_FILENAME = "checksums.json"
    CHUNK_SIZE = 64 * 1024
    DEFAULT_PATTERNS = [
        "*.safetensors",
        "*.bin",
        "*.json",
        "*.model",
        "*.tiktoken",
        "*.txt",
    ]

    def __init__(
        self,
        model_path: str,
        checksums_source: Optional[str] = None,
        max_workers: int = 4,
    ):
        self.model_path = os.path.abspath(model_path)
        self.checksums_source = checksums_source
        self.max_workers = max_workers

    def verify(self) -> None:
        expected = self._load_expected_checksums()
        if not expected:
            raise IntegrityError(
                f"No checksums found. Provide --checksums or place {self.CHECKSUM_FILENAME} in model directory."
            )

        actual = self._compute_checksums(list(expected.keys()))
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

        print(f"[ModelFileVerifier] All {len(expected)} files verified successfully.")

    def generate_checksums(self, output_path: Optional[str] = None) -> Dict[str, str]:
        files = self._discover_files()
        if not files:
            raise IntegrityError(f"No model files found in {self.model_path}")

        checksums = self._compute_checksums(files)

        output = output_path or os.path.join(self.model_path, self.CHECKSUM_FILENAME)
        with open(output, "w") as f:
            json.dump(checksums, f, indent=2, sort_keys=True)

        print(
            f"[ModelFileVerifier] Generated checksums for {len(checksums)} files -> {output}"
        )
        return checksums

    def _load_expected_checksums(self) -> Dict[str, str]:
        if self.checksums_source:
            if os.path.isfile(self.checksums_source):
                return self._load_checksums_from_file(self.checksums_source)
            else:
                return self._load_checksums_from_hf(self.checksums_source)

        default_path = os.path.join(self.model_path, self.CHECKSUM_FILENAME)
        if os.path.isfile(default_path):
            return self._load_checksums_from_file(default_path)

        return {}

    def _load_checksums_from_file(self, path: str) -> Dict[str, str]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
            return data
        raise IntegrityError(f"Invalid checksums file format: {path}")

    def _load_checksums_from_hf(self, repo_id: str) -> Dict[str, str]:
        try:
            from huggingface_hub import HfFileSystem
        except ImportError:
            raise IntegrityError(
                "huggingface_hub not installed. Install it or provide a local checksums file."
            )

        fs = HfFileSystem()
        checksums = {}

        try:
            files = fs.ls(repo_id, detail=True)
        except Exception as e:
            raise IntegrityError(f"Failed to list files from HF repo {repo_id}: {e}")

        for file_info in files:
            if file_info.get("type") != "file":
                continue
            filename = os.path.basename(file_info.get("name", ""))
            lfs_info = file_info.get("lfs")
            if lfs_info and "sha256" in lfs_info:
                checksums[filename] = lfs_info["sha256"]
            elif "sha256" in file_info:
                checksums[filename] = file_info["sha256"]

        if not checksums:
            raise IntegrityError(
                f"No SHA256 checksums found in HF repo {repo_id}. "
                "Only LFS files have checksums. Generate a local checksums.json instead."
            )

        return checksums

    def _discover_files(self) -> List[str]:
        import fnmatch

        files = []
        for entry in os.listdir(self.model_path):
            if entry.startswith("."):
                continue
            full_path = os.path.join(self.model_path, entry)
            if not os.path.isfile(full_path):
                continue
            for pattern in self.DEFAULT_PATTERNS:
                if fnmatch.fnmatch(entry, pattern):
                    files.append(entry)
                    break
        return sorted(files)

    def _compute_checksums(self, filenames: List[str]) -> Dict[str, str]:
        results = {}

        def compute_one(filename: str) -> Tuple[str, str]:
            full_path = os.path.join(self.model_path, filename)
            sha256 = compute_sha256(full_path)
            return filename, sha256

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(compute_one, f): f for f in filenames}
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    name, checksum = future.result()
                    results[name] = checksum
                    print(
                        f"  [{len(results)}/{len(filenames)}] {name}: {checksum[:16]}..."
                    )
                except FileNotFoundError:
                    pass
                except Exception as e:
                    raise IntegrityError(
                        f"Failed to compute checksum for {filename}: {e}"
                    )

        return results


# ======== Utility Functions ========


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


# ======== CLI ========


def main():
    parser = argparse.ArgumentParser(
        description="Model File Verifier - Verify model file integrity using SHA256 checksums"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "generate", help="Generate checksums.json for a model"
    )
    gen_parser.add_argument(
        "--model-path", required=True, help="Path to model directory"
    )
    gen_parser.add_argument(
        "--output", help="Output path (default: <model-path>/checksums.json)"
    )
    gen_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify model files against checksums"
    )
    verify_parser.add_argument(
        "--model-path", required=True, help="Path to model directory"
    )
    verify_parser.add_argument(
        "--model-checksum",
        nargs="?",
        const="",
        default=None,
        help="Checksums source: JSON file path or HuggingFace repo ID. "
        "If specified without value, uses <model-path>/checksums.json",
    )
    verify_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    try:
        checksums_source = getattr(args, "model_checksum", None)
        if checksums_source == "":
            checksums_source = os.path.join(
                args.model_path, ModelFileVerifier.CHECKSUM_FILENAME
            )

        verifier = ModelFileVerifier(
            model_path=args.model_path,
            checksums_source=checksums_source,
            max_workers=args.workers,
        )

        if args.command == "generate":
            verifier.generate_checksums(args.output)
        elif args.command == "verify":
            verifier.verify()

    except IntegrityError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
