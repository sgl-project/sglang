"""Prepare a verified, shared MMLU cache for AMD CI runners."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

EXPECTED_ROWS = 14_042
EXPECTED_SIZE = 14_493_567
EXPECTED_SHA256 = "2182319e74add5855b87c4abf4136310b900f8d2772460054681feadb7f2c52f"


def _validate(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != EXPECTED_SIZE:
        return False, f"size={path.stat().st_size}, expected={EXPECTED_SIZE}"

    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
            rows += chunk.count(b"\n")

    actual_sha256 = digest.hexdigest()
    if rows != EXPECTED_ROWS:
        return False, f"rows={rows}, expected={EXPECTED_ROWS}"
    if actual_sha256 != EXPECTED_SHA256:
        return False, f"sha256={actual_sha256}, expected={EXPECTED_SHA256}"
    return True, "verified"


def _generate(staging_root: Path) -> Path:
    # sgl-eval currently has no public prepare-only CLI. Redirect its pinned
    # loader to container-local staging so the shared PVC never sees its
    # non-atomic shutil.move or the 166 MB source archive.
    from sgl_eval.evals import _loader
    from sgl_eval.evals._registry import _TABLE, _build_loader

    _loader._CACHE_ROOT = staging_root
    [mmlu] = [entry for entry in _TABLE if entry["name"] == "mmlu"]
    examples = _build_loader(mmlu)(None)
    if len(examples) != EXPECTED_ROWS:
        raise RuntimeError(
            f"generated {len(examples)} MMLU rows, expected {EXPECTED_ROWS}"
        )
    return staging_root / "mmlu" / "test.jsonl"


def _atomic_publish(source: Path, final_dir: Path) -> None:
    publish_dir = Path(
        tempfile.mkdtemp(prefix=".mmlu-publish-", dir=str(final_dir.parent))
    )
    try:
        published_file = publish_dir / "test.jsonl"
        shutil.copyfile(source, published_file)
        valid, reason = _validate(published_file)
        if not valid:
            raise RuntimeError(f"refusing to publish invalid MMLU cache: {reason}")
        published_file.chmod(0o444)
        publish_dir.chmod(0o555)
        try:
            os.replace(publish_dir, final_dir)
        except OSError:
            # Some shared filesystems do not coordinate advisory locks across
            # hosts. If another publisher won the race, accept only its fully
            # verified artifact; otherwise preserve the original error.
            valid, _ = _validate(final_dir / "test.jsonl")
            if not valid:
                raise
    finally:
        if publish_dir.exists():
            publish_dir.chmod(0o755)
            published_file = publish_dir / "test.jsonl"
            if published_file.exists():
                published_file.chmod(0o644)
            shutil.rmtree(publish_dir)


def _link_local_cache(final_dir: Path, local_root: Path) -> None:
    local_mmlu = local_root / "mmlu"
    local_root.mkdir(parents=True, exist_ok=True)

    if local_mmlu.is_symlink():
        if local_mmlu.resolve() == final_dir.resolve():
            return
        local_mmlu.unlink()
    elif local_mmlu.exists():
        # The container is per-job and disposable. Do not merge a potentially
        # partial local cache into the fleet-wide verified cache.
        shutil.rmtree(local_mmlu)
    local_mmlu.symlink_to(final_dir, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=Path.home() / ".cache" / "sgl_eval",
    )
    parser.add_argument("--sgl-eval-ref", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.sgl_eval_ref):
        raise SystemExit("--sgl-eval-ref must be a full lowercase commit SHA")

    revision_root = args.cache_root / args.sgl_eval_ref
    final_dir = revision_root / "mmlu"
    final_file = final_dir / "test.jsonl"
    revision_root.mkdir(parents=True, exist_ok=True)

    lock_path = revision_root / ".mmlu.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        valid, reason = _validate(final_file)
        if valid:
            print(f"Using verified shared MMLU cache: {final_file}")
        else:
            if final_dir.exists() or final_dir.is_symlink():
                quarantine = revision_root / (
                    f"mmlu.invalid-{int(time.time())}-{os.getpid()}"
                )
                os.replace(final_dir, quarantine)
                print(f"Quarantined invalid MMLU cache ({reason}): {quarantine}")

            with tempfile.TemporaryDirectory(prefix="sgl-eval-mmlu-") as tmp:
                generated = _generate(Path(tmp))
                valid, reason = _validate(generated)
                if not valid:
                    raise RuntimeError(
                        f"generated MMLU cache failed validation: {reason}"
                    )
                _atomic_publish(generated, final_dir)
                print(f"Published verified shared MMLU cache: {final_file}")

        _link_local_cache(final_dir, args.local_cache_root)


if __name__ == "__main__":
    main()
