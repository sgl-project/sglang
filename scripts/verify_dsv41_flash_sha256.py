#!/usr/bin/env python3
"""Verify DeepSeek-V4.1-Flash shards against official HuggingFace LFS SHA256.

Hashes come from the Hub tree (LFS oid), frozen next to the model as SHA256SUMS.
Does not download. Missing shards are reported, not hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODEL = Path(os.environ.get("MODEL_PATH", os.path.expanduser("~/models/DeepSeek-V4.1-Flash")))
TREE = MODEL / ".cache/huggingface/trees/dba1be0a40aa45a94ad051997016db3960a90277.json"
SUMS = MODEL / "SHA256SUMS"
GOT = MODEL / "SHA256SUMS.got"
LOG = MODEL / "SHA256SUMS.log"


def expected() -> dict[str, tuple[str, int]]:
    if TREE.is_file():
        files = json.loads(TREE.read_text())["files"]
        out = {}
        for i in range(1, 49):
            name = f"model-{i:05d}-of-00048.safetensors"
            meta = files[name]
            out[name] = (meta["lfs_sha256"], int(meta["lfs_size"]))
        return out
    if not SUMS.is_file():
        raise SystemExit(f"need {TREE} or {SUMS}")
    out = {}
    for line in SUMS.read_text().splitlines():
        sha, name = line.split()
        out[name] = (sha, -1)
    return out


def sha256_file(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    exp = expected()
    SUMS.write_text("".join(f"{sha}  {name}\n" for name, (sha, _) in exp.items()))
    got_lines: list[str] = []
    ok = miss = size_bad = hash_bad = 0
    log = []

    def emit(msg: str) -> None:
        log.append(msg)
        print(msg, flush=True)

    only = sys.argv[1:]  # optional shard names
    names = only or list(exp)
    for name in names:
        if name not in exp:
            emit(f"UNKNOWN  {name}")
            hash_bad += 1
            continue
        sha, size = exp[name]
        path = MODEL / name
        if not path.is_file():
            emit(f"MISSING  {name}")
            miss += 1
            continue
        have = path.stat().st_size
        if size >= 0 and have != size:
            emit(f"SIZE     {name} have={have} want={size}")
            size_bad += 1
            continue
        emit(f"HASHING  {name} ({have} bytes)")
        got = sha256_file(path)
        got_lines.append(f"{got}  {name}")
        if got != sha:
            emit(f"MISMATCH {name}")
            emit(f"  want {sha}")
            emit(f"  got  {got}")
            hash_bad += 1
        else:
            emit(f"OK       {name}")
            ok += 1
    if got_lines:
        prev = GOT.read_text() if GOT.is_file() else ""
        # replace lines for hashed names, keep others
        keep = {}
        for line in prev.splitlines():
            if not line.strip():
                continue
            g, n = line.split()
            keep[n] = g
        for line in got_lines:
            g, n = line.split()
            keep[n] = g
        GOT.write_text("".join(f"{keep[n]}  {n}\n" for n in sorted(keep)))
    LOG.write_text("\n".join(log) + "\n")
    emit(f"RESULT ok={ok} missing={miss} size_fail={size_bad} hash_fail={hash_bad}")
    return 0 if (miss + size_bad + hash_bad) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
