"""Download Hugging Face repos into this runner's cache for a PR test job.

Runs from the workflow's own commit in the step that holds HF_TOKEN, before
the job runs PR code. REPO_IDS comes from check-changes, which PR code
controls, so an id that is not a public model on the Hub is skipped.

    REPO_IDS="org/model-a org/model-b" python3 -S prefetch_hf_models.py

A problem is a warning, never a failure.
"""

import fnmatch
import glob
import hashlib
import json
import os
import re
import sys
from typing import List, Optional

POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prefetch_policy.json"
)

# HF caps a repo name at 96 characters and rejects `--` and `..` in it.
REPO_ID_RE = re.compile(
    r"^(?!.*(--|\.\.))[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"
)

# Formats no test loads: Meta's original/ checkpoints, GGUF, and ONNX / OpenVINO
# / TF / Flax / TFLite exports. A repo's `patterns` in prefetch_policy.json
# replace this filter.
_SKIPPED_DIRS = ("original/", "onnx/", "openvino/")
_SKIPPED_SUFFIXES = (".pth", ".gguf", ".onnx", ".h5", ".msgpack", ".tflite")


def warn(message: str) -> None:
    print(f"::warning::prefetch_hf_models: {message}", flush=True)


def files_to_download(
    repo_files: List[str], patterns: Optional[List[str]]
) -> List[str]:
    """The repo files matching ``patterns``, else the files a test can load.
    Where a directory has safetensors, loaders take those, so its `.bin` /
    `.pt` copies are left out."""
    if patterns is not None:
        return [f for f in repo_files if any(fnmatch.fnmatch(f, p) for p in patterns)]
    safetensors_dirs = {
        os.path.dirname(f) for f in repo_files if f.endswith(".safetensors")
    }
    return [
        f
        for f in repo_files
        if not f.startswith(_SKIPPED_DIRS)
        and not f.endswith(_SKIPPED_SUFFIXES)
        and not (f.endswith((".bin", ".pt")) and os.path.dirname(f) in safetensors_dirs)
    ]


def fetch_repo(api, repo_id: str, revision: str, patterns: Optional[List[str]]) -> None:
    from filelock import FileLock
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.model_info(repo_id, revision=revision)
    except RepositoryNotFoundError:
        # The scan also yields id-shaped strings that are not models.
        print(f"skip {repo_id}: not a public model on the Hub", flush=True)
        return
    # ModelInfo.private is None when the Hub does not say: only an explicit
    # False is public.
    if info.private is not False:
        warn(f"skip {repo_id}: not confirmed public")
        return
    files = files_to_download([s.rfilename for s in info.siblings], patterns)
    print(f"{repo_id}@{revision}: {len(files)} file(s)", flush=True)
    # The lock sglang's CI loader holds while it downloads a repo into this
    # cache (_get_lock_file_path in ci_weight_validation.py): its cleanup
    # deletes .incomplete files, ours included, unless we hold it too.
    key_hash = hashlib.sha256(repo_id.encode()).hexdigest()[:16]
    lock_path = os.path.join(
        constants.HF_HUB_CACHE, ".sglang_locks", f"download_{key_hash}.lock"
    )
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with FileLock(lock_path, mode=0o666):
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[glob.escape(f) for f in files],
            # sglang's CI loader downloads single-threaded too: parallel
            # writers raced on .incomplete files in the shared cache.
            max_workers=1,
        )


def main() -> int:
    from huggingface_hub import HfApi

    with open(POLICY_PATH, encoding="utf-8") as f:
        policy = json.load(f)["repos"]
    api = HfApi()
    for repo_id in sorted(set(os.environ.get("REPO_IDS", "").split())):
        if not REPO_ID_RE.match(repo_id):
            warn(f"skip malformed repo id {repo_id!r}")
            continue
        entry = policy.get(repo_id, {})
        if "skip" in entry:
            print(f"skip {repo_id}: {entry['skip']}", flush=True)
            continue
        for revision in entry.get("revisions", ["main"]):
            try:
                fetch_repo(api, repo_id, revision, entry.get("patterns"))
            except Exception as exc:
                warn(f"{repo_id}@{revision}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
