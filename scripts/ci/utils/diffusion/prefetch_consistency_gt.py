#!/usr/bin/env python3
"""
Prefetch the pinned ascend consistency-GT directory for the NPU diffusion CI.

Materializes sgl-project/ci-data-diffusion at the revision pinned in
``test_utils.py`` into a per-revision directory under a persistent cache root,
verifies that every consistency-checked NPU case can find its GT there, and
exports the path as ``SGLANG_CONSISTENCY_GT_DIR`` so the test run reads GT from
disk instead of fetching each file from raw.githubusercontent.

Run it after the dependency install step, from the repository root.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CACHE_ROOT = "~/.cache/sglang/consistency-gt"
DEFAULT_KEEP_DAYS = 14.0

# Written last and checked instead of the directory itself: a job killed
# mid-extract otherwise leaves a partial tree that the next run treats as a hit.
MARKER_NAME = ".complete"


def _resolve_gt_source() -> tuple[str, str, str]:
    """Return (repo, revision, in-repo GT path) from the constants the tests use."""
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.test.test_utils import (
        SGL_TEST_FILES_CI_DATA_REPO,
        SGL_TEST_FILES_CI_DATA_REVISION,
        SGL_TEST_FILES_CONSISTENCY_GT_BASE,
    )

    # Off NPU the base URL resolves to the parent sglang_generated directory,
    # whose files carry the same <case>_<n>gpu.<ext> names as ascend; prefetching
    # it would compare NPU output against CUDA GT without any check failing.
    if not current_platform.is_npu():
        raise SystemExit(
            "Error: this prefetch is NPU-only; current platform is not NPU"
        )

    prefix = (
        "https://raw.githubusercontent.com/"
        f"{SGL_TEST_FILES_CI_DATA_REPO}/{SGL_TEST_FILES_CI_DATA_REVISION}/"
    )
    if not SGL_TEST_FILES_CONSISTENCY_GT_BASE.startswith(prefix):
        raise SystemExit(
            "Error: the GT base URL no longer starts with the pinned raw prefix. "
            "Update this script together with test_utils.py."
        )

    return (
        SGL_TEST_FILES_CI_DATA_REPO,
        SGL_TEST_FILES_CI_DATA_REVISION,
        SGL_TEST_FILES_CONSISTENCY_GT_BASE[len(prefix) :],
    )


def _github_url(repo: str, suffix: str) -> str:
    # These runners reach github.com only through the infra proxy; the prefix is
    # applied the same way scripts/ci/npu/npu_ci_install_dependency.sh applies it.
    proxy = os.environ.get("GITHUB_PROXY_URL", "")
    return f"{proxy}https://github.com/{repo}{suffix}"


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _fetch_sparse_git(repo: str, revision: str, repo_path: str, dest: Path) -> bool:
    """Fetch only the GT subdirectory. False if the proxy does not pass git."""
    url = _github_url(repo, ".git")
    probe = subprocess.run(
        ["git", "ls-remote", url, "HEAD"],
        capture_output=True,
        timeout=180,
    )
    if probe.returncode != 0:
        print(
            "git ls-remote failed through the proxy, falling back to the tarball:\n"
            f"{probe.stderr.decode(errors='replace').strip()}",
            flush=True,
        )
        return False

    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "--quiet", str(dest)])
    _run(["git", "-C", str(dest), "remote", "add", "origin", url])
    # Written directly rather than via `sparse-checkout set`, which wants a HEAD
    # this repository does not have yet.
    _run(["git", "-C", str(dest), "config", "core.sparseCheckout", "true"])
    sparse_file = dest / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(f"{repo_path}/*\n", encoding="utf-8")

    # blob:none defers blob transfer to the checkout, and the sparse file limits
    # the checkout, so only the ascend blobs are ever sent.
    _run(
        [
            "git",
            "-C",
            str(dest),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            revision,
        ]
    )
    _run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"])
    shutil.rmtree(dest / ".git")
    return True


def _fetch_tarball(repo: str, revision: str, repo_path: str, dest: Path) -> None:
    """Fetch the whole tree at the revision and keep only the GT subdirectory."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "archive.tar.gz"
    _run(
        [
            "wget",
            "--quiet",
            "--tries=3",
            "-O",
            str(archive),
            _github_url(repo, f"/archive/{revision}.tar.gz"),
        ]
    )
    # gzip carries a CRC per member, so a truncated download fails extraction
    # here rather than surfacing later as an unreadable image. strip-components
    # drops the "<repo>-<sha>/" prefix so the layout matches the git path.
    _run(
        [
            "tar",
            "-xzf",
            str(archive),
            "-C",
            str(dest),
            "--strip-components=1",
            "--wildcards",
            f"*/{repo_path}/*",
        ]
    )
    archive.unlink()


def _iter_consistency_cases():
    from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import (
        PARAMETRIZED_CASE_GROUPS,
    )

    for suite, case_groups in PARAMETRIZED_CASE_GROUPS.items():
        for _, case_group in case_groups:
            for case in case_group:
                if case.run_consistency_check:
                    yield suite, case


def _verify_gt_dir(gt_dir: Path) -> list[str]:
    """Return one problem line per case whose GT is missing from the directory."""
    from sglang.multimodal_gen.test.test_utils import (
        get_consistency_gt_candidates,
        gt_exists,
    )

    # The lookups below read the directory through the same env var the tests
    # use, so this asks literally "would the test find its GT here?".
    os.environ["SGLANG_CONSISTENCY_GT_DIR"] = str(gt_dir)

    problems: list[str] = []
    checked = 0
    for suite, case in _iter_consistency_cases():
        checked += 1
        num_gpus = case.server_args.num_gpus
        is_video = case.server_args.modality == "video"
        output_format = case.sampling_params.output_format
        candidates = get_consistency_gt_candidates(
            case.id, num_gpus, is_video, output_format
        )
        if not gt_exists(
            case.id, num_gpus, is_video=is_video, output_format=output_format
        ):
            problems.append(f"{suite}/{case.id}: no GT among {', '.join(candidates)}")

    print(f"Verified GT for {checked} consistency-checked case(s)", flush=True)
    return problems


def _prune(cache_root: Path, keep_dir: Path, keep_days: float) -> None:
    # Age-based rather than "keep only the current revision": a PR that bumps the
    # pin runs alongside main-branch jobs on the same pool, and the two would
    # delete each other's tree on every run.
    cutoff = time.time() - keep_days * 86400
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir() or entry == keep_dir:
            continue
        marker = entry / MARKER_NAME
        # No marker means a crashed fetch or an interrupted prune.
        if marker.exists() and marker.stat().st_mtime >= cutoff:
            continue
        print(f"Pruning stale GT cache entry: {entry}", flush=True)
        shutil.rmtree(entry, ignore_errors=True)


def _install(tmp_dir: Path, final_dir: Path) -> None:
    (tmp_dir / MARKER_NAME).touch()
    try:
        tmp_dir.rename(final_dir)
    except OSError:
        # Another partition on this host installed the same revision first.
        print(f"Another job installed {final_dir.name} first, discarding our copy")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _fetch_into(repo: str, revision: str, repo_path: str, tmp_dir: Path) -> None:
    if not _fetch_sparse_git(repo, revision, repo_path, tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _fetch_tarball(repo, revision, repo_path, tmp_dir)


def _ensure_gt_dir(repo: str, revision: str, repo_path: str, cache_root: Path) -> Path:
    final_dir = cache_root / revision
    marker = final_dir / MARKER_NAME

    if marker.exists():
        problems = _verify_gt_dir(final_dir / repo_path)
        if not problems:
            marker.touch()
            print(f"Using cached GT for {revision}", flush=True)
            return final_dir / repo_path
        print(
            "Cached GT failed verification, refetching:\n  " + "\n  ".join(problems),
            flush=True,
        )
        shutil.rmtree(final_dir, ignore_errors=True)

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_root / f"{revision}.tmp.{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)

    _fetch_into(repo, revision, repo_path, tmp_dir)
    tmp_gt_dir = tmp_dir / repo_path
    if not tmp_gt_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(f"Error: {repo_path} missing from {repo}@{revision}")

    problems = _verify_gt_dir(tmp_gt_dir)
    if problems:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(
            "Error: the prefetched GT directory is incomplete:\n  "
            + "\n  ".join(problems)
        )

    _install(tmp_dir, final_dir)
    return final_dir / repo_path


def _export(gt_dir: Path) -> None:
    line = f"SGLANG_CONSISTENCY_GT_DIR={gt_dir}"
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch pinned ascend consistency GT for the NPU diffusion CI"
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("SGLANG_CONSISTENCY_GT_CACHE_ROOT", DEFAULT_CACHE_ROOT),
        help=f"Persistent cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--keep-days",
        type=float,
        default=DEFAULT_KEEP_DAYS,
        help=f"Delete unused revisions older than this (default: {DEFAULT_KEEP_DAYS})",
    )
    args = parser.parse_args()

    repo, revision, repo_path = _resolve_gt_source()
    cache_root = Path(args.cache_root).expanduser().resolve()
    print(f"GT source: {repo}@{revision}/{repo_path}")
    print(f"Cache root: {cache_root}")

    gt_dir = _ensure_gt_dir(repo, revision, repo_path, cache_root)
    _prune(cache_root, cache_root / revision, args.keep_days)
    _export(gt_dir)


if __name__ == "__main__":
    sys.exit(main())
