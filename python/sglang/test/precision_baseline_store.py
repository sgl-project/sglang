"""HF dataset store for nightly precision-regression baselines.

Layout: ``<model>/<YYYY>/<MM>/<DD>/run-<sglang_sha7>/{meta.json,
comparator_report.jsonl, tensors/*.pt}``. Root ``manifest.jsonl`` has one
row per run; rows carry a ``push_index`` so fetch picks the latest push
regardless of file order (prune may rewrite the file).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)


@dataclass
class HfStoreConfig:
    repo: str
    revision: str = "main"

    @classmethod
    def from_env(cls) -> HfStoreConfig:
        repo = os.environ.get("SGLANG_PRECISION_HF_REPO")
        if not repo:
            raise RuntimeError(
                "SGLANG_PRECISION_HF_REPO is not set. The precision baseline "
                "store is required (there is no local-only mode); set the repo "
                "and HF_TOKEN_PRECISION_STORE."
            )
        revision = os.environ.get("SGLANG_PRECISION_HF_REVISION", "main")
        return cls(repo=repo, revision=revision)


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def _today_path() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y/%m/%d")


def _push_index() -> int:
    return time.time_ns()


_T = TypeVar("_T")


def _with_retries(
    op: Callable[[], _T],
    *,
    what: str,
    attempts: int = 3,
    base_delay: float = 2.0,
) -> _T:
    """Exponential backoff on 429/5xx; auth/404 raise immediately."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return op()
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = status is None or status == 429 or 500 <= status < 600
            if not transient or attempt == attempts:
                raise
            last_exc = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"unreachable retry exit for {what}: {last_exc}")


def _row_recency_key(row: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    # Fall back to file position for legacy rows that predate push_index.
    explicit = row.get("push_index")
    try:
        explicit_i = int(explicit) if explicit is not None else -1
    except (TypeError, ValueError):
        explicit_i = -1
    return (explicit_i, fallback_index)


def _select_latest_run(
    rows: list[dict[str, Any]],
    *,
    model: str,
    capture_signature: Optional[str] = None,
) -> Optional[str]:
    # A baseline is only comparable to a target with the same capture shape, so
    # when a signature is given, mismatched (incl. legacy unsigned) rows are
    # skipped — fetch then returns None and the caller establishes a fresh one
    # instead of erroring on incompatible tensors.
    # A failed run must not become the next comparison baseline, or a persistent
    # regression is masked: today's regressed tensors (uploaded as "failed")
    # would be selected as the reference next run. Prefer non-failed rows and
    # fall back to a failed one only when no usable baseline exists.
    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if row.get("model") != model:
            continue
        if (
            capture_signature is not None
            and row.get("capture_signature") != capture_signature
        ):
            continue
        if "run_path" not in row:
            continue
        candidates.append((_row_recency_key(row, idx), row))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0])
    usable = [c for c in candidates if c[1].get("pass_label") != "failed"]
    chosen = usable or candidates
    return chosen[-1][1]["run_path"]


def fetch_latest_baseline(
    *,
    config: HfStoreConfig,
    model: str,
    target_tensors_dir: Path,
    capture_signature: Optional[str] = None,
) -> Optional[str]:
    # Tensors land flat in target_tensors_dir (no enclosing tensors/) so the
    # caller can treat it like a fresh dump dir.
    rows, _ = _read_manifest(config)
    run_path = _select_latest_run(
        rows, model=model, capture_signature=capture_signature
    )
    if run_path is None:
        return None

    snapshot_root = _with_retries(
        lambda: snapshot_download(
            repo_id=config.repo,
            repo_type="dataset",
            revision=config.revision,
            allow_patterns=[f"{run_path}/tensors/*"],
        ),
        what="snapshot download",
    )
    src = Path(snapshot_root) / run_path / "tensors"
    if not src.exists():
        return None

    target_tensors_dir.mkdir(parents=True, exist_ok=True)
    for fp in src.iterdir():
        if fp.is_file():
            shutil.copy2(fp, target_tensors_dir / fp.name)
    return run_path


def _read_manifest(config: HfStoreConfig) -> tuple[list[dict[str, Any]], str]:
    # Skip corrupt rows rather than bricking the store on a partial write.
    try:
        manifest_local = _with_retries(
            lambda: hf_hub_download(
                repo_id=config.repo,
                repo_type="dataset",
                filename="manifest.jsonl",
                revision=config.revision,
            ),
            what="manifest fetch",
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return [], ""

    text = Path(manifest_local).read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, text


_MANIFEST_PROMOTE_KEYS = (
    "hardware",
    "tp_size",
    "pass_label",
    "capture_signature",
    "num_layers_compared",
    "num_layers_passed",
    "num_layers_failed",
    "max_rel_diff",
    "ci_run_id",
)


def push_run(
    *,
    config: HfStoreConfig,
    model: str,
    sglang_commit: str,
    today_tensors_dir: Path,
    meta: dict[str, Any],
    comparator_report: Optional[Path] = None,
    force: bool = False,
) -> str:
    # Dedup: same model+date+sha → skip tensor upload but still refresh meta
    # + comparator_report + append a new manifest row, so pass-1 baseline and
    # pass-2 stats both land. force=True re-uploads tensors too.
    api = HfApi()
    date_str, date_path = _today_path()
    model_sanitized = _sanitize_model_name(model)
    sha7 = (
        sglang_commit[:7] if sglang_commit and sglang_commit != "unknown" else "no_sha"
    )
    run_path = f"{model_sanitized}/{date_path}/run-{sha7}"

    existing_rows, existing_text = _read_manifest(config)
    tensors_already_present = any(r.get("run_path") == run_path for r in existing_rows)
    skip_tensors = tensors_already_present and not force

    with tempfile.TemporaryDirectory() as stage_dir:
        stage = Path(stage_dir)
        run_dir = stage / "run"
        run_dir.mkdir(parents=True)
        if not skip_tensors:
            tensors_out = run_dir / "tensors"
            tensors_out.mkdir()
            for fp in today_tensors_dir.iterdir():
                if fp.is_file() and fp.suffix == ".pt":
                    shutil.copy2(fp, tensors_out / fp.name)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if comparator_report is not None and comparator_report.exists():
            shutil.copy2(comparator_report, run_dir / "comparator_report.jsonl")

        commit_msg = (
            f"refresh meta {sha7} for {model} on {date_str}"
            if skip_tensors
            else f"add run {sha7} for {model} on {date_str}"
        )
        _with_retries(
            lambda: api.upload_folder(
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                folder_path=str(run_dir),
                path_in_repo=run_path,
                commit_message=commit_msg,
            ),
            what="upload_folder",
        )

    manifest_row = {
        "date": date_str,
        "model": model,
        "run_path": run_path,
        "sglang_commit": sglang_commit,
        "push_index": _push_index(),
        **{k: meta.get(k) for k in _MANIFEST_PROMOTE_KEYS if k in meta},
    }

    new_manifest_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_out:
            tmp_out.write(existing_text)
            if existing_text and not existing_text.endswith("\n"):
                tmp_out.write("\n")
            tmp_out.write(json.dumps(manifest_row) + "\n")
            new_manifest_path = tmp_out.name

        _with_retries(
            lambda: api.upload_file(
                path_or_fileobj=new_manifest_path,
                path_in_repo="manifest.jsonl",
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                commit_message=f"manifest += {sha7} {date_str}",
            ),
            what="manifest upload",
        )
    finally:
        if new_manifest_path and os.path.exists(new_manifest_path):
            os.unlink(new_manifest_path)
    return run_path


def prune_old_runs(
    *,
    config: HfStoreConfig,
    model: Optional[str] = None,
    keep_days: int = 30,
    weekly_archive: bool = True,
    dry_run: bool = True,
) -> dict[str, list[str]]:
    # dry_run defaults True because model=None+keep_days=0 would wipe the
    # store. Live mode rewrites the manifest before deleting folders so a
    # mid-run failure leaves manifest pointing at the kept rows only.
    api = HfApi()
    rows, _ = _read_manifest(config)
    if not rows:
        return {"kept": [], "pruned": []}

    cutoff_date = datetime.now(timezone.utc).date()

    def _row_date(row: dict[str, Any]) -> Optional[datetime]:
        try:
            return datetime.strptime(row["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except (KeyError, ValueError):
            return None

    kept_rows: list[dict[str, Any]] = []
    pruned_rows: list[dict[str, Any]] = []

    by_model_week: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            kept_rows.append(row)
            continue
        dt = _row_date(row)
        if dt is None:
            kept_rows.append(row)
            continue
        age_days = (cutoff_date - dt.date()).days
        if age_days <= keep_days:
            kept_rows.append(row)
            continue
        iso_year, iso_week, _ = dt.isocalendar()
        by_model_week.setdefault((row.get("model", ""), iso_year, iso_week), []).append(
            row
        )

    for week_rows in by_model_week.values():
        week_rows.sort(key=lambda r: r.get("date", ""))
        if weekly_archive and week_rows:
            kept_rows.append(week_rows[-1])
            pruned_rows.extend(week_rows[:-1])
        else:
            pruned_rows.extend(week_rows)

    kept_rows.sort(key=lambda r: (r.get("date", ""), r.get("model", "")))

    report = {
        "kept": [r["run_path"] for r in kept_rows if "run_path" in r],
        "pruned": [r["run_path"] for r in pruned_rows if "run_path" in r],
    }
    if dry_run or not pruned_rows:
        return report

    rewritten: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_out:
            for r in kept_rows:
                tmp_out.write(json.dumps(r) + "\n")
            rewritten = tmp_out.name
        _with_retries(
            lambda: api.upload_file(
                path_or_fileobj=rewritten,
                path_in_repo="manifest.jsonl",
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                commit_message=f"manifest -= {len(pruned_rows)} pruned",
            ),
            what="manifest rewrite (prune)",
        )
    finally:
        if rewritten and os.path.exists(rewritten):
            os.unlink(rewritten)

    for r in pruned_rows:
        rp = r.get("run_path")
        if not rp:
            continue
        try:
            api.delete_folder(
                repo_id=config.repo,
                repo_type="dataset",
                path_in_repo=rp,
                revision=config.revision,
                commit_message=f"prune {rp}",
            )
        except Exception:
            # Folder may already be missing; manifest no longer points at it.
            pass

    return report
