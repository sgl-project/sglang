#!/usr/bin/env python3
"""Rewrite class-level ``kv_size_thres`` in e2e tests from CI logs.

Mines scheduled/nightly logs for KV allocation lines (or explicit
``kv_size_mb``) and writes a class attribute on each Test* class::

    class TestFoo(CustomTestCase):
        kv_size_thres = 12345.6  # auto; update_memory_thresholds.py
        # or multi-runner:
        kv_size_thres = {"h200": 12000.0, "b200": 18000.0}

Floor = min(observations) * factor (default 0.99) so multi-launch and
PD prefill+decode can share one threshold. Subclasses can override.

Only injects into files that launch a server
(``popen_launch_server`` / ``popen_launch_pd_server`` /
``PDDisaggregationServerBase``).

Usage:
    python3 scripts/ci/utils/update_memory_thresholds.py --dry-run
    python3 scripts/ci/utils/update_memory_thresholds.py --run-id ...
    python3 scripts/ci/utils/update_memory_thresholds.py --migrate-only
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.test.memory_threshold import (  # noqa: E402
    CLASS_ATTR,
    gpu_family_from_text,
)

REPO = "sgl-project/sglang"
PR_TEST_WORKFLOW = "pr-test.yml"
NIGHTLY_WORKFLOW = "nightly-test-nvidia.yml"
DEFAULT_FACTOR = 0.99

TEST_START_RE = re.compile(
    r"python3\s+(?:\S*?/)?(?P<path>test/(?:registered|manual)/\S+\.py)"
)
SUITE_FROM_RUN_SUITE_RE = re.compile(
    r"run_suite\.py\b[^\n]*?--suite\s+(?P<suite>[^\s\\]+)"
)
KV_SIZE_MB_RE = re.compile(
    r"(?:kv_size_mb|memory_usage\.kv_size_mb)[=:\s]+(?P<mb>[\d.]+)",
    re.I,
)
KV_GB_RE = re.compile(
    r"KV Cache is allocated\.[^\n]*?(?:KV size:\s*(?P<kv>[\d.]+)\s*GB|"
    r"K size:\s*(?P<k>[\d.]+)\s*GB,\s*V size:\s*(?P<v>[\d.]+)\s*GB)"
)
SWA_GB_RE = re.compile(r"SWAKVPool mem usage:\s*(?P<v>[\d.]+)\s*GB", re.I)
MAMBA_GB_RE = re.compile(
    r"(?:Mamba Cache is allocated|max_mamba_cache_size)[^\n]*?"
    r"conv_state size:\s*(?P<conv>[\d.]+)\s*GB,?\s*"
    r"ssm_state size:\s*(?P<ssm>[\d.]+)\s*GB",
    re.I,
)
LAUNCH_MARKERS = (
    "popen_launch_server",
    "popen_launch_pd_server",
    "PDDisaggregationServerBase",
)

# Legacy module-level auto blocks.
_LEGACY_BLOCKS = (
    (
        "# --- KV_SIZE_THRES begin (auto; update_memory_thresholds.py) ---",
        "# --- KV_SIZE_THRES end ---",
    ),
    (
        "# --- MIN_KV_BUFFER_MB begin (auto; update_memory_thresholds.py) ---",
        "# --- MIN_KV_BUFFER_MB end ---",
    ),
    (
        "# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---",
        "# --- MEMORY_CAPACITY_FLOORS end ---",
    ),
)

_AUTO_COMMENT = f"# auto; update_memory_thresholds.py"


def _run(cmd: List[str], *, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def list_recent_runs(workflow: str, *, event: Optional[str] = None, limit: int = 3):
    q = f"repos/{REPO}/actions/workflows/{workflow}/runs?per_page={limit}&branch=main"
    if event:
        q += f"&event={event}"
    data = json.loads(_run(["gh", "api", q]))
    return [r for r in data.get("workflow_runs", []) if r.get("status") == "completed"]


def list_jobs(run_id) -> list:
    jobs, page = [], 1
    while True:
        data = json.loads(
            _run(
                [
                    "gh",
                    "api",
                    f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100&page={page}",
                ]
            )
        )
        batch = data.get("jobs", [])
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def download_job_log(job_id, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["gh", "run", "view", f"--job={job_id}", "--log"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout:
        return False
    dest.write_text(r.stdout, errors="replace")
    return True


def normalize_test_file(path: str) -> str:
    path = path.replace("\\", "/")
    for marker in ("/sglang/test/", "test/registered/", "test/manual/", "test/"):
        idx = path.find(marker)
        if idx >= 0:
            if marker.startswith("/sglang/"):
                return path[idx + len("/sglang/") :]
            return path[idx:]
    return path.lstrip("./")


def file_launches_server(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(m in text for m in LAUNCH_MARKERS)


def estimate_kv_size_mb_from_chunk(text: str) -> Optional[float]:
    m = KV_SIZE_MB_RE.search(text)
    if m:
        return float(m.group("mb"))

    kv_vals: List[float] = []
    for m in KV_GB_RE.finditer(text):
        if m.group("kv") is not None:
            kv_vals.append(float(m.group("kv")))
        else:
            kv_vals.append(float(m.group("k")) + float(m.group("v")))
    for m in SWA_GB_RE.finditer(text):
        kv_vals.append(float(m.group("v")))
    kv_gb = max(kv_vals) if kv_vals else None

    mamba_gb = 0.0
    for mm in MAMBA_GB_RE.finditer(text):
        mamba_gb = max(mamba_gb, float(mm.group("conv")) + float(mm.group("ssm")))

    if kv_gb is None and mamba_gb <= 0:
        return None
    return round(((kv_gb or 0.0) + mamba_gb) * 1024.0, 1)


def parse_job_log(text: str, *, job_name: str) -> List[Tuple[str, str, float]]:
    suite_m = SUITE_FROM_RUN_SUITE_RE.search(text)
    suite = suite_m.group("suite") if suite_m else ""
    gpu = gpu_family_from_text(job_name) or gpu_family_from_text(suite) or "unknown"
    if gpu == "unknown":
        return []

    current = None
    chunks: Dict[str, List[str]] = defaultdict(list)
    order: List[str] = []
    for line in text.splitlines():
        m = TEST_START_RE.search(line)
        if m:
            current = normalize_test_file(m.group("path"))
            if current not in chunks:
                order.append(current)
            continue
        if current:
            chunks[current].append(line)

    out: List[Tuple[str, str, float]] = []
    for tf in order:
        body = "\n".join(chunks[tf])
        parts = re.split(r"(?=KV Cache is allocated\.)", body)
        vals: List[float] = []
        for part in parts:
            mb = estimate_kv_size_mb_from_chunk(part)
            if mb is not None:
                vals.append(mb)
        if not vals:
            mb = estimate_kv_size_mb_from_chunk(body)
            if mb is not None:
                vals.append(mb)
        if vals:
            out.append((tf, gpu, min(vals)))
    return out


def collect(run_ids: Sequence[str], cache_dir: Path) -> List[Tuple[str, str, float]]:
    obs: List[Tuple[str, str, float]] = []
    for run_id in run_ids:
        print(f"run {run_id}", flush=True)
        for j in list_jobs(run_id):
            name = j.get("name", "")
            if "gpu" not in name.lower() and "nightly" not in name.lower():
                continue
            dest = cache_dir / f"run_{run_id}" / f"job_{j['id']}.txt"
            print(f"  job {j['id']} {name}", flush=True)
            if not download_job_log(j["id"], dest):
                print("    download failed", flush=True)
                continue
            text = dest.read_text(errors="replace")
            got = parse_job_log(text, job_name=name)
            print(f"    +{len(got)} samples", flush=True)
            obs.extend(got)
    return obs


def aggregate(
    obs: List[Tuple[str, str, float]], factor: float
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for tf, gpu, mb in obs:
        buckets[(tf, gpu)].append(mb)

    result: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (tf, gpu), vals in buckets.items():
        result[tf][gpu] = round(sum(vals) / len(vals) * factor, 1)
    return result


def format_value(by_gpu: Dict[str, float]) -> str:
    if len(by_gpu) == 1:
        return str(next(iter(by_gpu.values())))
    inner = ", ".join(f'"{g}": {by_gpu[g]}' for g in sorted(by_gpu.keys()))
    return "{" + inner + "}"


def strip_legacy_module_blocks(src: str) -> Tuple[str, Optional[str]]:
    """Remove legacy module auto-blocks; return (src, extracted_value_expr)."""
    extracted = None
    for begin, end in _LEGACY_BLOCKS:
        if begin not in src or end not in src:
            continue
        pre, rest = src.split(begin, 1)
        mid, post = rest.split(end, 1)
        # mid like "\n# gpu=...\nKV_SIZE_THRES = 1.2\n"
        m = re.search(
            r"(?:KV_SIZE_THRES|MIN_KV_BUFFER_MB|MEMORY_CAPACITY_FLOORS)\s*=\s*(.+)",
            mid,
            re.S,
        )
        if m:
            # take first assignment expression only (may be multi-line dict)
            expr = m.group(1).strip()
            # if multi-line dict, mid already has full body between begin/end
            assign_m = re.search(
                r"(?:KV_SIZE_THRES|MIN_KV_BUFFER_MB)\s*=\s*(.+?)\s*$",
                mid.strip(),
                re.S | re.M,
            )
            if assign_m:
                extracted = assign_m.group(1).strip()
            else:
                extracted = expr
        pre, post = pre.rstrip("\n"), post.lstrip("\n")
        src = (pre + "\n\n" + post) if pre and post else pre + post
    return src, extracted


def _class_has_setupclass(node: ast.ClassDef) -> bool:
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "setUpClass":
                return True
    return False


def _class_targets(node: ast.ClassDef) -> bool:
    """Whether this class should receive kv_size_thres."""
    if node.name.startswith("Test"):
        return True
    if _class_has_setupclass(node):
        return True
    return False


def _find_existing_kv_assign(
    node: ast.ClassDef,
) -> Optional[ast.Assign | ast.AnnAssign]:
    for item in node.body:
        if isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name) and t.id == CLASS_ATTR:
                    return item
        if isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and item.target.id == CLASS_ATTR:
                return item
    return None


def _insert_lineno_0based(node: ast.ClassDef) -> int:
    """Line index (0-based) at which to insert a new class body statement.

    Must not land between a decorator and its function.
    """
    first = node.body[0]
    # After docstring.
    if (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    ):
        # If docstring is the only stmt... still insert after it.
        if len(node.body) == 1:
            return first.end_lineno or first.lineno
        first = node.body[1]
        after_doc = first  # may be decorated
        if isinstance(after_doc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if after_doc.decorator_list:
                return after_doc.decorator_list[0].lineno - 1
        return after_doc.lineno - 1

    if isinstance(first, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if first.decorator_list:
            return first.decorator_list[0].lineno - 1
    return first.lineno - 1


def inject_class_attr(src: str, value_expr: str) -> str:
    """Set ``kv_size_thres = <value_expr>`` on each eligible class."""
    src, _ = strip_legacy_module_blocks(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    lines = src.splitlines(keepends=True)
    # Process classes bottom-up so line numbers stay valid.
    classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and _class_targets(n)
    ]
    for node in sorted(classes, key=lambda n: n.lineno, reverse=True):
        indent = "    "
        if node.body:
            # Prefer indent from a non-decorator body line.
            sample = lines[node.body[0].lineno - 1]
            m = re.match(r"^(\s*)", sample)
            if m and m.group(1):
                indent = m.group(1)

        new_line = f"{indent}{CLASS_ATTR} = {value_expr}  {_AUTO_COMMENT}\n"
        existing = _find_existing_kv_assign(node)
        if existing is not None:
            start = existing.lineno - 1
            end = (existing.end_lineno or existing.lineno) - 1
            lines[start : end + 1] = [new_line]
            continue

        insert_at = _insert_lineno_0based(node)
        lines.insert(insert_at, new_line)
    return "".join(lines)


def inject(src: str, by_gpu: Dict[str, float]) -> str:
    return inject_class_attr(src, format_value(by_gpu))


def migrate_file(path: Path) -> bool:
    """Convert legacy module KV_SIZE_THRES block → class kv_size_thres."""
    old = path.read_text(encoding="utf-8")
    stripped, extracted = strip_legacy_module_blocks(old)
    if extracted is None:
        # Already class-level or nothing to do.
        if CLASS_ATTR in old and "KV_SIZE_THRES" not in old:
            return False
        if "KV_SIZE_THRES" not in old:
            return False
        # Loose module assign without markers.
        m = re.search(r"^KV_SIZE_THRES\s*=\s*(.+)$", old, re.M)
        if not m:
            return False
        extracted = m.group(1).strip()
        stripped = re.sub(r"^KV_SIZE_THRES\s*=\s*.+\n?", "", old, flags=re.M)

    new = inject_class_attr(stripped, extracted)
    if new != old:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", action="append", default=[])
    p.add_argument("--limit-runs", type=int, default=3)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "sglang_memory_threshold_logs",
    )
    p.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only convert legacy module KV_SIZE_THRES blocks to class attrs",
    )
    args = p.parse_args(argv)

    if args.migrate_only:
        n = 0
        for path in sorted((REPO_ROOT / "test").rglob("*.py")):
            if "KV_SIZE_THRES" not in path.read_text(encoding="utf-8", errors="ignore"):
                continue
            if args.dry_run:
                print("would migrate", path.relative_to(REPO_ROOT))
                n += 1
                continue
            if migrate_file(path):
                print("migrated", path.relative_to(REPO_ROOT))
                n += 1
        print(f"migrated {n} files")
        return 0

    import shutil

    if not shutil.which("gh"):
        print("gh CLI required", file=sys.stderr)
        return 1

    run_ids = args.run_id
    if not run_ids:
        for r in list_recent_runs(
            PR_TEST_WORKFLOW, event="schedule", limit=args.limit_runs
        ):
            run_ids.append(str(r["id"]))
        for r in list_recent_runs(NIGHTLY_WORKFLOW, limit=args.limit_runs):
            if r.get("head_branch") == "main":
                run_ids.append(str(r["id"]))
    obs = collect(run_ids, args.cache_dir)
    print(f"observations={len(obs)}", flush=True)
    if not obs:
        return 1
    agg = aggregate(obs, args.factor)
    n = 0
    skipped_no_launch = 0
    for tf, by_gpu in sorted(agg.items()):
        path = REPO_ROOT / tf
        if not path.is_file():
            print("skip missing", tf)
            continue
        if not file_launches_server(path):
            skipped_no_launch += 1
            continue
        old = path.read_text(encoding="utf-8")
        new = inject(old, by_gpu)
        if new == old:
            print("unchanged", tf, by_gpu.keys())
            continue
        print(("would " if args.dry_run else "") + f"update {tf} {dict(by_gpu)}")
        if not args.dry_run:
            path.write_text(new, encoding="utf-8")
        n += 1
    print(f"updated {n} files (skipped_no_launch={skipped_no_launch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
