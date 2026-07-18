#!/usr/bin/env python3
"""Rewrite per-class ``kv_size_thres`` in e2e tests from CI logs.

Each **test class** gets its own floor (subclasses can override). Multi-class
files must not share one value — attribution uses::

    [CI Test Method] ClassName.test_method

KV allocations that appear before a class's first method (setUpClass) are
assigned to that class. Floor = min(observations for that class) * factor
so multi-launch / PD prefill+decode within one class still share one thres.

::

    class TestFoo(CustomTestCase):
        kv_size_thres = 12345.6  # auto; update_memory_thresholds.py

    class TestBar(CustomTestCase):
        kv_size_thres = 800.0  # auto; update_memory_thresholds.py

Usage:
    python3 scripts/ci/utils/update_memory_thresholds.py --dry-run
    python3 scripts/ci/utils/update_memory_thresholds.py --run-id ...
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
# Printed by CustomTestCase before each test method (after setUpClass).
CI_METHOD_RE = re.compile(
    r"\[CI Test Method\]\s+(?P<cls>[A-Za-z_][A-Za-z0-9_]*)\.(?P<meth>[A-Za-z_][A-Za-z0-9_]*)"
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

_AUTO_COMMENT = "# auto; update_memory_thresholds.py"
_AUTO_LINE_RE = re.compile(
    rf"^[ \t]*{CLASS_ATTR}\s*=\s*.+?[ \t]*{_AUTO_COMMENT}[ \t]*\n?",
    re.M,
)

# Observation: (test_file, class_name, gpu_family, kv_size_mb)
Obs = Tuple[str, str, str, float]


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


def estimate_kv_size_mb_from_line(line: str) -> Optional[float]:
    m = KV_SIZE_MB_RE.search(line)
    if m:
        return float(m.group("mb"))

    kv_gb = None
    m = KV_GB_RE.search(line)
    if m:
        if m.group("kv") is not None:
            kv_gb = float(m.group("kv"))
        else:
            kv_gb = float(m.group("k")) + float(m.group("v"))
    m = SWA_GB_RE.search(line)
    if m:
        v = float(m.group("v"))
        kv_gb = v if kv_gb is None else max(kv_gb, v)

    mamba_gb = 0.0
    mm = MAMBA_GB_RE.search(line)
    if mm:
        mamba_gb = float(mm.group("conv")) + float(mm.group("ssm"))

    if kv_gb is None and mamba_gb <= 0:
        return None
    return round(((kv_gb or 0.0) + mamba_gb) * 1024.0, 1)


def parse_job_log(text: str, *, job_name: str) -> List[Obs]:
    """Per-class KV observations for one job log."""
    suite_m = SUITE_FROM_RUN_SUITE_RE.search(text)
    suite = suite_m.group("suite") if suite_m else ""
    gpu = gpu_family_from_text(job_name) or gpu_family_from_text(suite) or "unknown"
    if gpu == "unknown":
        return []

    current_file: Optional[str] = None
    # KV sizes seen since the last [CI Test Method] (setUpClass of the next class).
    pending: List[float] = []
    out: List[Obs] = []

    for line in text.splitlines():
        m = TEST_START_RE.search(line)
        if m:
            current_file = normalize_test_file(m.group("path"))
            pending = []
            continue

        if current_file is None:
            continue

        mb = estimate_kv_size_mb_from_line(line)
        if mb is not None and mb > 0:
            pending.append(mb)
            continue

        cm = CI_METHOD_RE.search(line)
        if cm:
            cls_name = cm.group("cls")
            if pending:
                # One setup batch → one sample (min across TP ranks / multi-launch).
                out.append((current_file, cls_name, gpu, min(pending)))
                pending = []
            continue

    return out


def collect(run_ids: Sequence[str], cache_dir: Path) -> List[Obs]:
    obs: List[Obs] = []
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
            print(f"    +{len(got)} class samples", flush=True)
            obs.extend(got)
    return obs


def aggregate(obs: List[Obs], factor: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    """test_file -> class_name -> gpu_family -> floor."""
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for tf, cls, gpu, mb in obs:
        buckets[(tf, cls, gpu)].append(mb)

    result: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (tf, cls, gpu), vals in buckets.items():
        result[tf][cls][gpu] = round(sum(vals) / len(vals) * factor, 1)
    return result


def format_value(by_gpu: Dict[str, float]) -> str:
    if len(by_gpu) == 1:
        return str(next(iter(by_gpu.values())))
    inner = ", ".join(f'"{g}": {by_gpu[g]}' for g in sorted(by_gpu.keys()))
    return "{" + inner + "}"


def strip_legacy_module_blocks(src: str) -> str:
    for begin, end in _LEGACY_BLOCKS:
        if begin not in src or end not in src:
            continue
        pre, rest = src.split(begin, 1)
        _, post = rest.split(end, 1)
        pre, post = pre.rstrip("\n"), post.lstrip("\n")
        src = (pre + "\n\n" + post) if pre and post else pre + post
    return src


def strip_auto_class_attrs(src: str) -> str:
    """Remove prior auto-seeded ``kv_size_thres`` lines."""
    return _AUTO_LINE_RE.sub("", src)


def _insert_lineno_0based(node: ast.ClassDef) -> int:
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    ):
        if len(node.body) == 1:
            return first.end_lineno or first.lineno
        first = node.body[1]
        if isinstance(first, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if first.decorator_list:
                return first.decorator_list[0].lineno - 1
        return first.lineno - 1

    if isinstance(first, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if first.decorator_list:
            return first.decorator_list[0].lineno - 1
    return first.lineno - 1


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


def inject_per_class(src: str, class_floors: Dict[str, Dict[str, float]]) -> str:
    """Write ``kv_size_thres`` only on classes present in ``class_floors``."""
    src = strip_legacy_module_blocks(src)
    src = strip_auto_class_attrs(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    lines = src.splitlines(keepends=True)
    classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in class_floors
    ]
    for node in sorted(classes, key=lambda n: n.lineno, reverse=True):
        value_expr = format_value(class_floors[node.name])
        indent = "    "
        if node.body:
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
        "--strip-only",
        action="store_true",
        help="Only remove auto-seeded kv_size_thres lines from test files",
    )
    args = p.parse_args(argv)

    if args.strip_only:
        n = 0
        for path in sorted((REPO_ROOT / "test").rglob("*.py")):
            old = path.read_text(encoding="utf-8", errors="ignore")
            if CLASS_ATTR not in old and "KV_SIZE_THRES" not in old:
                continue
            new = strip_auto_class_attrs(strip_legacy_module_blocks(old))
            if new != old:
                print(
                    ("would " if args.dry_run else "")
                    + f"strip {path.relative_to(REPO_ROOT)}"
                )
                if not args.dry_run:
                    path.write_text(new, encoding="utf-8")
                n += 1
        print(f"stripped {n} files")
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

    # Also strip auto attrs from files that no longer appear (stale floors).
    touched_files = set(agg.keys())
    n_strip_stale = 0
    for path in sorted((REPO_ROOT / "test/registered").rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in touched_files:
            continue
        old = path.read_text(encoding="utf-8", errors="ignore")
        if _AUTO_COMMENT not in old:
            continue
        new = strip_auto_class_attrs(strip_legacy_module_blocks(old))
        if new != old:
            print(("would " if args.dry_run else "") + f"strip-stale {rel}")
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
            n_strip_stale += 1

    n = 0
    skipped_no_launch = 0
    for tf, class_floors in sorted(agg.items()):
        path = REPO_ROOT / tf
        if not path.is_file():
            print("skip missing", tf)
            continue
        if not file_launches_server(path):
            skipped_no_launch += 1
            continue
        old = path.read_text(encoding="utf-8")
        new = inject_per_class(old, class_floors)
        if new == old:
            print("unchanged", tf, list(class_floors.keys()))
            continue
        print(
            ("would " if args.dry_run else "")
            + f"update {tf} classes={list(class_floors.keys())}"
        )
        if not args.dry_run:
            path.write_text(new, encoding="utf-8")
        n += 1
    print(
        f"updated {n} files (skipped_no_launch={skipped_no_launch}, "
        f"strip_stale={n_strip_stale})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
