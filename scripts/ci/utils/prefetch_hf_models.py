"""Download the Hugging Face repos one CI partition's tests need.

PR test jobs run PR code on persistent runners, so the tests hold no HF token.
This script runs first in the job, from a checkout of the workflow's own
commit (github.sha), with the token in its own step only. It selects the
partition's test files the way test/run_suite.py does and resolves the repo ids
they name from their syntax trees; the PR checkout is read as data, never
imported.

    python3 scripts/ci/utils/prefetch_hf_models.py --repo-root . \\
        --suite base-b-test-1-gpu-small \\
        --auto-partition-id 0 --auto-partition-size 8 --dry-run

A download problem is a warning, never a failure: the test then fails the way
it would have without the prefetch.
"""

import argparse
import ast
import fnmatch
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set

TRUSTED_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# HF caps a repo name at 96 characters and rejects `--` and `..` in it.
_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"
)

_DIFFUSION_SERVER_DIR = os.path.join(
    "python", "sglang", "multimodal_gen", "test", "server"
)

# Weights in formats no test loads: Meta's original checkpoints, GGUF, and
# the ONNX / TF / Flax exports some repos carry next to the PyTorch ones.
_SKIPPED_DIRS = ("original/", "onnx/", "openvino/")
_SKIPPED_SUFFIXES = (".pth", ".gguf", ".onnx", ".h5", ".msgpack", ".tflite")


def _load_base_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(TRUSTED_ROOT, rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci_register = _load_base_module(
    "ci_register", os.path.join("python", "sglang", "test", "ci", "ci_register.py")
)
list_stage_models = _load_base_module(
    "list_stage_models", os.path.join("scripts", "ci", "list_stage_models.py")
)


def warn(message: str) -> None:
    print(f"::warning::prefetch_hf_models: {message}", flush=True)


def select_partition_files(
    repo_root: str,
    suites: List[str],
    partition_id: Optional[int],
    partition_size: Optional[int],
    partition_model_file: Optional[str],
) -> List[str]:
    """Repo-relative test files `run_suite.py --hw cuda` runs with the same
    arguments."""
    paths = sorted(
        path
        for path in glob.glob(
            os.path.join(repo_root, "test", "registered", "**", "*.py"),
            recursive=True,
        )
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    )
    tests = []
    for path in paths:
        try:
            registries, _ = ci_register.ut_parse_one_file(path)
        except (SyntaxError, ValueError, OSError) as exc:
            warn(f"cannot parse {os.path.relpath(path, repo_root)}: {exc}")
            continue
        tests += [
            r
            for r in registries
            if r.backend == ci_register.HWBackend.CUDA
            and r.effective_suite in suites
            and not r.nightly
            and r.disabled is None
        ]
    if partition_size:
        live_est = ci_register.load_live_est(partition_model_file, suites, repo_root)
        tests = ci_register.auto_partition(
            tests, partition_id, partition_size, live_est=live_est
        )
    return [os.path.relpath(t.filename, repo_root) for t in tests]


def diffusion_plan_part(repo_root: str, plan_json: str, part: int) -> Dict:
    """`{"case_ids", "standalone_files"}` of one part of a diffusion partition
    plan (compute_diffusion_partitions.py), the files repo-relative."""
    plan = json.loads(plan_json)
    entry = next((p for p in plan["partitions"] if p["part"] == part), None)
    if entry is None:
        raise ValueError(f"diffusion part {part} is not in the plan")
    server_dir = os.path.join(repo_root, _DIFFUSION_SERVER_DIR)
    standalone_files = []
    for f in entry.get("standalone_files", []):
        rel = os.path.relpath(os.path.normpath(os.path.join(server_dir, f)), repo_root)
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            warn(f"ignoring standalone file outside the repo: {f}")
            continue
        standalone_files.append(rel)
    return {"case_ids": entry.get("case_ids", []), "standalone_files": standalone_files}


def diffusion_case_repo_ids(
    repo_root: str,
    case_ids: List[str],
    deny: Set[str],
    modules: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Set[str]]:
    """`case id -> repo ids` named by the call in gpu_cases.py that builds the
    case: `DiffusionTestCase("<id>", ...)` or a factory taking the id first."""
    path = os.path.join(repo_root, _DIFFUSION_SERVER_DIR, "gpu_cases.py")
    table = module_names(repo_root, path, deny, modules)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = {case_id: set() for case_id in case_ids}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in found
        ):
            found[node.args[0].value] |= value_repo_ids(node, table, deny)
    return found


def resolve_repo_ids(
    repo_root: str,
    files: List[str],
    case_ids: List[str],
    suites: List[str],
    overrides: Dict,
) -> Dict[str, List[str]]:
    """`source -> repo ids`, where a source is a test file, `case:<id>` for a
    diffusion case, or `suite:<name>` for the overrides' per-suite entries."""
    deny = set(overrides["deny"])
    modules: Dict[str, Dict[str, Set[str]]] = {}
    found: Dict[str, Set[str]] = {}
    if case_ids:
        try:
            cases = diffusion_case_repo_ids(repo_root, case_ids, deny, modules)
        except (OSError, SyntaxError) as exc:
            warn(f"cannot scan the diffusion cases: {exc}")
            cases = {}
        for case_id, ids in cases.items():
            found[f"case:{case_id}"] = ids
    for rel in files:
        ids = set(overrides["by_file"].get(rel, []))
        try:
            ids |= file_repo_ids(repo_root, os.path.join(repo_root, rel), deny, modules)
        except (OSError, SyntaxError) as exc:
            warn(f"cannot scan {rel}: {exc}")
        found[rel] = ids - set(overrides["unused_by_file"].get(rel, []))
    for suite in suites:
        found[f"suite:{suite}"] = set(overrides["by_suite"].get(suite, []))
    return {
        source: sorted(i for i in ids if is_hub_repo_id(i, repo_root))
        for source, ids in found.items()
    }


def file_repo_ids(
    repo_root: str, path: str, deny: Set[str], modules: Dict[str, Dict[str, Set[str]]]
) -> Set[str]:
    """Repo ids a test file names: its string literals, and the names it
    defines or imports from modules in the checkout. A name is resolved
    through its import, never by a bare-name match across modules: one
    module's `MODEL` is not another's."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    table = dict(module_names(repo_root, path, deny, modules, tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            table.update(imported_names(repo_root, path, node, deny, modules))
    return list_stage_models.extract_models_from_source(source, table, deny, tree)


def module_names(
    repo_root: str,
    path: str,
    deny: Set[str],
    modules: Dict[str, Dict[str, Set[str]]],
    tree: Optional[ast.Module] = None,
) -> Dict[str, Set[str]]:
    """`name -> repo ids` for the module-level names of ``path`` that name a
    repo: constants, and the classes and functions (fixtures, kits, helpers)
    whose body or bases do. ``tree`` is the parsed ``path``, when the caller
    has it."""
    if path in modules:
        return modules[path]
    table: Dict[str, Set[str]] = {}
    # Registered before the walk, so an import cycle reads the partial table.
    modules[path] = table
    if tree is None:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            table.update(imported_names(repo_root, path, stmt, deny, modules))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            ids = value_repo_ids(stmt.value, table, deny)
            for target in targets:
                if ids and isinstance(target, ast.Name):
                    table[target.id] = table.get(target.id, set()) | ids
    for stmt in tree.body:
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            ids = value_repo_ids(stmt, table, deny)
            if ids:
                table[stmt.name] = ids
    return table


def value_repo_ids(
    node: ast.AST, table: Dict[str, Set[str]], deny: Set[str]
) -> Set[str]:
    ids: Set[str] = set()
    stack = [node]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.JoinedStr):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if list_stage_models.looks_like_model_id(node.value, deny):
                ids.add(node.value)
        elif isinstance(node, ast.Name):
            ids |= table.get(node.id, set())
        stack.extend(ast.iter_child_nodes(node))
    return ids


def imported_names(
    repo_root: str,
    path: str,
    node: ast.ImportFrom,
    deny: Set[str],
    modules: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Set[str]]:
    source = import_source(repo_root, path, node.module, node.level)
    if source is None:
        return {}
    try:
        names = module_names(repo_root, source, deny, modules)
    except (OSError, SyntaxError) as exc:
        warn(f"cannot scan {os.path.relpath(source, repo_root)}: {exc}")
        return {}
    if any(alias.name == "*" for alias in node.names):
        return dict(names)
    return {
        alias.asname or alias.name: names[alias.name]
        for alias in node.names
        if alias.name in names
    }


def import_source(
    repo_root: str, path: str, module: Optional[str], level: int
) -> Optional[str]:
    """The test-support file `from <module> import ...` in ``path`` loads, or
    None. A test runs as a script, so its own directory is searched first,
    then python/, where the sglang package lives. Only test code is followed: the
    models a test needs are chosen there, while product code names models
    for reasons of its own."""
    if level:
        base = os.path.dirname(path)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        roots = [base]
    else:
        # A test runs as a script, so its own directory comes first on sys.path.
        roots = [os.path.dirname(path), os.path.join(repo_root, "python")]
    test_dirs = (
        os.path.join(repo_root, "python", "sglang", "test", ""),
        os.path.join(repo_root, "python", "sglang", "multimodal_gen", "test", ""),
        os.path.join(repo_root, "test", ""),
    )
    rel = (module or "").replace(".", os.sep)
    for root in roots:
        for candidate in (
            os.path.join(root, rel + ".py"),
            os.path.join(root, rel, "__init__.py"),
        ):
            if os.path.isfile(candidate):
                return candidate if candidate.startswith(test_dirs) else None
    return None


def is_hub_repo_id(value: str, repo_root: str) -> bool:
    return (
        _REPO_ID_RE.match(value) is not None
        and "--" not in value
        and ".." not in value
        and not os.path.exists(os.path.join(repo_root, value))
    )


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


def download_lock_path(repo_id: str, hub_cache: str) -> str:
    # The lock sglang's CI loader holds while it downloads a repo into this
    # cache (_get_lock_file_path in ci_weight_validation.py): its cleanup
    # deletes .incomplete files, ours included, unless we hold it too.
    key_hash = hashlib.sha256(repo_id.encode()).hexdigest()[:16]
    return os.path.join(hub_cache, ".sglang_locks", f"download_{key_hash}.lock")


def prefetch(repo_ids: List[str], specs: Dict[str, Dict]) -> None:
    """Download each repo at the revisions and files its `prefetch` entry in
    the overrides names, else every loadable file at main."""
    from huggingface_hub import HfApi

    api = HfApi()
    for repo_id in repo_ids:
        spec = specs.get(repo_id, {})
        if "skip" in spec:
            print(f"skip {repo_id}: {spec['skip']}", flush=True)
            continue
        for revision in spec.get("revisions", ["main"]):
            fetch_repo(api, repo_id, revision, spec.get("patterns"))


def fetch_repo(api, repo_id: str, revision: str, patterns: Optional[List[str]]) -> None:
    from filelock import FileLock
    from huggingface_hub import constants, snapshot_download

    try:
        info = api.model_info(repo_id, revision=revision)
    except Exception as exc:
        warn(
            f"skip {repo_id}@{revision}: no such model on the Hub ({type(exc).__name__})"
        )
        return
    # ModelInfo.private is None when the Hub does not say: only an explicit
    # False is public.
    if info.private is not False:
        warn(f"skip {repo_id}: not confirmed public")
        return
    try:
        files = files_to_download([s.rfilename for s in info.siblings], patterns)
    except (TypeError, AttributeError) as exc:
        warn(f"skip {repo_id}@{revision}: unexpected Hub response ({exc})")
        return
    print(f"{repo_id}@{revision}: {len(files)} file(s)", flush=True)
    lock_path = download_lock_path(repo_id, constants.HF_HUB_CACHE)
    try:
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
    except Exception as exc:
        warn(f"download of {repo_id}@{revision} failed: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="The PR checkout.")
    parser.add_argument("--suite", required=True, help="As run_suite.py --suite.")
    parser.add_argument("--auto-partition-id", type=int)
    parser.add_argument("--auto-partition-size", type=int)
    parser.add_argument("--partition-model-file")
    parser.add_argument(
        "--diffusion-plan-json",
        help="As DIFFUSION_PARTITION_PLAN_JSON: adds that part's cases and files.",
    )
    parser.add_argument("--diffusion-part", type=int)
    parser.add_argument(
        "--overrides",
        default=os.path.join(
            TRUSTED_ROOT, "scripts", "ci", "stage_models_overrides.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the partition's files and repo ids; download nothing.",
    )
    args = parser.parse_args()
    if args.diffusion_plan_json and args.diffusion_part is None:
        parser.error("--diffusion-plan-json requires --diffusion-part")

    repo_root = os.path.abspath(args.repo_root)
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    overrides = list_stage_models.load_overrides(args.overrides)

    files = select_partition_files(
        repo_root,
        suites,
        args.auto_partition_id,
        args.auto_partition_size,
        args.partition_model_file,
    )
    case_ids: List[str] = []
    if args.diffusion_plan_json:
        part = diffusion_plan_part(
            repo_root, args.diffusion_plan_json, args.diffusion_part
        )
        case_ids = part["case_ids"]
        files += part["standalone_files"]
    ids_by_source = resolve_repo_ids(repo_root, files, case_ids, suites, overrides)
    repo_ids = sorted({i for ids in ids_by_source.values() for i in ids})

    print(
        f"{len(files)} test file(s), {len(case_ids)} diffusion case(s), "
        f"{len(repo_ids)} repo id(s)"
    )
    for source, ids in ids_by_source.items():
        if ids or not source.startswith("suite:"):
            print(f"  {source}: {', '.join(ids) or '-'}")
    for repo_id in repo_ids:
        print(f"repo {repo_id}")
    sys.stdout.flush()

    if not args.dry_run:
        prefetch(repo_ids, overrides["prefetch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
