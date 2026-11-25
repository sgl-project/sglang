#!/usr/bin/env python3
"""Find the Hugging Face repo ids CI test files name, by static analysis.

Test sources are read as data, never imported, so the scan runs on a plain
runner with no sglang install. A file's ids are its string literals shaped
like a repo id, plus the ids behind the names it uses: names it defines, and
names it imports from test-support modules of the checkout, followed through
their own imports. A name resolves through its import, never by a bare-name
match across modules: one module's `MODEL` is not another's.
stage_models_overrides.json corrects the scan: `by_file` / `by_suite` add ids
it cannot see, `unused_by_file` drops ids a file names but never loads on
CUDA, `deny` lists strings that are not repo ids, and `datasets` lists Hub
dataset ids, which tests read but which are not models.

As a script, it prints for $GITHUB_OUTPUT the repos each partition of each
CUDA suite in check-changes' `partitions` needs, with the files
test/run_suite.py selects for that partition:

    python3 scripts/ci/hf_repo_scan.py --partitions "$PARTITIONS" \\
        --partition-model-file /tmp/partition-model.json
    prefetch_repos={"base-b-test-1-gpu-small": [["meta-llama/...", ...], ...], ...}
"""

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERRIDES_PATH = os.path.join("scripts", "ci", "stage_models_overrides.json")

# A HuggingFace repo id is `namespace/name`: exactly one slash, each side
# starting alphanumeric and made of alnum plus `.`, `_`, `-`. Note `.` is kept
# because real ids carry it (e.g. `RedHatAI/Llama-3.2-3B-quantized.w8a8`).
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# `namespace/...` values that look like model ids but are MIME types or similar.
_MIME_NAMESPACES = frozenset(
    {
        "application",
        "audio",
        "example",
        "font",
        "image",
        "message",
        "model",
        "multipart",
        "text",
        "video",
    }
)

# Trailing extensions that mark a file path or weight file, not a model name.
# (Real ids use suffixes like `-GGUF`/`.w8a8`, not these dotted extensions.)
_FILE_EXTENSIONS = (
    ".py",
    ".json",
    ".txt",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".sh",
    ".cu",
    ".cuh",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".so",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".csv",
    ".log",
    ".safetensors",
    ".bin",
    ".h5",
    ".gguf",
    ".pt",
    ".pth",
    ".onnx",
)


def looks_like_model_id(value: str, deny: Optional[Set[str]] = None) -> bool:
    """Heuristic: does ``value`` look like a HuggingFace repo id?

    Recall-favoring (a few false positives are cheap for cache-warming) but
    drops the obvious non-models: MIME types, file paths, numeric ratios.
    """
    if deny and value in deny:
        return False
    if not MODEL_ID_RE.match(value):
        return False
    if not any(c.isalpha() for c in value):  # e.g. "2/3"
        return False
    namespace, name = value.split("/", 1)
    if len(namespace) < 2 or len(name) < 2:  # e.g. "N/A"; real ids have longer parts
        return False
    if namespace.lower() in _MIME_NAMESPACES:
        return False
    if name.lower().endswith(_FILE_EXTENSIONS):
        return False
    return True


def load_overrides(path: Optional[str]) -> Dict[str, object]:
    """Read the overrides JSON (every key optional). ``suite_labels`` is read
    by list_stage_models.py only. A present-but-null key is treated as its
    default, so a hand-edit like ``"deny": null`` does not blow up downstream
    iteration."""
    overrides: Dict[str, object] = {
        "by_file": {},
        "by_suite": {},
        "unused_by_file": {},
        "deny": [],
        "datasets": [],
        "suite_labels": {},
    }
    if not path:
        return overrides
    if not os.path.exists(path):
        raise FileNotFoundError(f"overrides file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in overrides:
        if data.get(key) is not None:
            overrides[key] = data[key]
    return overrides


def load_ci_register(repo_root: str):
    """Import ci_register.py by path, sidestepping the heavy `sglang` package."""
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join(repo_root, "python", "sglang", "test", "ci", "ci_register.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepoIdScanner:
    def __init__(self, repo_root: str, overrides: Dict[str, object]):
        self.repo_root = repo_root
        self.by_file: Dict[str, List[str]] = overrides["by_file"]
        self.by_suite: Dict[str, List[str]] = overrides["by_suite"]
        self.unused_by_file: Dict[str, List[str]] = overrides["unused_by_file"]
        self.not_models = set(overrides["deny"]) | set(overrides["datasets"])
        # `{path: {name: repo ids}}` per scanned module.
        self.modules: Dict[str, Dict[str, Set[str]]] = {}
        # `{repo-relative path: error}` per file the scan could not read.
        self.errors: Dict[str, str] = {}
        self._test_dirs = tuple(
            os.path.join(repo_root, *parts, "")
            for parts in (
                ("python", "sglang", "test"),
                ("python", "sglang", "multimodal_gen", "test"),
                ("test",),
            )
        )

    def file_repo_ids(self, rel: str) -> Set[str]:
        """Repo ids the test file ``rel`` (repo-relative) needs."""
        path = os.path.join(self.repo_root, rel)
        ids = set(self.by_file.get(rel, []))
        tree = self._parse(path)
        if tree is not None:
            table = dict(self.module_repo_ids(path, tree))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    table.update(self._imported_repo_ids(path, node))
            ids |= self.node_repo_ids(tree, table)
        ids -= set(self.unused_by_file.get(rel, []))
        # An id-shaped path of the checkout is a local directory, not a repo.
        return {i for i in ids if not os.path.exists(os.path.join(self.repo_root, i))}

    def module_repo_ids(
        self, path: str, tree: Optional[ast.Module] = None
    ) -> Dict[str, Set[str]]:
        """`name -> repo ids` for the module-level names of ``path`` that name
        a repo: constants, and the classes and functions (fixtures, kits,
        helpers) whose body or bases do. ``tree`` is the parsed ``path``,
        when the caller has it."""
        if path in self.modules:
            return self.modules[path]
        table: Dict[str, Set[str]] = {}
        # Registered before the walk, so an import cycle reads the partial table.
        self.modules[path] = table
        if tree is None:
            tree = self._parse(path)
            if tree is None:
                return table
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                table.update(self._imported_repo_ids(path, stmt))
            elif (
                isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None
            ):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                ids = self.node_repo_ids(stmt.value, table)
                for target in targets:
                    if ids and isinstance(target, ast.Name):
                        table[target.id] = table.get(target.id, set()) | ids
        for stmt in tree.body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                ids = self.node_repo_ids(stmt, table)
                if ids:
                    table[stmt.name] = ids
        return table

    def node_repo_ids(self, node: ast.AST, table: Dict[str, Set[str]]) -> Set[str]:
        """Repo ids under ``node``: its literals shaped like one, and the ids
        ``table`` holds for the names it uses. The literal parts of an f-string
        are skipped: they are fragments of a computed value, such as
        ``f"org/model-{ver}"``."""
        fragments = {
            id(part)
            for child in ast.walk(node)
            if isinstance(child, ast.JoinedStr)
            for part in child.values
        }
        ids: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if id(child) not in fragments and looks_like_model_id(
                    child.value, self.not_models
                ):
                    ids.add(child.value)
            elif isinstance(child, ast.Name):
                ids |= table.get(child.id, set())
        return ids

    def _imported_repo_ids(
        self, path: str, node: ast.ImportFrom
    ) -> Dict[str, Set[str]]:
        source = self._import_source(path, node.module, node.level)
        if source is None:
            return {}
        names = self.module_repo_ids(source)
        if any(alias.name == "*" for alias in node.names):
            return dict(names)
        return {
            alias.asname or alias.name: names[alias.name]
            for alias in node.names
            if alias.name in names
        }

    def _import_source(
        self, path: str, module: Optional[str], level: int
    ) -> Optional[str]:
        """The test-support file `from <module> import ...` in ``path`` loads,
        or None. Only test code is followed: the models a test needs are
        chosen there, while product code names models for reasons of its own."""
        if level:
            base = os.path.dirname(path)
            for _ in range(level - 1):
                base = os.path.dirname(base)
            roots = [base]
        else:
            # A test runs as a script, so its own directory comes first on
            # sys.path, then python/, where the sglang package lives.
            roots = [os.path.dirname(path), os.path.join(self.repo_root, "python")]
        rel = (module or "").replace(".", os.sep)
        for root in roots:
            for candidate in (
                os.path.join(root, rel + ".py"),
                os.path.join(root, rel, "__init__.py"),
            ):
                if os.path.isfile(candidate):
                    return candidate if candidate.startswith(self._test_dirs) else None
        return None

    def _parse(self, path: str) -> Optional[ast.Module]:
        try:
            with open(path, encoding="utf-8") as f:
                return ast.parse(f.read())
        except (OSError, SyntaxError) as exc:
            rel = os.path.relpath(path, self.repo_root)
            self.errors[rel] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: cannot scan {rel}; its repo ids are missing: "
                f"{self.errors[rel]}",
                file=sys.stderr,
            )
            return None


def partition_repo_ids(
    repo_root: str, partitions: Dict[str, Dict], partition_model_file: Optional[str]
) -> Dict[str, List[List[str]]]:
    """`suite -> [repo ids of partition i]` for each CUDA suite of
    ``partitions`` (compute_partitions.py's output)."""
    ci_register = load_ci_register(repo_root)
    tests = ci_register.collect_tests(
        ci_register.registered_test_files(repo_root), sanity_check=False
    )
    scanner = RepoIdScanner(
        repo_root, load_overrides(os.path.join(repo_root, OVERRIDES_PATH))
    )
    result: Dict[str, List[List[str]]] = {}
    for suite, info in sorted(partitions.items()):
        suite_tests, _ = ci_register.filter_tests(
            tests, ci_register.HWBackend.CUDA, [suite]
        )
        if not suite_tests:
            continue
        live_est = ci_register.load_live_est(partition_model_file, [suite], repo_root)
        suite_ids = set(scanner.by_suite.get(suite, []))
        result[suite] = []
        for partition_id in range(info["size"]):
            ids = set(suite_ids)
            for test in ci_register.auto_partition(
                suite_tests, partition_id, info["size"], live_est=live_est
            ):
                ids |= scanner.file_repo_ids(os.path.relpath(test.filename, repo_root))
            result[suite].append(sorted(ids))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument(
        "--partitions", required=True, help="compute_partitions.py's JSON output."
    )
    parser.add_argument("--partition-model-file", default=None)
    args = parser.parse_args()
    # A scan problem must not stop the stages: their prefetch then has no
    # list, and a test needing a gated repo fails the way it would without one.
    try:
        result = partition_repo_ids(
            os.path.abspath(args.repo_root),
            json.loads(args.partitions),
            args.partition_model_file,
        )
    except Exception as exc:
        print(f"::warning::hf_repo_scan: {type(exc).__name__}: {exc}", file=sys.stderr)
        result = {}
    print(f"prefetch_repos={json.dumps(result, separators=(',', ':'), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
