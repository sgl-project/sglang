#!/usr/bin/env python3
"""Build a per-stage model inventory for NVIDIA (CUDA) CI.

Emits a mapping `CI suite -> [HuggingFace model ids]` so the models a stage
exercises can be pre-warmed into a runner cache. The mapping is produced by
*static analysis* of the registered test files (no GPU, no sglang import), so
it can run on a plain runner and stays fresh per commit.

How `suite -> files` is resolved
    Reuses the AST registry parser (`ut_parse_one_file` in ci_register.py, the
    same one `run_suite.py` uses): registered test files call
    `register_<backend>_ci(...)`; we group each file under its
    `effective_suite` for the requested backend (the property falls back to a
    legacy single-string `suite=` when `stage=`/`runner_config=` are unset).

How `file -> models` is resolved (best effort, recall-favoring)
    - A constant table built from `python/sglang/test/**/*.py` module-level
      assignments (`DEFAULT_MODEL_NAME_FOR_TEST = "meta-llama/..."`, including
      tuple/list values) plus each test file's own module-level constants.
    - Inline HuggingFace-id string literals in the file (f-string fragments are
      skipped: they are partial/dynamic and would yield truncated ids).
    - `ast.Name` references that resolve to a known model constant.
    Anything we cannot resolve is reported per suite as `unresolved_files`, and
    any file we cannot parse is reported in `parse_failures`, so recall gaps are
    visible rather than silent. A `--overrides` JSON file supplies models for
    dynamic cases and trims false positives.

How `runner label -> models` is aggregated
    Registration/prewarm decisions are made per GH runner *label* (a runner's
    `runs-on` tag), not per suite. Each suite's runner_config maps to a label
    via scripts/ci/runner_configs.yml (several configs can share one label,
    e.g. `4-gpu-h100` and `deepep-4-gpu-h100`), so `runner_labels` carries the
    per-label UNION -- the set a runner registered under that label must have
    cached before it takes jobs. Suites without a mappable runner_config are
    listed in `unmapped_suites`.

Usage:
    python3 scripts/ci/list_stage_models.py --backend cuda \
        --commit "$GITHUB_SHA" --output models-per-stage.json --markdown out.md
"""

import argparse
import ast
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

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

# Non-test helper files under test/registered/ (skipped by basename, matching
# scripts/ci/check_registered_tests.py). run_suite.py skips `cpu/utils.py` by
# path; excluding every `utils.py` by basename is a superset that drops no
# CUDA-registered test (the other `utils.py` registers CPU only).
_NON_TEST_BASENAMES = frozenset({"conftest.py", "__init__.py", "utils.py"})


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


def _string_values(node: ast.AST) -> List[str]:
    """Static string constants reachable from an assignment RHS.

    Shared test helpers often define model tables as lists of dataclass
    constructors or dicts, e.g. ``[ModelCase(base="org/model")]``. Walk the
    RHS recursively so name references to those tables can still populate the
    inventory. String fragments inside f-strings are skipped because they are
    partial/dynamic and can look like truncated model ids.
    """
    fstring_fragments = {
        id(part)
        for joined in ast.walk(node)
        if isinstance(joined, ast.JoinedStr)
        for part in joined.values
        if isinstance(part, ast.Constant)
    }
    out: List[str] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and id(child) not in fstring_fragments
        ):
            out.append(child.value)
    return out


def extract_constants_from_source(
    source: str, deny: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """Module-level ``NAME = "<model id>"`` (and tuple/list) assignments.

    Handles both bare ``Assign`` and annotated ``AnnAssign`` (``NAME: str =
    ...``). Returns ``{constant_name: {model_id, ...}}``; only model-shaped
    values are kept, so referencing a non-model constant later contributes
    nothing.
    """
    table: Dict[str, Set[str]] = {}
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        models = {v for v in _string_values(value) if looks_like_model_id(v, deny)}
        if not models:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                table.setdefault(target.id, set()).update(models)
    return table


def extract_models_from_source(
    source: str,
    const_table: Dict[str, Set[str]],
    deny: Optional[Set[str]] = None,
) -> Set[str]:
    """All model ids reachable from ``source``: inline literals + name refs.

    ``const_table`` is the merged (global + local) constant lookup. Name
    references resolve against it, so an imported ``DEFAULT_*`` constant is
    picked up even though its value is defined elsewhere. String fragments
    inside f-strings are skipped -- they are partial/dynamic and would yield
    truncated, non-existent ids (e.g. ``f"org/model-{ver}"``).
    """
    tree = ast.parse(source)
    fstring_fragments = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
        if isinstance(part, ast.Constant)
    }
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in fstring_fragments:
                continue
            if looks_like_model_id(node.value, deny):
                found.add(node.value)
        elif isinstance(node, ast.Name) and node.id in const_table:
            found.update(m for m in const_table[node.id] if not (deny and m in deny))
    return found


def build_global_constant_table(
    repo_root: str, deny: Optional[Set[str]] = None
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Constant table from every module under ``python/sglang/test/``.

    These shared helpers (e.g. test_utils, lora_utils) define the ``DEFAULT_*``
    model constants test files reference by name. Returns ``(table, errors)``
    where ``errors`` maps any unparsable helper to its exception string -- a
    broken shared helper drops constants across many suites, so the gap must be
    surfaced rather than swallowed.
    """
    table: Dict[str, Set[str]] = {}
    errors: Dict[str, str] = {}
    pattern = os.path.join(repo_root, "python", "sglang", "test", "**", "*.py")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                source = f.read()
            local = extract_constants_from_source(source, deny)
        except (OSError, SyntaxError) as exc:
            rel = os.path.relpath(path, repo_root)
            errors[rel] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: could not parse constant source {rel}; its model "
                f"constants are EXCLUDED: {errors[rel]}",
                file=sys.stderr,
            )
            continue
        for name, models in local.items():
            table.setdefault(name, set()).update(models)
    return table, errors


def _load_ci_register(repo_root: str):
    """Import ci_register.py by path, sidestepping the heavy `sglang` package."""
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join(repo_root, "python", "sglang", "test", "ci", "ci_register.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_suite_files(
    repo_root: str, backend_name: str, include_disabled: bool = False
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, bool],
    Dict[str, str],
    Dict[str, Optional[str]],
]:
    """Map ``effective_suite -> [relative test file]`` for one backend.

    By default only enabled (`disabled is None`) registries are grouped, since a
    disabled suite does not run and thus needs no cache warming. Pass
    ``include_disabled=True`` to also group disabled registries (useful to see
    what a suite *would* download once re-enabled). Returns the mapping,
    ``{suite: is_nightly}``, ``{file: parse error}`` for files whose registry
    could not be parsed (their models are excluded -- surfaced, not silently
    dropped), and ``{suite: runner_config}`` (None for legacy single-string
    ``suite=`` registrations, which carry no runner_config).
    """
    ci_register = _load_ci_register(repo_root)
    backend = getattr(ci_register.HWBackend, backend_name.upper())

    pattern = os.path.join(repo_root, "test", "registered", "**", "*.py")
    files = sorted(
        f
        for f in glob.glob(pattern, recursive=True)
        if os.path.basename(f) not in _NON_TEST_BASENAMES
    )

    suite_files: Dict[str, List[str]] = {}
    suite_nightly: Dict[str, bool] = {}
    suite_runner_config: Dict[str, Optional[str]] = {}
    errors: Dict[str, str] = {}
    for path in files:
        rel = os.path.relpath(path, repo_root)
        # Narrow catch: SyntaxError (bad source), ValueError (malformed
        # registration, raised by RegistryVisitor), OSError (vanished file). A
        # broader failure (e.g. AttributeError from a parser API drift) should
        # crash loudly rather than silently empty the inventory.
        try:
            registries, _ = ci_register.ut_parse_one_file(path)
        except (SyntaxError, ValueError, OSError) as exc:
            errors[rel] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: could not parse {rel}; its models are EXCLUDED from "
                f"the inventory: {errors[rel]}",
                file=sys.stderr,
            )
            continue
        for r in registries:
            if r.backend != backend:
                continue
            if r.disabled is not None and not include_disabled:
                continue
            suite = r.effective_suite
            if suite is None:
                continue
            if rel not in suite_files.setdefault(suite, []):
                suite_files[suite].append(rel)
            suite_nightly[suite] = suite_nightly.get(suite, False) or bool(r.nightly)
            # Modern registrations name the suite `{stage}-test-{runner_config}`,
            # so every registry in a suite shares one runner_config; legacy
            # `suite=` registrations have none (stays None).
            if r.runner_config is not None:
                suite_runner_config[suite] = r.runner_config
            else:
                suite_runner_config.setdefault(suite, None)
    return suite_files, suite_nightly, errors, suite_runner_config


def load_overrides(path: Optional[str]) -> Dict[str, object]:
    """Read the overrides JSON: ``by_file``, ``by_suite``, ``deny``,
    ``suite_labels`` (all optional).

    ``suite_labels`` maps a legacy ``suite=`` registration (which carries no
    runner_config) to the GH runner label(s) its dispatching workflow
    hardcodes in ``runs-on`` -- a LIST, since one suite can run on several
    labels. A present-but-null key is treated as its default, so a hand-edit
    like ``"deny": null`` does not blow up downstream iteration.
    """
    overrides: Dict[str, object] = {
        "by_file": {},
        "by_suite": {},
        "deny": [],
        "suite_labels": {},
    }
    if not path:
        return overrides
    if not os.path.exists(path):
        raise FileNotFoundError(f"overrides file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("by_file", "by_suite", "deny", "suite_labels"):
        if data.get(key) is not None:
            overrides[key] = data[key]
    return overrides


# One runner_config entry in runner_configs.yml: a two-space-indented key with
# a flow-style (inline `{...}`) mapping. This is the file's documented shape;
# entries are matched line-by-line so the tool stays stdlib-only (the workflow
# installs nothing, so PyYAML is not available).
_RUNNER_CONFIG_LINE_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*\{(.*)\}\s*$")
_RUNS_ON_RE = re.compile(r"\bruns_on:\s*([^,}\s]+)")

# runner_configs.yml uses this placeholder for the dynamically-selected b200
# runner label (resolved at workflow-load time by runner_configs.py --map).
# The inventory keeps it literal unless --b200-runner substitutes it, so the
# consumer can see the group is dynamic rather than silently guessing a label.
B200_SENTINEL = "$b200_runner"


def load_runner_labels(path: str) -> Dict[str, str]:
    """Parse ``{runner_config: runs_on label}`` out of runner_configs.yml.

    The mapping is what turns per-suite model sets into per-runner-LABEL sets:
    a runner is registered under a `runs_on` label (several runner_configs can
    share one, e.g. `4-gpu-h100` and `deepep-4-gpu-h100` both run on
    `4-gpu-h100`), so a runner's cache must cover the union of every suite
    that can land on its label. Raises ValueError on an entry without
    `runs_on` or a file with no entries at all -- a format drift must fail
    the workflow loudly, not silently empty the label aggregation.
    """
    labels: Dict[str, str] = {}
    in_section = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("runner_configs:"):
                in_section = True
                continue
            if in_section and line.strip() and not line.startswith(" "):
                break  # next top-level key
            if not in_section:
                continue
            m = _RUNNER_CONFIG_LINE_RE.match(line)
            if not m:
                continue
            name, body = m.group(1), m.group(2)
            runs_on = _RUNS_ON_RE.search(body)
            if not runs_on:
                raise ValueError(f"{path}: runner_config {name!r} has no runs_on field")
            labels[name] = runs_on.group(1)
    if not labels:
        raise ValueError(
            f"{path}: no runner_configs entries parsed -- format drift? "
            f"(expected two-space-indented `name: {{...}}` lines under "
            f"a `runner_configs:` key)"
        )
    return labels


def build_inventory(
    repo_root: str,
    backend_name: str,
    overrides: Dict[str, object],
    commit: str,
    include_disabled: bool = False,
    b200_runner: Optional[str] = None,
) -> Dict[str, object]:
    deny: Set[str] = set(overrides.get("deny", []))  # type: ignore[arg-type]
    by_file: Dict[str, List[str]] = overrides.get("by_file", {})  # type: ignore[assignment]
    by_suite: Dict[str, List[str]] = overrides.get("by_suite", {})  # type: ignore[assignment]
    suite_labels_override: Dict[str, List[str]] = overrides.get("suite_labels", {})  # type: ignore[assignment]

    global_table, table_errors = build_global_constant_table(repo_root, deny)
    suite_files, suite_nightly, registry_errors, suite_runner_config = (
        collect_suite_files(repo_root, backend_name, include_disabled)
    )

    runner_labels_map: Dict[str, str] = {}
    runner_configs_path = os.path.join(repo_root, "scripts", "ci", "runner_configs.yml")
    if os.path.exists(runner_configs_path):
        runner_labels_map = load_runner_labels(runner_configs_path)
    else:
        print(
            f"WARNING: {runner_configs_path} not found; every suite will be "
            f"reported as unmapped_suites (no per-runner-label aggregation).",
            file=sys.stderr,
        )

    # Resolve models once per file (a file can belong to several suites).
    file_models: Dict[str, Set[str]] = {}
    extract_errors: Dict[str, str] = {}
    for files in suite_files.values():
        for rel in files:
            if rel in file_models:
                continue
            abs_path = os.path.join(repo_root, rel)
            try:
                with open(abs_path, encoding="utf-8") as f:
                    source = f.read()
                local_table = extract_constants_from_source(source, deny)
                merged = dict(global_table)
                for name, models in local_table.items():
                    merged.setdefault(name, set()).update(models)
                resolved = extract_models_from_source(source, merged, deny)
            except (OSError, SyntaxError) as exc:
                extract_errors[rel] = f"{type(exc).__name__}: {exc}"
                print(
                    f"WARNING: could not extract models from {rel}; treating it "
                    f"as unresolved: {extract_errors[rel]}",
                    file=sys.stderr,
                )
                resolved = set()
            resolved.update(by_file.get(rel, []))
            file_models[rel] = resolved

    suites: Dict[str, object] = {}
    all_models: Set[str] = set()
    for suite in sorted(suite_files):
        models: Set[str] = set(by_suite.get(suite, []))
        unresolved: List[str] = []
        for rel in suite_files[suite]:
            resolved = file_models.get(rel, set())
            if resolved:
                models.update(resolved)
            else:
                unresolved.append(rel)
        all_models.update(models)
        suites[suite] = {
            "nightly": suite_nightly.get(suite, False),
            "runner_config": suite_runner_config.get(suite),
            "models": sorted(models),
            "test_file_count": len(suite_files[suite]),
            "unresolved_files": sorted(unresolved),
        }

    # Per-runner-LABEL aggregation: registration/prewarm decisions are made per
    # GH runner label (what a runner is registered with), and several suites --
    # via several runner_configs -- can route to one label. A runner's cache
    # must cover the UNION of every suite that can land on it. Label
    # resolution order: an explicit `suite_labels` override (legacy suites
    # whose runs-on lives hardcoded in their dispatching workflow; may name
    # several labels), else runner_config -> runner_configs.yml. Suites we
    # cannot map land in `unmapped_suites` -- visible, never silently
    # dropped, same contract as unresolved_files.
    label_models: Dict[str, Set[str]] = {}
    label_suites: Dict[str, List[str]] = {}
    unmapped_suites: List[str] = []
    for suite in sorted(suites):
        labels = suite_labels_override.get(suite)
        if labels is None:
            rc = suite_runner_config.get(suite)
            label = runner_labels_map.get(rc) if rc is not None else None
            labels = [label] if label is not None else []
        if not labels:
            unmapped_suites.append(suite)
            continue
        for label in labels:
            if label == B200_SENTINEL and b200_runner:
                label = b200_runner
            label_models.setdefault(label, set()).update(suites[suite]["models"])
            label_suites.setdefault(label, []).append(suite)
    runner_labels: Dict[str, object] = {
        label: {
            "models": sorted(label_models[label]),
            "suites": label_suites[label],
        }
        for label in sorted(label_models)
    }

    parse_failures = {}
    parse_failures.update(table_errors)
    parse_failures.update(registry_errors)
    parse_failures.update(extract_errors)

    return {
        "generated_at_commit": commit,
        "backend": backend_name.lower(),
        "suite_count": len(suites),
        "model_count": len(all_models),
        "runner_label_count": len(runner_labels),
        "all_models": sorted(all_models),
        "parse_failures": dict(sorted(parse_failures.items())),
        "runner_labels": runner_labels,
        "unmapped_suites": unmapped_suites,
        "suites": suites,
    }


def render_markdown(inventory: Dict[str, object]) -> str:
    suites: Dict[str, dict] = inventory["suites"]  # type: ignore[assignment]
    failures = inventory.get("parse_failures") or {}
    runner_labels: Dict[str, dict] = inventory.get("runner_labels") or {}  # type: ignore[assignment]
    unmapped = inventory.get("unmapped_suites") or []
    lines = [
        f"## NVIDIA CI model inventory (`{inventory['backend']}`)",
        "",
        f"- Commit: `{inventory['generated_at_commit']}`",
        f"- Suites: **{inventory['suite_count']}**, "
        f"distinct models: **{inventory['model_count']}**, "
        f"runner labels: **{inventory.get('runner_label_count', 0)}**",
    ]
    if failures:
        lines.append(
            f"- ⚠️ Unparsable files: **{len(failures)}** (see `parse_failures`)"
        )
    if unmapped:
        lines.append(
            f"- ⚠️ Suites with no runner label: **{len(unmapped)}** "
            f"({', '.join(f'`{s}`' for s in unmapped)})"
        )
    if runner_labels:
        lines += [
            "",
            "### Per runner label (prewarm a runner's cache with this union)",
            "",
            "| Runner label | Suites | Models |",
            "| --- | ---: | --- |",
        ]
        for label in sorted(runner_labels):
            info = runner_labels[label]
            models = ", ".join(info["models"]) if info["models"] else "_(none)_"
            lines.append(f"| `{label}` | {len(info['suites'])} | {models} |")
    lines += [
        "",
        "### Per suite",
        "",
        "| Suite | Nightly | Models | Unresolved files |",
        "| --- | :---: | --- | ---: |",
    ]
    for suite in sorted(suites):
        info = suites[suite]
        models = ", ".join(info["models"]) if info["models"] else "_(none)_"
        nightly = "✓" if info["nightly"] else ""
        lines.append(
            f"| `{suite}` | {nightly} | {models} | {len(info['unresolved_files'])} |"
        )
    return "\n".join(lines) + "\n"


def resolve_commit(arg: Optional[str], repo_root: str) -> str:
    if arg:
        return arg
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        help="Hardware backend name (default: cuda).",
    )
    parser.add_argument(
        "--overrides",
        default=os.path.join("scripts", "ci", "stage_models_overrides.json"),
        help="Path to the overrides JSON (relative paths resolve under "
        "--repo-root). A warning is printed if it does not exist.",
    )
    parser.add_argument("--commit", default=None, help="Commit sha to record.")
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also include suites whose tests are currently disabled.",
    )
    parser.add_argument(
        "--b200-runner",
        default=None,
        help="Concrete runner label to substitute for the $b200_runner "
        "placeholder in runner_configs.yml (default: keep the placeholder "
        "as the runner_labels key, marking the group as dynamically routed).",
    )
    parser.add_argument(
        "--output", default=None, help="Write JSON here (default: stdout)."
    )
    parser.add_argument(
        "--markdown", default=None, help="Also write a Markdown summary here."
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    # Resolve a relative overrides path against repo_root (not cwd) so the flag
    # works when invoked from elsewhere with --repo-root.
    overrides_path = args.overrides
    if not os.path.isabs(overrides_path):
        overrides_path = os.path.join(repo_root, overrides_path)
    if not os.path.exists(overrides_path):
        print(
            f"WARNING: overrides file not found at {overrides_path}; "
            f"proceeding without overrides.",
            file=sys.stderr,
        )
        overrides_path = None
    overrides = load_overrides(overrides_path)
    commit = resolve_commit(args.commit, repo_root)

    inventory = build_inventory(
        repo_root,
        args.backend,
        overrides,
        commit,
        args.include_disabled,
        b200_runner=args.b200_runner,
    )
    payload = json.dumps(inventory, indent=2, sort_keys=False) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        print(
            f"Wrote {args.output}: {inventory['suite_count']} suites, "
            f"{inventory['runner_label_count']} runner labels, "
            f"{inventory['model_count']} distinct models, "
            f"{len(inventory['parse_failures'])} parse failures, "
            f"{len(inventory['unmapped_suites'])} unmapped suites.",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(render_markdown(inventory))

    return 0


if __name__ == "__main__":
    sys.exit(main())
