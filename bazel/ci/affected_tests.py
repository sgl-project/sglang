"""Select Bazel tests affected by a Git diff.

The migration is intentionally additive: this tool reports Bazel graph
coverage and command-ready target groups, but it does not alter SGLang's
existing CI test registration or scheduling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

GLOBAL_FILES = {
    ".bazelrc",
    ".bazelversion",
    "MODULE.bazel",
    "MODULE.bazel.lock",
    "WORKSPACE",
    "WORKSPACE.bazel",
    ".github/workflows/_pr-test-check-changes.yml",
    ".github/workflows/pr-test-bazel.yml",
    ".github/workflows/pr-test-extra.yml",
}
LOCK_FILE_NAMES = {
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}
BUILD_FILE_NAMES = {"BUILD", "BUILD.bazel"}


@dataclass(frozen=True)
class Change:
    status: str
    path: str


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and process.returncode:
        command = " ".join(args)
        raise RuntimeError(
            f"Command failed ({process.returncode}): {command}\n{process.stderr}"
        )
    return process


def _commit(repo: Path, ref: str) -> str:
    return _run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=repo,
    ).stdout.strip()


def _usable_base(ref: str) -> bool:
    return bool(ref) and any(character != "0" for character in ref)


def resolve_range(repo: Path, base: str, head: str) -> tuple[str, str]:
    resolved_head = _commit(repo, head or "HEAD")
    if _usable_base(base):
        resolved_base = _commit(repo, base)
    else:
        resolved_base = _commit(repo, f"{resolved_head}^")
    return resolved_base, resolved_head


def parse_name_status(raw: bytes) -> list[Change]:
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError("Unexpected git --name-status -z output")

    changes = []
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        path = fields[index + 1].decode("utf-8", errors="surrogateescape")
        changes.append(Change(status=status[0], path=path))
    return changes


def git_changes(repo: Path, base: str, head: str) -> list[Change]:
    process = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base}...{head}",
            "--",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    return parse_name_status(process.stdout)


def _safe_repo_path(path: str) -> PurePosixPath:
    result = PurePosixPath(path)
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"Git path escapes the repository: {path!r}")
    return result


def _package(repo: Path, path: str) -> tuple[str, str] | None:
    relative = _safe_repo_path(path)
    parent_parts = relative.parent.parts
    for size in range(len(parent_parts), -1, -1):
        directory = repo.joinpath(*parent_parts[:size])
        for build_file in ("BUILD.bazel", "BUILD"):
            if (directory / build_file).is_file():
                package = PurePosixPath(*parent_parts[:size]).as_posix()
                if package == ".":
                    package = ""
                return package, build_file
    return None


def _label(package: str, name: str) -> str:
    return f"//{package}:{name}" if package else f"//:{name}"


def source_and_build_labels(repo: Path, path: str) -> tuple[str, str] | None:
    package = _package(repo, path)
    if package is None:
        return None
    package_name, build_file = package
    relative = _safe_repo_path(path)
    package_path = PurePosixPath(package_name) if package_name else PurePosixPath()
    name = relative.relative_to(package_path).as_posix()
    return _label(package_name, name), _label(package_name, build_file)


def global_reason(change: Change) -> str | None:
    path = PurePosixPath(change.path)
    if change.path in GLOBAL_FILES or path.name.startswith(".bazelrc"):
        return "global Bazel or CI configuration"
    if (
        path.name in LOCK_FILE_NAMES
        or path.name.endswith(".lock")
        or ".lock." in path.name
    ):
        return "dependency lock"
    if change.status == "D" and path.name in BUILD_FILE_NAMES:
        return "deleted package definition"
    if change.status == "D" and path.suffix == ".bzl":
        return "deleted Starlark definition"
    return None


def _quote(value: str) -> str:
    return json.dumps(value)


def _set(labels: Sequence[str]) -> str:
    return "set(" + " ".join(_quote(label) for label in labels) + ")"


class BazelQuery:
    def __init__(self, repo: Path, bazel: str = "bazel") -> None:
        self.repo = repo
        self.bazel = bazel

    def labels(
        self,
        expression: str,
        *,
        sky: bool = False,
        missing_label: str | None = None,
    ) -> list[str]:
        args = [
            self.bazel,
            "query",
            "--noshow_progress",
            "--output=label",
        ]
        if sky:
            args.append("--universe_scope=//...")
        args.append(expression)
        process = _run(args, cwd=self.repo, check=False)
        if process.returncode:
            if missing_label and f"no such target '{missing_label}'" in process.stderr:
                return []
            command = " ".join(args)
            raise RuntimeError(
                f"Command failed ({process.returncode}): {command}\n"
                f"{process.stderr}"
            )
        return sorted(line for line in process.stdout.splitlines() if line)


def _file_plan(
    repo: Path,
    change: Change,
    query: BazelQuery,
) -> dict[str, object]:
    reason = global_reason(change)
    if reason:
        return {
            "status": change.status,
            "path": change.path,
            "mapping": "global",
            "global_reason": reason,
            "source_label": None,
            "owners": [],
            "_seed": None,
            "_sky": False,
        }

    labels = source_and_build_labels(repo, change.path)
    if labels is None:
        return {
            "status": change.status,
            "path": change.path,
            "mapping": "uncovered",
            "source_label": None,
            "owners": [],
            "_seed": None,
            "_sky": False,
        }

    source_label, build_label = labels
    path = PurePosixPath(change.path)
    if change.status == "D" or path.name in BUILD_FILE_NAMES:
        seed = f"siblings({_set([build_label])})"
        owner_expression = f'kind(".* rule", {seed})'
        mapping = "package"
        sky = False
        missing_label = None
    elif path.suffix == ".bzl":
        loaded_packages = f"siblings(rbuildfiles({_quote(change.path)}))"
        seed = loaded_packages
        owner_expression = f'kind(".* rule", {loaded_packages})'
        mapping = "starlark-load"
        sky = True
        missing_label = None
    else:
        seed = _set([source_label])
        owner_expression = f'kind(".* rule", rdeps(//..., {seed}, 1))'
        mapping = "source"
        sky = False
        missing_label = source_label

    owners = query.labels(owner_expression, sky=sky, missing_label=missing_label)
    if not owners:
        mapping = "uncovered"
        seed = None
        sky = False
    return {
        "status": change.status,
        "path": change.path,
        "mapping": mapping,
        "source_label": source_label,
        "owners": owners,
        "_seed": seed,
        "_sky": sky,
    }


def _classify_tests(
    query: BazelQuery, selected_tests: Sequence[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    if not selected_tests:
        return [], [], [], []
    tests = _set(selected_tests)
    cuda = query.labels(
        "("
        f'attr(tags, "requires-cuda", {tests}) union '
        f'attr(target_compatible_with, "//bazel/platforms:cuda", {tests})'
        ")"
    )
    rocm = query.labels(
        "("
        f'attr(tags, "requires-rocm", {tests}) union '
        f'attr(target_compatible_with, "//bazel/platforms:rocm", {tests})'
        ")"
    )
    manual = query.labels(f'attr(tags, "manual", {tests})')
    accelerator = set(cuda) | set(rocm)
    cpu = sorted(set(selected_tests) - accelerator)
    return cpu, cuda, rocm, manual


def select(
    repo: Path,
    changes: Sequence[Change],
    query: BazelQuery,
    *,
    base: str,
    head: str,
) -> dict[str, object]:
    file_plans = [_file_plan(repo, change, query) for change in changes]
    global_change = any(plan["mapping"] == "global" for plan in file_plans)

    if global_change:
        selected_tests = query.labels("tests(//...)")
    else:
        seeds = [str(plan["_seed"]) for plan in file_plans if plan["_seed"] is not None]
        if seeds:
            seed = " union ".join(f"({item})" for item in seeds)
            sky = any(bool(plan["_sky"]) for plan in file_plans)
            affected = f"rdeps(//..., ({seed}))"
            selected_tests = query.labels(
                "("
                f'kind(".*_test rule", {affected}) union '
                f'tests(kind("test_suite rule", ({seed})))'
                ")",
                sky=sky,
            )
        else:
            selected_tests = []

    cpu, cuda, rocm, manual = _classify_tests(query, selected_tests)
    uncovered = sorted(
        str(plan["path"]) for plan in file_plans if plan["mapping"] == "uncovered"
    )

    public_plans = []
    for plan in file_plans:
        public_plans.append(
            {key: value for key, value in plan.items() if not key.startswith("_")}
        )

    return {
        "schema_version": 1,
        "base": base,
        "head": head,
        "global_change": global_change,
        "changed_files": public_plans,
        "selected_tests": selected_tests,
        "classification": {
            "cpu": cpu,
            "cuda": cuda,
            "rocm": rocm,
            # Manual is orthogonal to backend. A manual accelerator test is
            # intentionally present in both lists.
            "manual": manual,
        },
        "uncovered_files": uncovered,
    }


def _markdown_labels(labels: Sequence[str]) -> str:
    if not labels:
        return "_None_"
    return "\n".join(f"- `{label}`" for label in labels)


def render_markdown(result: dict[str, object]) -> str:
    classification = result["classification"]
    assert isinstance(classification, dict)
    changed_files = result["changed_files"]
    assert isinstance(changed_files, list)

    lines = [
        "## Bazel affected-test selection",
        "",
        f"- Base: `{result['base']}`",
        f"- Head: `{result['head']}`",
        f"- Global change: `{str(result['global_change']).lower()}`",
        "",
        "| Class | Targets |",
        "| --- | ---: |",
    ]
    for category in ("cpu", "cuda", "rocm", "manual"):
        targets = classification[category]
        assert isinstance(targets, list)
        lines.append(f"| {category.upper()} | {len(targets)} |")

    lines.extend(
        [
            "",
            "Manual is an orthogonal tag: manual accelerator tests also appear "
            "in their backend class.",
            "",
            "| Changed file | Mapping | Direct owners |",
            "| --- | --- | ---: |",
        ]
    )
    for plan in changed_files:
        assert isinstance(plan, dict)
        path = str(plan["path"]).replace("|", "\\|")
        owners = plan["owners"]
        assert isinstance(owners, list)
        lines.append(f"| `{path}` | {plan['mapping']} | {len(owners)} |")

    for category in ("cpu", "cuda", "rocm", "manual"):
        targets = classification[category]
        assert isinstance(targets, list)
        lines.extend(
            [
                "",
                f"<details><summary>{category.upper()} targets</summary>",
                "",
                _markdown_labels(targets),
                "",
                "</details>",
            ]
        )

    uncovered = result["uncovered_files"]
    assert isinstance(uncovered, list)
    lines.extend(["", "### Uncovered files", "", _markdown_labels(uncovered), ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="",
        help="Base commit. Empty/all-zero values fall back to HEAD^.",
    )
    parser.add_argument("--head", default="HEAD", help="Head commit.")
    parser.add_argument(
        "--bazel",
        default=os.environ.get("BAZEL", "bazel"),
        help="Bazel executable.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--fail-on-uncovered",
        action="store_true",
        help="Exit 2 when changed files have no Bazel owner.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(
        _run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    )
    base, head = resolve_range(repo, args.base, args.head)
    changes = git_changes(repo, base, head)
    result = select(
        repo,
        changes,
        BazelQuery(repo, args.bazel),
        base=base,
        head=head,
    )

    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(serialized)
    if args.json_output:
        args.json_output.write_text(serialized, encoding="utf-8")
    if args.summary_output:
        with args.summary_output.open("a", encoding="utf-8") as summary:
            summary.write(render_markdown(result))

    return 2 if args.fail_on_uncovered and result["uncovered_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
