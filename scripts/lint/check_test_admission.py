#!/usr/bin/env python3
"""Incremental admission checks for registered tests.

The repository has a large legacy test inventory, so this checker deliberately
examines only files changed from the PR base (or staged files in a local
pre-commit run). This makes the policy a ratchet instead of a flag day.
"""

import ast
import datetime as dt
import os
import pathlib
import re
import subprocess
import sys

_REGISTER_PREFIX = "register_"
_CUDA_REGISTER = "register_cuda_ci"
_ACCELERATOR_REGISTERS = {
    "register_amd_ci",
    "register_mlx_ci",
    "register_musa_ci",
    "register_npu_ci",
    "register_xpu_ci",
}
_ISSUE = re.compile(r"(?:#\d+|https?://\S+)")
_UNTIL = re.compile(r"\buntil[ :=]+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_ACCELERATOR_COUNT = re.compile(r"(?:^|-)(\d+)-(?:gpu|npu)(?:-|$)")
_DEFAULT_PR_ACCELERATOR_SECONDS = 1200


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def _keyword_literal(node: ast.Call, name: str):
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _positional_or_keyword_literal(node: ast.Call, position: int, name: str):
    if len(node.args) > position and isinstance(node.args[position], ast.Constant):
        return node.args[position].value
    return _keyword_literal(node, name)


def _has_marker(lines: list[str], lineno: int, marker: str) -> bool:
    start = max(0, lineno - 4)
    return any(marker in line.lower() for line in lines[start : lineno - 1])


def _check_temporary_reason(
    reason: object, *, path: pathlib.Path, lineno: int, today: dt.date
) -> list[str]:
    location = f"{path}:{lineno}"
    if not isinstance(reason, str):
        return [f"{location}: temporary disable/skip reason must be a string literal"]
    errors = []
    if _ISSUE.search(reason) is None:
        errors.append(f"{location}: temporary disable/skip must reference an issue")
    match = _UNTIL.search(reason)
    if match is None:
        errors.append(
            f'{location}: temporary disable/skip must include "until YYYY-MM-DD"'
        )
    else:
        try:
            deadline = dt.date.fromisoformat(match.group(1))
        except ValueError:
            errors.append(f"{location}: invalid temporary-disable deadline")
        else:
            if deadline < today:
                errors.append(
                    f"{location}: temporary disable/skip expired on {deadline}"
                )
    return errors


def check_file(path: pathlib.Path, *, today: dt.date | None = None) -> list[str]:
    today = today or dt.date.today()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    lines = source.splitlines()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    registrations = [
        node
        for node in calls
        if (_call_name(node) or "").startswith(_REGISTER_PREFIX)
        and (_call_name(node) or "").endswith("_ci")
    ]
    register_names = {_call_name(node) for node in registrations}
    errors = []

    for node in registrations:
        disabled = _positional_or_keyword_literal(node, 3, "disabled")
        if disabled is not None:
            errors.extend(
                _check_temporary_reason(
                    disabled, path=path, lineno=node.lineno, today=today
                )
            )

        name = _call_name(node)
        if (
            _CUDA_REGISTER in register_names
            and name in _ACCELERATOR_REGISTERS
            and not _has_marker(lines, node.lineno, "backend-specific:")
        ):
            errors.append(
                f"{path}:{node.lineno}: mixed accelerator coverage needs a nearby "
                '"backend-specific:" comment explaining what this lane can catch'
            )

        stage = _keyword_literal(node, "stage")
        suite = _positional_or_keyword_literal(node, 1, "suite")
        est_time = _positional_or_keyword_literal(node, 0, "est_time")
        runner = _keyword_literal(node, "runner_config")
        dispatch = stage if isinstance(stage, str) else suite
        runner_hint = runner if isinstance(runner, str) else suite
        accelerator_match = (
            _ACCELERATOR_COUNT.search(runner_hint)
            if isinstance(runner_hint, str)
            else None
        )
        if (
            isinstance(dispatch, str)
            and dispatch.startswith(("base-", "stage-"))
            and isinstance(est_time, (int, float))
            and accelerator_match is not None
            and disabled is None
        ):
            accelerator_seconds = est_time * int(accelerator_match.group(1))
            if (
                accelerator_seconds > _DEFAULT_PR_ACCELERATOR_SECONDS
                and not _has_marker(lines, node.lineno, "ci-cost-override:")
            ):
                errors.append(
                    f"{path}:{node.lineno}: default-PR registration costs "
                    f"{accelerator_seconds:g} weighted accelerator-seconds (limit "
                    f"{_DEFAULT_PR_ACCELERATOR_SECONDS}); move it to extra/nightly or add "
                    'a nearby "ci-cost-override:" rationale'
                )

    for node in calls:
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "unittest"
            and node.func.attr == "skip"
        ):
            continue
        reason = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else None
        )
        errors.extend(
            _check_temporary_reason(reason, path=path, lineno=node.lineno, today=today)
        )

    return errors


def _git_lines(*args: str) -> list[str] | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def changed_registered_files() -> list[pathlib.Path]:
    staged = _git_lines("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    names = staged or []
    if not names:
        base_ref = os.environ.get("GITHUB_BASE_REF", "main")
        candidates = [f"origin/{base_ref}", base_ref]
        for candidate in candidates:
            if _git_lines("rev-parse", "--verify", candidate) is None:
                continue
            merge_base = _git_lines("merge-base", candidate, "HEAD")
            if not merge_base:
                continue
            names = (
                _git_lines(
                    "diff", "--name-only", "--diff-filter=ACMR", merge_base[0], "HEAD"
                )
                or []
            )
            break

    return sorted(
        pathlib.Path(name)
        for name in names
        if name.startswith("test/registered/")
        and name.endswith(".py")
        and pathlib.Path(name).is_file()
    )


def main(paths: list[str] | None = None) -> int:
    selected = (
        [pathlib.Path(path) for path in paths] if paths else changed_registered_files()
    )
    errors = [error for path in selected for error in check_file(path)]
    if not errors:
        return 0
    print("ERROR: registered-test admission policy violations:")
    for error in errors:
        print(f"  {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
