"""Reproduce a whole mechanical refactor and diff it byte-for-byte against a commit.

You write a ``transform()`` that recreates the change; ``verify_mechanical_refactor``
checks out the base commit in a throwaway worktree, runs the transform, runs pre-commit,
and diffs the result against the target commit. An empty diff is a PASS. Use this for a
single mechanical PR, and for renames / inlines where a formatter re-wraps lines
(reproduce-and-diff is more robust there than inspecting the diff).

The ``Repro`` builder composes faithful relocation primitives (``move_symbol``,
``lower_call_sites``, ``requalify_call_sites``, ``remove_import``, ``add_import``) into a
transform, so a move that a formatter re-wrapped can be reproduced and certified. Each
primitive does only a relocation-faithful edit (it never changes logic), so a byte match
after the formatter certifies the commit is exactly that relocation. The primitives are
deliberately small and AST-driven; see reproduce-mode.md.

This module is self-contained and independent of the move-verifier.
"""

import ast
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


def exec_command(cmd: str, cwd: str | None = None, check: bool = True) -> str:
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_add_and_commit(message: str, cwd: str) -> None:
    exec_command(f"git add -A && git commit -m {shlex.quote(message)}", cwd=cwd)


def dedent(text: str, n: int) -> str:
    """Remove exactly n leading spaces from each line."""
    lines = text.splitlines(keepends=True)
    return "".join(line[n:] if line[:n] == " " * n else line for line in lines)


def verify_mechanical_refactor(
    base_commit: str,
    target_commit: str,
    transform: "Callable[[Path], None]",
) -> None:
    repo_root = exec_command("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{base_commit[:8]}"

    try:
        print(f"[1/4] Creating worktree at {base_commit[:8]}...")
        exec_command(
            f"git worktree add -b {branch_name} {worktree_dir} {base_commit}",
            cwd=repo_root,
        )

        print("[2/4] Running transformation...")
        transform(Path(worktree_dir))

        print("[3/4] Running pre-commit...")
        exec_command("pre-commit run --all-files", cwd=worktree_dir, check=False)
        if exec_command("git status --porcelain", cwd=worktree_dir):
            git_add_and_commit("pre-commit fixes", cwd=worktree_dir)

        print(f"[4/4] Diffing against {target_commit[:8]}...")
        diff = exec_command(
            f"git diff {target_commit} -- .",
            cwd=worktree_dir,
            check=False,
        )

        if diff:
            print(f"\nFAIL: diff is non-empty:\n{diff}")
            sys.exit(1)
        else:
            print("\nPASS: transform reproduces the commit exactly.")

    finally:
        print(f"\nWorktree left at: {worktree_dir}")
        print(f"Branch: {branch_name}")
        print("To clean up manually:")
        print(f"  git worktree remove {worktree_dir} && git branch -D {branch_name}")


# Decorators a method sheds when it becomes a free function; carried on one side of a move.
_MOVE_DECORATORS = {"@staticmethod", "@classmethod"}


def _find_def(tree: ast.AST, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _def_span(node: ast.AST) -> tuple[int, int]:
    """(first, last) 1-based line numbers of a def, including its decorators."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno


def _replace_span(text: str, sl: int, sc: int, el: int, ec: int, repl: str) -> str:
    """Replace the text from (sl, sc) to (el, ec) -- 1-based lines, 0-based columns, end
    exclusive (ast node span semantics) -- with ``repl``."""
    lines = text.splitlines(keepends=True)
    before = "".join(lines[: sl - 1]) + lines[sl - 1][:sc]
    after = lines[el - 1][ec:] + "".join(lines[el:])
    return before + repl + after


def _slice_span(text: str, sl: int, sc: int, el: int, ec: int) -> str:
    lines = text.splitlines(keepends=True)
    if sl == el:
        return lines[sl - 1][sc:ec]
    return lines[sl - 1][sc:] + "".join(lines[sl : el - 1]) + lines[el - 1][:ec]


def _had_magic_comma(call_text: str) -> bool:
    """Whether a call kept a trailing comma before its closing paren -- the formatter's
    'magic trailing comma' that forces one-argument-per-line. A rewrite must preserve it so
    the formatter re-wraps the call the same way."""
    inner = call_text.rstrip()
    return inner.endswith(")") and inner[:-1].rstrip().endswith(",")


def _rewrite_calls(text: str, predicate: "Callable", build: "Callable") -> str:
    """Replace every call node ``predicate`` accepts with the unparse of ``build(node)``,
    preserving a magic trailing comma; later edits are applied first so spans stay valid.
    """
    repls = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and predicate(node):
            new_text = ast.unparse(build(node))
            orig = _slice_span(
                text, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
            )
            if _had_magic_comma(orig) and new_text.endswith(")"):
                new_text = new_text[:-1] + ",)"
            repls.append(
                (
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                    new_text,
                )
            )
    for sl, sc, el, ec, r in sorted(repls, reverse=True):
        text = _replace_span(text, sl, sc, el, ec, r)
    return text


class Repro:
    """Builds a faithful relocation transform from primitives, then reproduces a commit.

    Operations are recorded and applied in order to a throwaway worktree at ``base``; the
    formatter (pre-commit) runs, and the result is diffed against ``target``. ``run`` prints
    PASS on an empty diff, otherwise the residual -- exactly what the relocation does not
    account for (a bundled change, or a re-derived scaffold a human must confirm)."""

    def __init__(self, base: str, target: str) -> None:
        self.base = base
        self.target = target
        self.ops: list[Callable[[Path], None]] = []

    def lower_call_sites(self, name: str, owner: str, *, paths: list[str]) -> "Repro":
        """Rewrite ``owner.name(receiver, rest)`` to ``receiver.name(rest)`` -- a static
        call becoming an instance-method call when ``name`` moves onto a class."""

        def op(root: Path) -> None:
            for rel in paths:
                path = root / rel

                def predicate(node: ast.Call) -> bool:
                    return (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and bool(node.args)
                        and ast.unparse(node.func.value) == owner
                    )

                def build(node: ast.Call) -> ast.Call:
                    return ast.Call(
                        func=ast.Attribute(
                            value=node.args[0], attr=name, ctx=ast.Load()
                        ),
                        args=node.args[1:],
                        keywords=node.keywords,
                    )

                path.write_text(_rewrite_calls(path.read_text(), predicate, build))

        self.ops.append(op)
        return self

    def requalify_call_sites(
        self, name: str, owner: str, *, paths: list[str]
    ) -> "Repro":
        """Rewrite ``owner.name(args)`` to ``name(args)`` -- dropping the qualifier when
        ``name`` moves to a module-level free function."""

        def op(root: Path) -> None:
            for rel in paths:
                path = root / rel

                def predicate(node: ast.Call) -> bool:
                    return (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and ast.unparse(node.func.value) == owner
                    )

                def build(node: ast.Call) -> ast.Call:
                    return ast.Call(
                        func=ast.Name(id=name, ctx=ast.Load()),
                        args=node.args,
                        keywords=node.keywords,
                    )

                path.write_text(_rewrite_calls(path.read_text(), predicate, build))

        self.ops.append(op)
        return self

    def remove_import(
        self, rel: str, import_text: str, *, in_function: str | None = None
    ) -> "Repro":
        """Remove every import statement whose text contains ``import_text`` (and a trailing
        blank), optionally scoped to one function so a same-text module-level import (e.g. a
        ``TYPE_CHECKING`` guard) is left untouched."""

        def op(root: Path) -> None:
            path = root / rel
            lines = path.read_text().splitlines(keepends=True)
            tree = ast.parse("".join(lines))
            scope: tuple[int, int] | None = None
            if in_function is not None:
                fn = _find_def(tree, in_function)
                assert fn is not None, f"function {in_function} not found in {rel}"
                scope = (fn.lineno, fn.end_lineno)
            spans: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if scope is not None and not (scope[0] <= node.lineno <= scope[1]):
                        continue
                    seg = "".join(lines[node.lineno - 1 : node.end_lineno])
                    if import_text in seg:
                        lo, hi = node.lineno, node.end_lineno
                        if hi < len(lines) and lines[hi].strip() == "":
                            hi += 1
                        spans.append((lo, hi))
            assert spans, f"import {import_text!r} not found in {rel}"
            for lo, hi in sorted(spans, reverse=True):
                del lines[lo - 1 : hi]
            path.write_text("".join(lines))

        self.ops.append(op)
        return self

    def add_import(self, rel: str, import_stmt: str) -> "Repro":
        """Append an import after the last top-level import; the formatter's import sorter
        places it (so the exact insertion point does not matter)."""

        def op(root: Path) -> None:
            path = root / rel
            lines = path.read_text().splitlines(keepends=True)
            last = 0
            for node in ast.parse("".join(lines)).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last = max(last, node.end_lineno)
            path.write_text("".join(lines[:last] + [import_stmt + "\n"] + lines[last:]))

        self.ops.append(op)
        return self

    def move_symbol(
        self,
        name: str,
        *,
        src: str,
        dst: str,
        into_class: str | None,
        dedent: int = 0,
    ) -> "Repro":
        """Cut ``def name`` (with decorators) from ``src`` and paste it into ``dst`` -- at
        the end of ``into_class`` (or module level when None) -- dropping a move decorator
        and dedenting by ``dedent``. The body is moved verbatim; the formatter normalises
        the surrounding blank lines."""

        def op(root: Path) -> None:
            src_path = root / src
            dst_path = root / dst
            src_lines = src_path.read_text().splitlines(keepends=True)
            node = _find_def(ast.parse("".join(src_lines)), name)
            assert node is not None, f"{name} not found in {src}"
            start, end = _def_span(node)
            block = src_lines[start - 1 : end]
            src_path.write_text("".join(src_lines[: start - 1] + src_lines[end:]))

            kept = [ln for ln in block if ln.strip() not in _MOVE_DECORATORS]
            if dedent:
                kept = [
                    ln[dedent:] if ln[:dedent] == " " * dedent else ln for ln in kept
                ]
            method_text = "".join(kept)

            dst_lines = dst_path.read_text().splitlines(keepends=True)
            if into_class is not None:
                cls = _find_class(ast.parse("".join(dst_lines)), into_class)
                assert cls is not None, f"class {into_class} not found in {dst}"
                at = cls.end_lineno
            else:
                at = len(dst_lines)
            dst_path.write_text(
                "".join(dst_lines[:at] + ["\n", method_text] + dst_lines[at:])
            )

        self.ops.append(op)
        return self

    def run(self) -> str:
        """Apply the operations to a worktree at base, run pre-commit, diff against target.
        Returns the residual diff ("" on a clean reproduction)."""
        repo_root = exec_command("git rev-parse --show-toplevel")
        worktree = tempfile.mkdtemp(prefix="repro-")
        branch = Path(worktree).name
        try:
            exec_command(
                f"git worktree add -b {branch} {worktree} {self.base}", cwd=repo_root
            )
            for op in self.ops:
                op(Path(worktree))
            exec_command("pre-commit run --all-files", cwd=worktree, check=False)
            if exec_command("git status --porcelain", cwd=worktree):
                git_add_and_commit("repro", cwd=worktree)
            diff = exec_command(
                f"git diff {self.target} -- .", cwd=worktree, check=False
            )
            if diff:
                print(f"\nRESIDUAL ({len(diff.splitlines())} lines):\n{diff}")
            else:
                print("\nPASS: reproduces the commit byte-for-byte.")
            return diff
        finally:
            exec_command(
                f"git worktree remove --force {worktree}", cwd=repo_root, check=False
            )
            exec_command(f"git branch -D {branch}", cwd=repo_root, check=False)
