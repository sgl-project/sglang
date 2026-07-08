"""Reproduce a whole mechanical refactor and diff it byte-for-byte against a commit.

You write a ``transform()`` that recreates the change; ``verify_mechanical_refactor``
checks out the base commit in a throwaway worktree, runs the transform, runs pre-commit on
the changed files, and diffs the result against the target commit. An empty diff is a PASS.
Use this for a single mechanical PR -- a relocation, a whole-file split, or a rename where a
formatter re-wraps lines, which reproduce-and-byte-diff certifies exactly.

The ``Repro`` builder composes faithful relocation primitives (``move_symbol``,
``extract_to_new_module``, ``extract_symbols_to_new_module``, ``extract_function``,
``lower_call_sites``, ``requalify_call_sites``, ``remove_import``, ``remove_imported_name``,
``add_import``, ``repath_import``, ``add_typechecking_import``) into a transform, so a move
that a formatter re-wrapped can be reproduced and certified. Each primitive does only a relocation-faithful
edit (it never changes logic), so a byte match after the formatter certifies the commit is
exactly that relocation. The primitives are deliberately small and AST-driven; see
how-to-guide.md.

This module is self-contained and needs only git and the standard library.
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
        raise RuntimeError(f"command failed: {cmd}\n{result.stderr.strip()}")
    return result.stdout.strip()


def git_add_and_commit(message: str, cwd: str) -> None:
    exec_command(f"git add -A && git commit -m {shlex.quote(message)}", cwd=cwd)


def dedent(text: str, n: int) -> str:
    """Remove exactly n leading spaces from each line."""
    lines = _split_keepends(text)
    return "".join(line[n:] if line[:n] == " " * n else line for line in lines)


def _split_keepends(text: str) -> list[str]:
    """Split into lines ending in "\\n" only -- unlike ``str.splitlines``, a form feed or
    other exotic line break stays inside its line, matching ast's line numbering."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _read_source(path: Path) -> str:
    """Read preserving the file's line endings (no universal-newline translation), so a
    CRLF file round-trips byte-for-byte through the primitives."""
    with path.open("r", newline="") as f:
        return f.read()


def _write_source(path: Path, text: str) -> None:
    with path.open("w", newline="") as f:
        f.write(text)


def _newline_style(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


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
        exec_command("git add -A", cwd=worktree_dir)
        changed = exec_command(
            f"git diff --cached --name-only --diff-filter=ACMR {base_commit}",
            cwd=worktree_dir,
        ).split()
        if changed:
            files = " ".join(shlex.quote(path) for path in changed)
            exec_command(
                f"pre-commit run --files {files}", cwd=worktree_dir, check=False
            )
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


def _byte_slice(line: str, start: int | None, end: int | None) -> str:
    """Slice a line by UTF-8 byte offsets -- ast col_offsets count bytes, not characters."""
    return line.encode("utf-8")[start:end].decode("utf-8")


def _replace_span(text: str, sl: int, sc: int, el: int, ec: int, repl: str) -> str:
    """Replace the text from (sl, sc) to (el, ec) -- 1-based lines, 0-based byte columns,
    end exclusive (ast node span semantics) -- with ``repl``."""
    lines = _split_keepends(text)
    before = "".join(lines[: sl - 1]) + _byte_slice(lines[sl - 1], None, sc)
    after = _byte_slice(lines[el - 1], ec, None) + "".join(lines[el:])
    return before + repl + after


def _slice_span(text: str, sl: int, sc: int, el: int, ec: int) -> str:
    lines = _split_keepends(text)
    if sl == el:
        return _byte_slice(lines[sl - 1], sc, ec)
    return (
        _byte_slice(lines[sl - 1], sc, None)
        + "".join(lines[sl : el - 1])
        + _byte_slice(lines[el - 1], None, ec)
    )


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


def _drop_self_annotation(method_text: str, name: str) -> str:
    """Drop the type annotation from a moved method's ``self`` parameter -- relocating
    ``def foo(self: Target)`` into ``Target`` makes the annotation redundant. The body is
    otherwise untouched, so the relocation stays byte-faithful. ``method_text`` may still be
    class-indented, so it is dedented to parse and the columns are mapped back."""
    first_line = method_text.split("\n", 1)[0]
    base_indent = len(first_line) - len(first_line.lstrip(" "))
    fn = _find_def(ast.parse(dedent(method_text, base_indent)), name)
    if fn is None or not fn.args.args:
        return method_text
    first = fn.args.args[0]
    if first.arg != "self" or first.annotation is None:
        return method_text
    annotation = first.annotation
    return _replace_span(
        method_text,
        first.lineno,
        first.col_offset + base_indent + len("self"),
        annotation.end_lineno,
        annotation.end_col_offset + base_indent,
        "",
    )


class Repro:
    """Builds a faithful relocation transform from primitives, then reproduces a commit.

    Operations are recorded and applied in order to a throwaway worktree at ``base``; the
    formatter (pre-commit) runs, and the result is diffed against ``target``. ``run`` prints
    PASS on an empty diff, otherwise the residual -- exactly what the relocation does not
    account for (a bundled change, or a re-derived scaffold a human must confirm)."""

    def __init__(self, base: str, target: str, repo_root: str | None = None) -> None:
        self.base = base
        self.target = target
        self.repo_root = repo_root
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

                _write_source(
                    path, _rewrite_calls(_read_source(path), predicate, build)
                )

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

                _write_source(
                    path, _rewrite_calls(_read_source(path), predicate, build)
                )

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
            lines = _split_keepends(_read_source(path))
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
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def remove_imported_name(
        self, rel: str, *, module: str | None, name: str, asname: str | None = None
    ) -> "Repro":
        """Drop a single imported ``name`` from a module-level import: from a ``from module
        import a, b`` keep the rest and drop only ``name``; when it was the sole name -- or for
        a plain ``import name`` (``module=None``) -- remove the whole statement. The symbol's
        home changed, so an importer that no longer references it loses exactly that name; the
        import sorter rewrites the surviving line. An import diff is always whitelisted, so this
        realises a lost name directly instead of relying on the formatter to prune it.
        """

        def alias_text(alias: ast.alias) -> str:
            return alias.name + (f" as {alias.asname}" if alias.asname else "")

        def op(root: Path) -> None:
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            edits: list[tuple[int, int, str | None]] = []
            for node in ast.parse("".join(lines)).body:
                if module is None:
                    if not isinstance(node, ast.Import):
                        continue
                else:
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    if "." * node.level + (node.module or "") != module:
                        continue
                kept = [
                    a for a in node.names if not (a.name == name and a.asname == asname)
                ]
                if len(kept) == len(node.names):
                    continue
                if not kept:
                    edits.append((node.lineno, node.end_lineno, None))
                else:
                    keyword = "import " if module is None else f"from {module} import "
                    rebuilt = keyword + ", ".join(alias_text(a) for a in kept) + nl
                    edits.append((node.lineno, node.end_lineno, rebuilt))
            assert edits, f"import of {name!r} from {module!r} not found in {rel}"
            for lo, hi, repl in sorted(edits, reverse=True):
                if repl is None:
                    del lines[lo - 1 : hi]
                else:
                    lines[lo - 1 : hi] = [repl]
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def add_import(self, rel: str, import_stmt: str) -> "Repro":
        """Append an import after the last top-level import; the formatter's import sorter
        places it (so the exact insertion point does not matter)."""

        def op(root: Path) -> None:
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            last = 0
            for node in ast.parse("".join(lines)).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last = max(last, node.end_lineno)
            _write_source(
                path, "".join(lines[:last] + [import_stmt + nl] + lines[last:])
            )

        self.ops.append(op)
        return self

    def add_typechecking_import(self, rel: str, import_stmt: str) -> "Repro":
        """Append ``import_stmt`` inside the file's ``if TYPE_CHECKING:`` block -- a moved
        definition whose annotations reference a type needs that type imported there. The
        import sorter orders the block, so the exact insertion point does not matter."""

        def op(root: Path) -> None:
            path = root / rel
            lines = _split_keepends(_read_source(path))
            for node in ast.parse("".join(lines)).body:
                if isinstance(node, ast.If) and ast.unparse(node.test) in (
                    "TYPE_CHECKING",
                    "typing.TYPE_CHECKING",
                ):
                    indent = " " * node.body[0].col_offset
                    at = node.body[-1].end_lineno
                    lines.insert(
                        at, indent + import_stmt + _newline_style("".join(lines))
                    )
                    _write_source(path, "".join(lines))
                    return
            raise AssertionError(f"no `if TYPE_CHECKING:` block in {rel}")

        self.ops.append(op)
        return self

    def repath_import(
        self, rel: str, *, old_module: str, new_module: str, name: str
    ) -> "Repro":
        """Repath every function-scoped ``from old_module import ... name ...`` to
        ``from new_module import ...`` in place -- the moved symbol's home changed, so its
        importer adjusts. Only imports nested below module level are touched; a module-level
        import is left to the import sorter via add_import / remove_import."""

        def op(root: Path) -> None:
            path = root / rel
            lines = _split_keepends(_read_source(path))
            tree = ast.parse("".join(lines))
            top_level = {id(node) for node in tree.body}
            changed = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and id(node) not in top_level
                    and node.module == old_module
                    and any(alias.name == name for alias in node.names)
                ):
                    lines[node.lineno - 1] = lines[node.lineno - 1].replace(
                        f"from {old_module} import", f"from {new_module} import", 1
                    )
                    changed = True
            assert changed, f"nested import of {name} from {old_module} not in {rel}"
            _write_source(path, "".join(lines))

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
        drop_self_annotation: bool = False,
        before: str | None = None,
        leave_delegate: str | None = None,
        delegate_name: str | None = None,
    ) -> "Repro":
        """Cut ``def name`` (with decorators) from ``src`` and paste it into ``dst`` --
        immediately above the sibling def ``before`` when given (so the relocated def lands in
        the chain's order), else at the end of ``into_class`` (or module level when None) --
        dropping a move decorator and dedenting by ``dedent``. When ``drop_self_annotation``,
        the moved method's ``self: Target`` annotation is dropped (redundant inside the class).
        The body is moved verbatim; the formatter normalises the surrounding blank lines.
        """

        def op(root: Path) -> None:
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            src_nl = _newline_style("".join(src_lines))
            node = _find_def(ast.parse("".join(src_lines)), name)
            assert node is not None, f"{name} not found in {src}"
            start, end = _def_span(node)
            block = src_lines[start - 1 : end]
            if leave_delegate is not None:
                args = node.args
                parts = [p.arg for p in args.posonlyargs + args.args if p.arg != "self"]
                if args.vararg is not None:
                    parts.append(f"*{args.vararg.arg}")
                parts += [f"{k.arg}={k.arg}" for k in args.kwonlyargs]
                if args.kwarg is not None:
                    parts.append(f"**{args.kwarg.arg}")
                # The signature spans the def header only (def line through the line whose
                # colon opens the body). node.body[0].lineno would skip over any leading
                # comment/blank lines (not AST nodes), wrongly absorbing them into the
                # delegate, so the header end is found by a bracket-balanced scan instead.
                depth = 0
                header_end = start
                for offset in range(start - 1, node.body[0].lineno - 1):
                    line = src_lines[offset]
                    depth += line.count("(") - line.count(")")
                    depth += line.count("[") - line.count("]")
                    if depth <= 0 and line.rstrip().endswith(":"):
                        header_end = offset + 1
                        break
                signature = src_lines[start - 1 : header_end]
                body_indent = " " * node.body[0].col_offset
                forward = (
                    f"{body_indent}return self.{leave_delegate}."
                    f"{delegate_name or name}({', '.join(parts)})" + src_nl
                )
                delegate = "".join(signature) + forward
                _write_source(
                    src_path,
                    "".join(src_lines[: start - 1] + [delegate] + src_lines[end:]),
                )
            else:
                _write_source(
                    src_path, "".join(src_lines[: start - 1] + src_lines[end:])
                )

            kept = [ln for ln in block if ln.strip() not in _MOVE_DECORATORS]
            if dedent:
                kept = [
                    ln[dedent:] if ln[:dedent] == " " * dedent else ln for ln in kept
                ]
            method_text = "".join(kept)
            if drop_self_annotation:
                method_text = _drop_self_annotation(method_text, name)

            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            dst_tree = ast.parse("".join(dst_lines))
            container = dst_tree.body
            if into_class is not None:
                cls = _find_class(dst_tree, into_class)
                assert cls is not None, f"class {into_class} not found in {dst}"
                container = cls.body
            target = None
            if before is not None:
                target = next(
                    (
                        n
                        for n in container
                        if isinstance(
                            n,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        )
                        and n.name == before
                    ),
                    None,
                )
            if target is not None:
                at = _def_span(target)[0] - 1
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [method_text, dst_nl] + dst_lines[at:]),
                )
            else:
                at = (
                    container[-1].end_lineno
                    if into_class is not None
                    else len(dst_lines)
                )
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [dst_nl, method_text] + dst_lines[at:]),
                )

        self.ops.append(op)
        return self

    def extract_to_new_module(
        self,
        src: str,
        dst: str,
        *,
        symbols: list[str],
        future_import: bool = True,
    ) -> "Repro":
        """Cut the contiguous tail of ``src`` -- the moved ``symbols`` and the module
        scaffolding that leads into them (imports, a ``TYPE_CHECKING`` guard, a logger,
        module constants) -- and write it as the new module ``dst``, prepending
        ``from __future__ import annotations`` when ``future_import``. The body is moved
        verbatim; the formatter sorts the imports and normalises the blank lines."""

        def op(root: Path) -> None:
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            body = ast.parse("".join(src_lines)).body
            wanted = set(symbols)
            scaffolding = (
                ast.Import,
                ast.ImportFrom,
                ast.If,
                ast.Assign,
                ast.AnnAssign,
            )
            cut = len(body)
            while cut > 0:
                node = body[cut - 1]
                is_symbol = (
                    isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and node.name in wanted
                )
                if is_symbol or isinstance(node, scaffolding):
                    cut -= 1
                else:
                    break
            tail = body[cut:]
            present = {
                node.name
                for node in tail
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            assert wanted <= present, f"{wanted - present} not in the cut tail of {src}"

            decorators = getattr(tail[0], "decorator_list", [])
            start = min([tail[0].lineno] + [d.lineno for d in decorators])
            block = "".join(src_lines[start - 1 :])
            _write_source(src_path, "".join(src_lines[: start - 1]))

            nl = _newline_style(block)
            prefix = "from __future__ import annotations" + nl if future_import else ""
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _write_source(dst_path, prefix + block)

        self.ops.append(op)
        return self

    def extract_symbols_to_new_module(
        self,
        src: str,
        dst: str,
        *,
        symbols: list[str],
        header: str,
        order: list[str],
        drop_assigns: list[str] | None = None,
    ) -> "Repro":
        """Relocate the named top-level defs/classes from *scattered* positions in ``src`` into
        a new module ``dst`` whose authored ``header`` (the imports, module constants, a logger,
        a ``TYPE_CHECKING`` block -- harmless or re-derived boilerplate reproduced from the
        target) precedes them. Unlike ``extract_to_new_module``, the symbols need not be a
        contiguous tail: each is cut from ``src`` verbatim, so its body stays a proven
        relocation, and the cut blocks are appended in ``order`` (their order in the target).
        ``drop_assigns`` names module-level assignments (e.g. a ``_is_hip = is_hip()`` constant)
        that moved into the new module's header, so they are deleted from ``src`` too -- their
        relocated copy is reproduced in the authored ``header``. The formatter normalises the
        spacing; the byte diff then certifies the bodies are exactly the source's, while only
        the small header is authored."""

        def op(root: Path) -> None:
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            src_nl = _newline_style("".join(src_lines))
            wanted = set(symbols)
            dropped = set(drop_assigns or [])
            assert set(order) == wanted, f"order {order} must permute symbols {symbols}"
            tree = ast.parse("".join(src_lines))
            nodes = {
                node.name: node
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name in wanted
            }
            missing = wanted - set(nodes)
            assert not missing, f"{missing} not top-level defs/classes in {src}"
            spans = {name: _def_span(node) for name, node in nodes.items()}
            blocks = {
                name: "".join(src_lines[start - 1 : end])
                for name, (start, end) in spans.items()
            }
            assign_spans: list[tuple[int, int]] = []
            found_assigns: set[str] = set()
            for node in tree.body:
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if names & dropped:
                    assign_spans.append((node.lineno, node.end_lineno))
                    found_assigns |= names & dropped
            assert (
                found_assigns == dropped
            ), f"{dropped - found_assigns} not assigned in {src}"
            cuts = list(spans.values()) + assign_spans
            for start, end in sorted(cuts, reverse=True):
                del src_lines[start - 1 : end]
            _write_source(src_path, "".join(src_lines))

            gap = src_nl * 3
            relocated = gap.join(blocks[name].rstrip("\r\n") for name in order)
            prefix = header.rstrip("\r\n") + gap if header.strip() else ""
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _write_source(dst_path, prefix + relocated + src_nl)

        self.ops.append(op)
        return self

    def extract_function(
        self,
        src: str,
        dst: str,
        *,
        name: str,
        signature: str,
        body: str,
        body_indent: int,
        call: str,
        return_text: str | None = None,
        before: str | None = None,
        into_class: str | None = None,
    ) -> "Repro":
        """Extract an inline block into a new ``name`` function. The block ``body`` is cut from
        ``src`` *verbatim* (so the byte diff certifies the function body is exactly the source's),
        re-indented from ``body_indent`` to a function-body indent, and wrapped under the
        authored ``signature`` (with ``return_text`` appended when given); the def is inserted
        into ``dst`` (above the sibling ``before`` or at the end of ``into_class`` / module), and
        the block in ``src`` is replaced by the authored ``call``.

        This is the certifiable core of an extract-function: the bulk (the relocated body) is
        machine-checked, and only the small signature/return/call interface is authored. It is
        faithful **only** when the body is moved unchanged -- a de-self (``self.x`` -> a
        parameter), a control-flow restructure, or a bookkeeping consolidation must be done as a
        separate semantic commit first, since those are not relocations (see
        mental-model-prep-and-move.md)."""

        def reindent(text: str, shift: int) -> str:
            if shift <= 0:
                return dedent(text, -shift)
            pad = " " * shift
            return "".join(
                pad + line if line.strip() else line
                for line in text.splitlines(keepends=True)
            )

        def op(root: Path) -> None:
            src_path = root / src
            src_text = _read_source(src_path)
            assert src_text.count(body) == 1, f"block not found uniquely in {src}"
            _write_source(src_path, src_text.replace(body, call, 1))

            dst_path = root / dst
            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            function = (
                signature.rstrip("\r\n") + dst_nl + reindent(body, 4 - body_indent)
            )
            if return_text is not None:
                function = function.rstrip("\r\n") + dst_nl + return_text
            function = function.rstrip("\r\n") + dst_nl

            dst_tree = ast.parse("".join(dst_lines))
            container = dst_tree.body
            if into_class is not None:
                cls = _find_class(dst_tree, into_class)
                assert cls is not None, f"class {into_class} not found in {dst}"
                container = cls.body
            anchor = None
            if before is not None:
                anchor = next(
                    (
                        node
                        for node in container
                        if isinstance(
                            node,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        )
                        and node.name == before
                    ),
                    None,
                )
            if anchor is not None:
                at = _def_span(anchor)[0] - 1
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [function, dst_nl] + dst_lines[at:]),
                )
            else:
                at = (
                    container[-1].end_lineno
                    if into_class is not None
                    else len(dst_lines)
                )
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [dst_nl, function] + dst_lines[at:]),
                )

        self.ops.append(op)
        return self

    def delete_file(self, path: str) -> "Repro":
        """Delete a source module that its symbols' relocation left empty (the chain deletes
        the leftover scaffolding-only file). Run after the moves that empty it."""

        def op(root: Path) -> None:
            target = root / path
            if target.exists():
                target.unlink()

        self.ops.append(op)
        return self

    def run(self) -> str:
        """Apply the operations to a worktree at base, run pre-commit, diff against target.
        Returns the residual diff ("" on a clean reproduction)."""
        repo_root = self.repo_root or exec_command("git rev-parse --show-toplevel")
        worktree = tempfile.mkdtemp(prefix="repro-")
        branch = Path(worktree).name
        try:
            exec_command(
                f"git worktree add -b {branch} {worktree} {self.base}", cwd=repo_root
            )
            for op in self.ops:
                op(Path(worktree))
            exec_command("git add -A", cwd=worktree)
            changed = exec_command(
                f"git diff --cached --name-only --diff-filter=ACMR {self.base}",
                cwd=worktree,
            ).split()
            if changed:
                files = " ".join(shlex.quote(path) for path in changed)
                exec_command(
                    f"pre-commit run --files {files}", cwd=worktree, check=False
                )
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
