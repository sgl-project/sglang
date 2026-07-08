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
import io
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
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


def _find_unique_def(
    tree: ast.AST, name: str, *, from_class: str | None = None, where: str
) -> ast.AST:
    """Resolve ``def name`` and refuse ambiguity: with same-named defs in scope the
    first-match lookup could silently cut the wrong body, so the caller must scope the
    search with ``from_class``."""
    root: ast.AST = tree
    if from_class is not None:
        cls = _find_class(tree, from_class)
        assert cls is not None, f"class {from_class} not found in {where}"
        root = cls
    elif isinstance(tree, ast.Module):
        top_level = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(top_level) == 1:
            return top_level[0]
    matches = [
        node
        for node in ast.walk(root)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert matches, f"{name} not found in {where}"
    assert (
        len(matches) == 1
    ), f"{len(matches)} defs named {name} in {where}; pass from_class to disambiguate"
    return matches[0]


def _def_header_end(def_text: str) -> int:
    """1-based line (within ``def_text``, which starts at the def line) of the colon that
    opens the body. Tokenize-based, so parentheses inside string defaults do not confuse
    the bracket depth."""
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(def_text).readline):
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0:
                return token.start[0]
    raise AssertionError("no header-ending colon found")


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


def _node_slice(text: str, node: ast.AST) -> str:
    return _slice_span(
        text, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
    )


def _rewrite_matching_calls(
    text: str, predicate: "Callable", rewrite: "Callable"
) -> str:
    """Rewrite every call ``predicate`` accepts by splicing the original source text
    (never ``ast.unparse``, which would re-spell literals and drop comments). One call is
    rewritten per pass and the text re-parsed, so a matching call nested inside another
    match is rewritten on a later pass instead of being overwritten."""
    while True:
        node = next(
            (
                n
                for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Call) and predicate(n)
            ),
            None,
        )
        if node is None:
            return text
        text = _replace_span(
            text,
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
            rewrite(text, node),
        )


def _lowered_call_text(text: str, node: ast.Call) -> str:
    """Original call text with the leading receiver argument spliced out and made the
    call's new receiver: ``Owner.foo(recv, rest...)`` -> ``recv.foo(rest...)``. All other
    argument bytes (literal spelling, comments, the magic trailing comma) are untouched.
    """
    receiver = node.args[0]
    receiver_src = _node_slice(text, receiver)
    assert (
        "\n" not in receiver_src and "#" not in receiver_src
    ), f"receiver {receiver_src!r} must be single-line and comment-free"
    opener = _slice_span(
        text,
        node.func.end_lineno,
        node.func.end_col_offset,
        receiver.lineno,
        receiver.col_offset,
    )
    assert "#" not in opener, f"comment before the receiver in {opener!r}"
    seg = _slice_span(
        text,
        receiver.end_lineno,
        receiver.end_col_offset,
        node.end_lineno,
        node.end_col_offset,
    )
    head, comma, rest = seg.partition(",")
    assert "#" not in head, f"comment after the receiver in {head!r}"
    if comma:
        assert head.strip() == "", f"unexpected text {head!r} after the receiver"
        rest = rest.lstrip(" \t")
    else:
        assert head.strip() == ")", f"unexpected text {head!r} after the receiver"
        rest = head.lstrip(" \t")
    return f"{receiver_src}.{node.func.attr}({rest}"


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


def _multiline_string_interior_lines(top_level_text: str) -> set[int]:
    """1-based lines of ``top_level_text`` that lie inside a multi-line string token
    (every line after the token's opening line, including the closing-delimiter line).
    Re-indenting those lines would change the literal's value, not its layout."""
    interior: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(top_level_text).readline):
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            interior.update(range(token.start[0] + 1, token.end[0] + 1))
    return interior


def _audit_extract_header(
    header: str, removed_assigns: dict[str, str | None], where: str
) -> None:
    """Refuse header content the extraction cannot vouch for. The header of a scattered
    extraction is authored text reproduced from the target commit, so anything beyond
    imports, a TYPE_CHECKING import block, a logger, or a byte-equivalent copy of an
    assignment deleted from the source would let arbitrary new code ride into the new
    module under a PASS verdict."""
    header_assigned: set[str] = set()
    for stmt in ast.parse(header).body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
            and all(isinstance(sub, (ast.Import, ast.ImportFrom)) for sub in stmt.body)
        ):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [x.id for x in targets if isinstance(x, ast.Name)]
            value_src = ast.unparse(stmt.value) if stmt.value is not None else None
            if value_src == "logging.getLogger(__name__)":
                continue
            if names and all(
                n in removed_assigns and removed_assigns[n] == value_src for n in names
            ):
                header_assigned.update(names)
                continue
        raise AssertionError(
            f"unverifiable header statement in {where}: {ast.unparse(stmt)!r} is "
            f"neither scaffolding nor a relocated source assignment"
        )
    missing = set(removed_assigns) - header_assigned
    assert not missing, (
        f"drop_assigns {sorted(missing)} deleted from the source but not reproduced "
        f"in the header of {where}"
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

                _write_source(
                    path,
                    _rewrite_matching_calls(
                        _read_source(path), predicate, _lowered_call_text
                    ),
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

                def rewrite(text: str, node: ast.Call) -> str:
                    call_src = _node_slice(text, node)
                    func_src = _node_slice(text, node.func)
                    return name + call_src[len(func_src) :]

                _write_source(
                    path,
                    _rewrite_matching_calls(_read_source(path), predicate, rewrite),
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
                fn = _find_unique_def(tree, in_function, where=rel)
                scope = (fn.lineno, fn.end_lineno)
            compound = (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.Match,
            )
            simple_stmt_lines: dict[int, int] = {}
            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.stmt) and not isinstance(stmt, compound):
                    for lineno in range(stmt.lineno, stmt.end_lineno + 1):
                        simple_stmt_lines[lineno] = simple_stmt_lines.get(lineno, 0) + 1
            pattern = re.compile(rf"(?<![\w.]){re.escape(import_text)}(?![\w.])")
            matched: list[ast.stmt] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if scope is not None and not (scope[0] <= node.lineno <= scope[1]):
                        continue
                    seg = _slice_span(
                        "".join(lines),
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )
                    if pattern.search(seg):
                        matched.append(node)
            assert matched, f"import {import_text!r} not found in {rel}"
            whole: list[tuple[int, int]] = []
            shared: list[ast.stmt] = []
            for node in matched:
                alone = all(
                    simple_stmt_lines[lineno] == 1
                    for lineno in range(node.lineno, node.end_lineno + 1)
                )
                if alone:
                    lo, hi = node.lineno, node.end_lineno
                    if hi < len(lines) and lines[hi].strip() == "":
                        hi += 1
                    whole.append((lo, hi))
                else:
                    shared.append(node)
            text = "".join(lines)
            for node in sorted(
                shared, key=lambda n: (n.lineno, n.col_offset), reverse=True
            ):
                text = _replace_span(
                    text,
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                    "",
                )
                fixed = _split_keepends(text)
                joined_line = fixed[node.lineno - 1]
                cleaned = re.sub(r";\s*;", ";", joined_line)
                cleaned = re.sub(r"^(\s*);\s*", r"\1", cleaned)
                cleaned = re.sub(r"\s*;(\s*)$", r"\1", cleaned)
                fixed[node.lineno - 1] = cleaned
                text = "".join(fixed)
            lines = _split_keepends(text)
            for lo, hi in sorted(whole, reverse=True):
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
                dropped_alias = next(
                    (a for a in node.names if a.name == name and a.asname == asname),
                    None,
                )
                if dropped_alias is None:
                    continue
                kept = [a for a in node.names if a is not dropped_alias]
                if not kept:
                    edits.append((node.lineno, node.end_lineno, None))
                    continue
                stmt_lines = lines[node.lineno - 1 : node.end_lineno]
                if any("#" in ln for ln in stmt_lines):
                    own = dropped_alias.lineno
                    own_line = lines[own - 1]
                    assert own_line.strip().rstrip(",").strip() == alias_text(
                        dropped_alias
                    ), (
                        f"cannot drop {name!r}: it shares a line with other text and "
                        f"the import holds comments that a rebuild would delete"
                    )
                    edits.append((own, own, None))
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
            body = ast.parse("".join(lines)).body
            last = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                last = body[0].end_lineno
            for node in body:
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
                    spelled = "." * node.level + (node.module or "")
                    replaced = lines[node.lineno - 1].replace(
                        f"from {spelled} import", f"from {new_module} import", 1
                    )
                    assert (
                        replaced != lines[node.lineno - 1]
                    ), f"import spelling {spelled!r} not found on its line in {rel}"
                    lines[node.lineno - 1] = replaced
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
        from_class: str | None = None,
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
            node = _find_unique_def(
                ast.parse("".join(src_lines)), name, from_class=from_class, where=src
            )
            start, end = _def_span(node)
            block = src_lines[start - 1 : end]
            decorator_lines = node.lineno - start
            if leave_delegate is not None:
                assert not any(
                    ln.strip() in _MOVE_DECORATORS for ln in block[:decorator_lines]
                ), f"leave_delegate on a {_MOVE_DECORATORS} method has no self to forward"
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
                # delegate, so the header end is found by tokenizing the def.
                header_end = (
                    node.lineno
                    - 1
                    + _def_header_end("".join(src_lines[node.lineno - 1 : end]))
                )
                signature = src_lines[start - 1 : header_end]
                body_indent = " " * node.body[0].col_offset
                returning = (
                    "return await"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "return"
                )
                forward = (
                    f"{body_indent}{returning} self.{leave_delegate}."
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

            kept = [
                ln
                for index, ln in enumerate(block)
                if not (index < decorator_lines and ln.strip() in _MOVE_DECORATORS)
            ]
            if dedent > 0:
                kept = [
                    ln[dedent:] if ln[:dedent] == " " * dedent else ln for ln in kept
                ]
            elif dedent < 0:
                pad = " " * -dedent
                kept = [pad + ln if ln.strip() else ln for ln in kept]
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
                assert target is not None, f"before={before!r} not found in {dst}"
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

            def is_scaffolding(node: ast.stmt) -> bool:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return True
                if isinstance(node, ast.If):
                    return ast.unparse(node.test) in (
                        "TYPE_CHECKING",
                        "typing.TYPE_CHECKING",
                    )
                if isinstance(node, ast.Assign):
                    return all(isinstance(x, ast.Name) for x in node.targets)
                if isinstance(node, ast.AnnAssign):
                    return isinstance(node.target, ast.Name)
                return False

            cut = len(body)
            while cut > 0:
                node = body[cut - 1]
                is_symbol = (
                    isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and node.name in wanted
                )
                if is_symbol or is_scaffolding(node):
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
            assign_rewrites: list[tuple[int, int, str]] = []
            removed_assigns: dict[str, str] = {}
            found_assigns: set[str] = set()
            for node in tree.body:
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                hit = names & dropped
                if not hit:
                    continue
                assert len(names) == len(
                    targets
                ), f"drop_assigns {sorted(hit)}: non-name targets in {src}"
                value_src = ast.unparse(node.value) if node.value is not None else None
                for dropped_name in hit:
                    removed_assigns[dropped_name] = value_src
                surviving = [
                    x.id
                    for x in targets
                    if isinstance(x, ast.Name) and x.id not in dropped
                ]
                if surviving:
                    kept_stmt = (
                        " = ".join(surviving)
                        + " = "
                        + _slice_span(
                            "".join(src_lines),
                            node.value.lineno,
                            node.value.col_offset,
                            node.value.end_lineno,
                            node.value.end_col_offset,
                        )
                        + src_nl
                    )
                    assign_rewrites.append((node.lineno, node.end_lineno, kept_stmt))
                else:
                    assign_spans.append((node.lineno, node.end_lineno))
                found_assigns |= hit
            assert (
                found_assigns == dropped
            ), f"{dropped - found_assigns} not assigned in {src}"
            if header.strip() or removed_assigns:
                _audit_extract_header(header, removed_assigns, where=dst)
            cuts = [(start, end, None) for start, end in spans.values()]
            cuts += [(start, end, None) for start, end in assign_spans]
            cuts += assign_rewrites
            for start, end, repl in sorted(
                cuts, key=lambda c: (c[0], c[1]), reverse=True
            ):
                if repl is None:
                    del src_lines[start - 1 : end]
                else:
                    src_lines[start - 1 : end] = [repl]
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
            if shift == 0:
                return text
            interior = _multiline_string_interior_lines(dedent(text, body_indent))
            lines = _split_keepends(text)
            if shift < 0:
                return "".join(
                    (
                        line[-shift:]
                        if index + 1 not in interior and line[:-shift] == " " * -shift
                        else line
                    )
                    for index, line in enumerate(lines)
                )
            pad = " " * shift
            return "".join(
                pad + line if line.strip() and index + 1 not in interior else line
                for index, line in enumerate(lines)
            )

        def op(root: Path) -> None:
            src_path = root / src
            src_text = _read_source(src_path)
            assert src_text.count(body) == 1, f"block not found uniquely in {src}"
            at = src_text.find(body)
            assert (
                at == 0 or src_text[at - 1] == "\n"
            ), f"block matches mid-line in {src}; it must start at a line boundary"
            _write_source(src_path, src_text.replace(body, call, 1))

            dst_path = root / dst
            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            sig_first = _split_keepends(signature)[0]
            sig_indent = len(sig_first) - len(sig_first.lstrip(" "))
            function = (
                signature.rstrip("\r\n")
                + dst_nl
                + reindent(body, sig_indent + 4 - body_indent)
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
        the leftover scaffolding-only file). Run after the moves that empty it. Refuses a
        file that still holds anything beyond a docstring, imports, or a TYPE_CHECKING
        block -- deleting live code is not a relocation."""

        def op(root: Path) -> None:
            target = root / path
            if not target.exists():
                return
            leftover = [
                ast.unparse(stmt)
                for stmt in ast.parse(_read_source(target)).body
                if not (
                    isinstance(stmt, (ast.Import, ast.ImportFrom))
                    or (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    )
                    or (
                        isinstance(stmt, ast.If)
                        and ast.unparse(stmt.test)
                        in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
                    )
                )
            ]
            assert not leftover, (
                f"{path} still holds non-scaffolding code, refusing to delete: "
                f"{leftover[:3]}"
            )
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
