"""Infer a faithful reproduce recipe for a move commit, then emit and run a self-contained
reproduce script. Lets the verifier turn a commit a formatter re-wrapped into an auditable,
runnable reproduce script -- no one hand-writes it.

A recipe is inferred from the commit's diff and its before-state AST: which symbols moved
(src -> dst, into which class, or into a new module), which call sites were adapted, which
imports were repathed, and the symmetric module-level import diff each file gained or lost
(realised directly with add_import / remove_imported_name, since an import diff is always
whitelisted).
``recipe_to_script`` emits a standalone ``repro_scripts/<sha>.py`` (importing only the
reproduce util); running it reproduces the commit and diffs it byte-for-byte.
``generate_range`` writes a whole folder (scripts + output.log + output.html) for a range.

Handles a method moved onto an existing class (call sites lowered), a method moved to a
module-level free function (call sites requalified), a free-function-source move to an
existing module (callers repath their import), and a new-file extract -- where the prep
commit staged the whole module body (scaffolding plus def) as a trailing block in the
source, so the move cuts that tail into the new file (extract_to_new_module). A rename or a
statement-level reorder relocates no def and is reported unsupported. Runnable directly:

    python3 mechanical_refactor_generate_proof.py <commit>
    python3 mechanical_refactor_generate_proof.py <base>..<tip> --match -move: --out DIR
"""

import ast
import html
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_reproduce_utils as rr


def _git_output(args: list[str], cwd: str) -> str:
    """Raw stdout of a git command ("" if it fails). Not stripped, so ``ast`` line numbers
    stay aligned with a file's real lines."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _removed_symbol_names(lines: list[str]) -> set[str]:
    """Names of top-level defs and classes among removed diff lines (the extract's source
    relinquishes these), so a moved class is found, not just a moved def."""
    return {
        m.group(2)
        for ln in lines
        if (m := re.match(r"\s*(?:async\s+)?(def|class)\s+(\w+)", ln))
    }


def _def_indent(lines: list[str], name: str) -> int | None:
    for line in lines:
        match = re.match(r"(\s*)(?:async\s+)?def\s+" + re.escape(name) + r"\b", line)
        if match:
            return len(match.group(1))
    return None


def _per_file_diff(commit: str, root: str) -> dict[str, dict]:
    """Per-file removed/added content lines (whitespace intact) and a new-file flag."""
    out = _git_output(
        ["show", commit, "--format=", "--no-color", "--no-ext-diff"], root
    )
    files: dict[str, dict] = {}
    path: str | None = None
    for line in out.splitlines():
        if line.startswith("diff --git"):
            path = line.split(" b/")[-1]
            files[path] = {"removed": [], "added": [], "new": False, "deleted": False}
        elif line.startswith("new file"):
            files[path]["new"] = True
        elif line.startswith("deleted file"):
            files[path]["deleted"] = True
        elif line.startswith(("+++", "---", "@@", "index ")):
            continue
        elif line.startswith("+"):
            files[path]["added"].append(line[1:])
        elif line.startswith("-"):
            files[path]["removed"].append(line[1:])
    return files


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= node.end_lineno:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.name)
    return best[1] if best else None


def _enclosing_class_of_def(tree: ast.AST, name: str) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == name
                ):
                    return node.name
    return None


def _nested_in_function(tree: ast.AST, name: str) -> bool:
    target = rr._find_def(tree, name)
    if target is None:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not target
            and node.lineno <= target.lineno <= node.end_lineno
        ):
            return True
    return False


def _module_of_path(path: str) -> str:
    return path.removeprefix("python/").removesuffix(".py").replace("/", ".")


def _import_pairs(text: str) -> dict:
    """Module-level imports keyed one entry per imported name, so removing a single name from
    a multi-name ``from x import a, b`` is not mistaken for the whole statement changing. The
    value is the one-name ``add_import`` text for a gained name (the import sorter merges it).
    """
    pairs: dict = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                stmt = "import " + alias.name
                if alias.asname:
                    stmt += f" as {alias.asname}"
                pairs[stmt] = stmt
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                name = alias.name + (f" as {alias.asname}" if alias.asname else "")
                pairs[(module, alias.name, alias.asname)] = (
                    f"from {module} import {name}"
                )
    return pairs


def _typechecking_pairs(text: str) -> dict:
    """Imports inside the module's ``if TYPE_CHECKING:`` block, keyed per name (same shape as
    ``_import_pairs``), so a type-only import the destination gains for a moved annotation is
    inferable separately from the runtime imports."""
    pairs: dict = {}
    for node in ast.parse(text).body:
        if not (
            isinstance(node, ast.If)
            and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
        ):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                module = "." * stmt.level + (stmt.module or "")
                for alias in stmt.names:
                    name = alias.name + (f" as {alias.asname}" if alias.asname else "")
                    pairs[(module, alias.name, alias.asname)] = (
                        f"from {module} import {name}"
                    )
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    stmt_text = "import " + alias.name
                    if alias.asname:
                        stmt_text += f" as {alias.asname}"
                    pairs[stmt_text] = stmt_text
    return pairs


def _local_import_of(
    tree: ast.AST, fn_name: str, module: str, symbol: str
) -> str | None:
    fn = rr._find_def(tree, fn_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any(alias.name == symbol for alias in node.names)
        ):
            return f"from {module} import {symbol}"
    return None


def _removal_from_key(path: str, key) -> dict:
    """Turn an ``_import_pairs`` key the target dropped into a ``remove_imported_name`` call. A
    ``(module, name, asname)`` key drops one name from a ``from`` import; a bare ``import x``
    statement string drops a plain import."""
    if isinstance(key, tuple):
        module, name, asname = key
        return {"path": path, "module": module, "name": name, "asname": asname}
    rest = key.removeprefix("import ")
    if " as " in rest:
        name, asname = rest.split(" as ", 1)
    else:
        name, asname = rest, None
    return {"path": path, "module": None, "name": name, "asname": asname}


@dataclass
class Recipe:
    base: str
    target: str
    supported: bool = True
    moves: list = field(default_factory=list)
    extracts: list = field(default_factory=list)
    lowerings: list = field(default_factory=list)
    repaths: list = field(default_factory=list)
    import_removals: list = field(default_factory=list)
    module_import_removals: list = field(default_factory=list)
    import_additions: list = field(default_factory=list)
    typechecking_additions: list = field(default_factory=list)
    deletes: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _infer_call_adaptations(
    recipe: Recipe,
    files: dict[str, dict],
    *,
    name: str,
    src: str,
    src_class: str,
    into_class: str | None,
    commit: str,
    root: str,
) -> None:
    """A method-source move adapts its call sites: a move onto a class lowers the receiver
    out of the args; a move to a module-level free function drops the qualifier. A caller is
    a before-state call ``<src_class>.name(...)`` -- matched on ``src_class`` (not a loose
    text search) so the moved body's own same-named calls on a different receiver are
    excluded -- and its orphaned local import of ``src_class`` is removed."""
    kind = "lower" if into_class is not None else "requalify"
    src_module = _module_of_path(src)
    for path, f in files.items():
        before = _git_output(["show", f"{commit}^:{path}"], root)
        try:
            tree = ast.parse(before)
        except SyntaxError:
            continue
        caller_fns: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                and (node.args or kind == "requalify")
                and ast.unparse(node.func.value) == src_class
            ):
                fn = _enclosing_function(tree, node.lineno)
                if fn is not None:
                    caller_fns.add(fn)
        if not caller_fns:
            continue
        recipe.lowerings.append(
            {"name": name, "owner": src_class, "path": path, "kind": kind}
        )
        for fn in sorted(caller_fns):
            imp = _local_import_of(tree, fn, src_module, src_class)
            if imp is not None and any(imp in r for r in f["removed"]):
                recipe.import_removals.append(
                    {"path": path, "text": imp, "in_function": fn}
                )


def _infer_function_scoped_repaths(
    recipe: Recipe,
    files: dict[str, dict],
    *,
    name: str,
    src: str,
    dst: str,
    commit: str,
    root: str,
) -> None:
    """A moved free function keeps the same bare call, so a caller only repaths its import.
    Module-level repaths fall out of the symmetric import diff; only function-scoped imports
    (which ``add_import`` cannot place) need an explicit in-place repath."""
    src_module = _module_of_path(src)
    dst_module = _module_of_path(dst)
    for path in sorted(files):
        if path == src:
            continue
        before = _git_output(["show", f"{commit}^:{path}"], root)
        try:
            tree = ast.parse(before)
        except SyntaxError:
            continue
        top_level = {id(node) for node in tree.body}
        nested = any(
            isinstance(node, ast.ImportFrom)
            and id(node) not in top_level
            and node.module == src_module
            and any(alias.name == name for alias in node.names)
            for node in ast.walk(tree)
        )
        if nested:
            recipe.repaths.append(
                {
                    "path": path,
                    "old_module": src_module,
                    "new_module": dst_module,
                    "name": name,
                }
            )


def _self_annotation_dropped(src_def: ast.AST | None, dst_def: ast.AST | None) -> bool:
    """Whether the move drops a ``self: Target`` annotation -- the source had it and the
    destination does not. Some class moves keep it (a retyped self that stays annotated), so
    this is inferred from both sides rather than assumed."""

    def has_self_annotation(node: ast.AST | None) -> bool:
        return bool(
            node is not None
            and node.args.args
            and node.args.args[0].arg == "self"
            and node.args.args[0].annotation is not None
        )

    return has_self_annotation(src_def) and not has_self_annotation(dst_def)


def _wants_future_import(files: dict[str, dict], src: str, dst: str) -> bool:
    future = "from __future__ import annotations"
    gained = any(future in line for line in files[dst]["added"])
    travelled = any(future in line for line in files[src]["removed"])
    return gained and not travelled


def _next_sibling_def_name(
    dst_tree: ast.AST, name: str, into_class: str | None
) -> str | None:
    """The name of the def that immediately follows ``name`` at its scope in the destination
    (module level, or inside ``into_class``), or None when ``name`` is the last def there. Lets
    a move reinsert the relocated def in the chain's order instead of appending at the end.
    """
    container: list = []
    if into_class is not None:
        cls = rr._find_class(dst_tree, into_class)
        container = cls.body if cls is not None else []
    else:
        container = getattr(dst_tree, "body", [])
    defs = [
        n
        for n in container
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for i, node in enumerate(defs):
        if node.name == name:
            return defs[i + 1].name if i + 1 < len(defs) else None
    return None


def _symbols_form_tail(src_text: str, symbols: list[str]) -> bool:
    """Whether ``symbols`` sit at the end of the source as a contiguous block of defs/classes
    and the scaffolding leading into them -- the trailing block a prep commit stages for a
    new-module extract. A method still inside a class, or a symbol separated from the tail by
    other code, fails this and is not an extractable tail."""
    body = ast.parse(src_text).body
    wanted = set(symbols)
    scaffolding = (ast.Import, ast.ImportFrom, ast.If, ast.Assign, ast.AnnAssign)
    cut = len(body)
    while cut > 0:
        node = body[cut - 1]
        is_symbol = (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in wanted
        )
        if is_symbol or isinstance(node, scaffolding):
            cut -= 1
        else:
            break
    present = {
        node.name
        for node in body[cut:]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return wanted <= present


def infer_recipe(commit: str, root: str) -> Recipe:
    """Infer a faithful relocation recipe for a move commit from its diff + before-state.

    A move onto an existing module/class becomes a ``move_symbol``; a move whose destination
    file is new becomes an ``extract_to_new_module`` (the prep commit staged the whole module
    body -- scaffolding plus def -- as a trailing block in the source, so the move cuts that
    tail into the new file). Method-source moves adapt their call sites; free-function-source
    moves keep the bare call and only repath imports. A rename or a statement-level reorder
    relocates no def, so nothing is inferred and the commit is reported unsupported."""
    files = _per_file_diff(commit, root)
    recipe = Recipe(base=f"{commit}~1", target=commit)

    def def_names(lines: list[str]) -> set[str]:
        return {
            m.group(1)
            for ln in lines
            if (m := re.match(r"\s*(?:async\s+)?def\s+(\w+)", ln))
        }

    new_files = {p for p, f in files.items() if f["new"]}

    # A new file is a staged module body cut from one source: its top-level defs and classes
    # are exactly the relocated symbols (the prep commit inlined them, scaffolding included, as
    # a trailing block in the source). Take the symbol list from the new file itself so a
    # moved class -- not just a moved def -- is in the cut tail.
    for dst in sorted(new_files):
        dst_after = _git_output(["show", f"{commit}:{dst}"], root)
        try:
            dst_body = ast.parse(dst_after).body
        except SyntaxError:
            continue
        symbols = [
            node.name
            for node in dst_body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not symbols:
            continue
        srcs = {
            p
            for name in symbols
            for p, f in files.items()
            if p not in new_files and name in _removed_symbol_names(f["removed"])
        }
        if len(srcs) != 1:
            recipe.supported = False
            recipe.notes.append(
                f"{dst}: extract source not a single file ({sorted(srcs)})"
            )
            continue
        src = next(iter(srcs))
        if not _symbols_form_tail(
            _git_output(["show", f"{commit}^:{src}"], root), symbols
        ):
            recipe.supported = False
            recipe.notes.append(
                f"{dst}: source is not a staged trailing block "
                "(prep should inline the module body at the source tail first)"
            )
            continue
        recipe.extracts.append(
            {
                "src": src,
                "dst": dst,
                "symbols": symbols,
                "future_import": _wants_future_import(files, src, dst),
            }
        )

    # A move whose destination already exists becomes a move_symbol (the def relocated in
    # order); a moved class to an existing file is left unsupported (move_symbol moves defs).
    all_removed = [ln for f in files.values() for ln in f["removed"]]
    all_added = [ln for f in files.values() for ln in f["added"]]
    for name in sorted(def_names(all_removed) & def_names(all_added)):
        src = next(
            (p for p, f in files.items() if name in def_names(f["removed"])), None
        )
        dst = next((p for p, f in files.items() if name in def_names(f["added"])), None)
        if src is None or dst is None or src == dst or dst in new_files:
            continue
        src_before = _git_output(["show", f"{commit}^:{src}"], root)
        if _nested_in_function(ast.parse(src_before), name):
            recipe.notes.append(f"skip {name}: nested function (moves with parent)")
            continue
        src_tree = ast.parse(src_before)
        src_class = _enclosing_class_of_def(src_tree, name)
        dst_tree = ast.parse(_git_output(["show", f"{commit}:{dst}"], root))
        into_class = _enclosing_class_of_def(dst_tree, name)
        dst_def = rr._find_def(dst_tree, name)
        src_indent = _def_indent(files[src]["removed"], name) or 0
        dst_indent = _def_indent(files[dst]["added"], name) or 0
        recipe.moves.append(
            {
                "name": name,
                "src": src,
                "dst": dst,
                "into_class": into_class,
                "dedent": src_indent - dst_indent,
                "dst_order": dst_def.lineno if dst_def else 0,
                "before": _next_sibling_def_name(dst_tree, name, into_class),
                "drop_self_annotation": _self_annotation_dropped(
                    rr._find_def(src_tree, name), dst_def
                ),
            }
        )
        if src_class is not None:
            _infer_call_adaptations(
                recipe,
                files,
                name=name,
                src=src,
                src_class=src_class,
                into_class=into_class,
                commit=commit,
                root=root,
            )
        else:
            _infer_function_scoped_repaths(
                recipe, files, name=name, src=src, dst=dst, commit=commit, root=root
            )

    # Module-level imports a file gained or lost are realised directly from the symmetric
    # base<->target diff: a gained name is added (the destination needs the moved code's
    # imports, or a caller of a moved free function gains one), a lost name is removed. An
    # import diff is always whitelisted, so this is deterministic and does not depend on the
    # formatter pruning (this repo's ruff has no F811, so a still-used symbol repointed to a new
    # module would otherwise leave a duplicate). A file written whole by extract_to_new_module
    # (the new file, or the extract source whose tail the cut removed) is skipped.
    extract_dsts = {ex["dst"] for ex in recipe.extracts}
    extract_srcs = {ex["src"] for ex in recipe.extracts}
    for path in sorted(files):
        if path in new_files or path in extract_dsts:
            continue
        before = _git_output(["show", f"{commit}^:{path}"], root)
        after = _git_output(["show", f"{commit}:{path}"], root)
        before_pairs = _import_pairs(before) if before.strip() else {}
        after_pairs = _import_pairs(after) if after.strip() else {}
        for key, stmt in after_pairs.items():
            if key not in before_pairs:
                recipe.import_additions.append({"path": path, "text": stmt})
        if path not in extract_srcs:
            for key in before_pairs:
                if key not in after_pairs:
                    recipe.module_import_removals.append(_removal_from_key(path, key))
        before_tc = _typechecking_pairs(before) if before.strip() else {}
        after_tc = _typechecking_pairs(after) if after.strip() else {}
        for key, stmt in after_tc.items():
            if key not in before_tc:
                recipe.typechecking_additions.append({"path": path, "text": stmt})

    # A move source the commit deletes (its defs all relocated, leaving only scaffolding) is
    # removed after the moves; move_symbol only cuts defs, it does not delete the emptied file.
    move_srcs = {mv["src"] for mv in recipe.moves}
    for path, f in files.items():
        if f.get("deleted") and path in move_srcs:
            recipe.deletes.append(path)

    if not recipe.moves and not recipe.extracts:
        recipe.supported = False
        if not recipe.notes:
            recipe.notes.append(
                "no def relocated (rename or statement-level change): review as prep"
            )
    return recipe


def _recipe_ops(recipe: Recipe) -> list:
    """The ordered relocation operations a recipe replays, as ``(method, args, kwargs)`` --
    shared by ``build_repro`` (which runs them on a Repro) and ``recipe_to_script`` (which
    renders them as ``r.method(...)`` lines), so the emitted script and the in-process run can
    never drift.

    Call sites and import repaths/removals run BEFORE the moves, so a call to a moved method
    from inside another moved method is adapted while still in the source and travels with the
    body. The moves (in destination order) and the new-module extracts relocate next.
    Module-level import additions/removals run LAST, so a consumer import lands after an extract
    has cut the source tail (otherwise it would be swept into the new module). Same-destination
    moves are emitted in reverse destination order so each move's ``before`` anchor (a sibling
    further down) is already present when the move is inserted."""
    ops: list = []
    for lo in recipe.lowerings:
        method = (
            "requalify_call_sites" if lo["kind"] == "requalify" else "lower_call_sites"
        )
        ops.append((method, (lo["name"], lo["owner"]), {"paths": [lo["path"]]}))
    for rp in recipe.repaths:
        ops.append(
            (
                "repath_import",
                (rp["path"],),
                {
                    "old_module": rp["old_module"],
                    "new_module": rp["new_module"],
                    "name": rp["name"],
                },
            )
        )
    for im in recipe.import_removals:
        ops.append(
            (
                "remove_import",
                (im["path"], im["text"]),
                {"in_function": im["in_function"]},
            )
        )
    for mv in sorted(recipe.moves, key=lambda m: (m["dst"], -m["dst_order"])):
        ops.append(
            (
                "move_symbol",
                (mv["name"],),
                {
                    "src": mv["src"],
                    "dst": mv["dst"],
                    "into_class": mv["into_class"],
                    "dedent": mv["dedent"],
                    "drop_self_annotation": mv["drop_self_annotation"],
                    "before": mv.get("before"),
                },
            )
        )
    for ex in recipe.extracts:
        ops.append(
            (
                "extract_to_new_module",
                (ex["src"], ex["dst"]),
                {"symbols": ex["symbols"], "future_import": ex["future_import"]},
            )
        )
    for path in recipe.deletes:
        ops.append(("delete_file", (path,), {}))
    for im in recipe.module_import_removals:
        ops.append(
            (
                "remove_imported_name",
                (im["path"],),
                {"module": im["module"], "name": im["name"], "asname": im["asname"]},
            )
        )
    for im in recipe.import_additions:
        ops.append(("add_import", (im["path"], im["text"]), {}))
    for im in recipe.typechecking_additions:
        ops.append(("add_typechecking_import", (im["path"], im["text"]), {}))
    return ops


def build_repro(recipe: Recipe) -> rr.Repro:
    """Compose a Repro from the recipe's canonical ordered operations (``_recipe_ops``)."""
    repro = rr.Repro(base=recipe.base, target=recipe.target)
    for method, args, kwargs in _recipe_ops(recipe):
        getattr(repro, method)(*args, **kwargs)
    return repro


def recipe_to_script(recipe: Recipe, subject: str) -> str:
    """A standalone, auditable reproduce script (imports only the reproduce util)."""
    lines = [
        '"""Auto-generated reproduce script. Audit each call, then run.',
        "",
        f"commit:  {recipe.target}",
        f"subject: {subject}",
        "",
        "Each call is a faithful relocation primitive. Running this reproduces the commit",
        "in a throwaway worktree and diffs it byte-for-byte; PASS means the commit is",
        "exactly these relocations.",
        '"""',
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
        "from mechanical_refactor_reproduce_utils import Repro",
        "",
        f"r = Repro(base={recipe.base!r}, target={recipe.target!r})",
    ]
    for method, args, kwargs in _recipe_ops(recipe):
        rendered = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        lines.append(f"r.{method}(" + ", ".join(rendered) + ")")
    lines += ["r.run()", ""]
    return "\n".join(lines)


@dataclass
class GenResult:
    commit: str
    subject: str
    supported: bool
    passed: bool
    residual: str
    script: str
    notes: list


def generate_range(
    rev_range: str,
    *,
    match: str | None = None,
    out_dir: str,
    repo_root: str | None = None,
) -> list[GenResult]:
    """For each matched commit: infer a recipe, emit repro_scripts/<sha>.py, run it, and
    record PASS / residual. Writes output.log + output.html and copies the reproduce util
    so the folder is self-contained."""
    root = repo_root or _repo_root()
    commits = _git_output(["rev-list", "--reverse", rev_range], root).split()
    pattern = re.compile(match) if match else None

    out = Path(out_dir)
    scripts_dir = out / "repro_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (out / "mechanical_refactor_reproduce_utils.py").write_text(
        Path(rr.__file__).read_text()
    )

    results: list[GenResult] = []
    for commit in commits:
        subject = _git_output(["log", "-1", "--format=%s", commit], root).strip()
        if pattern is not None and not pattern.search(subject):
            continue
        recipe = infer_recipe(commit, root)
        script = recipe_to_script(recipe, subject)
        (scripts_dir / f"{commit[:9]}.py").write_text(script)
        passed, residual = False, ""
        relocates = bool(recipe.moves or recipe.extracts)
        if recipe.supported and relocates:
            try:
                residual = build_repro(recipe).run()
                passed = residual == ""
            except Exception as exc:
                residual = f"reproduce raised {type(exc).__name__}: {exc}"
        results.append(
            GenResult(
                commit=commit,
                subject=subject,
                supported=recipe.supported and relocates,
                passed=passed,
                residual=residual,
                script=script,
                notes=recipe.notes,
            )
        )

    _write_log(out / "output.log", rev_range, results)
    _write_html(out / "output.html", rev_range, results)
    return results


def _write_log(path: Path, rev_range: str, results: list[GenResult]) -> None:
    n_pass = sum(1 for r in results if r.passed)
    lines = [
        f"reproduce-gen: {rev_range}",
        f"{len(results)} commit(s): {n_pass} reproduced, {len(results) - n_pass} not",
        "",
    ]
    for r in results:
        if r.passed:
            verdict = "PASS"
        elif not r.supported:
            verdict = "UNSUPPORTED (" + "; ".join(r.notes) + ")"
        else:
            verdict = f"RESIDUAL ({len(r.residual.splitlines())} lines)"
        lines.append(f"{r.commit[:9]}  {verdict}  {r.subject}")
    path.write_text("\n".join(lines) + "\n")


def _write_html(path: Path, rev_range: str, results: list[GenResult]) -> None:
    payload = {
        "title": rev_range,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [asdict(r) for r in results],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(rev_range)).replace(
            "__DATA_JSON__", data
        )
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>reproduce-gen __TITLE__</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d1d9e0;padding:10px 16px}
header h1{margin:0 0 4px;font-size:15px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:6px;font-size:12px}
.chip.ok{background:#e6ffec;border:1px solid #4ac26b}
.chip.no{background:#ffebe9;border:1px solid #ff8182}
main{padding:8px 16px 40px}
.card{border:1px solid #d1d9e0;border-left-width:4px;border-radius:6px;margin:8px 0}
.card.ok{border-left-color:#4ac26b}.card.no{border-left-color:#ff8182}.card.un{border-left-color:#d4a72c}
.hd{padding:6px 10px;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:8px}
.hd:hover{background:#f6f8fa}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:10px;flex:none}
.badge.ok{background:#e6ffec;border:1px solid #4ac26b;color:#1a7f37}
.badge.no{background:#ffebe9;border:1px solid #ff8182;color:#cf222e}
.badge.un{background:#fff8c5;border:1px solid #d4a72c;color:#7d4e00}
.hd code{color:#59636e;flex:none}
.bd{display:none;padding:6px 10px;border-top:1px solid #f0f1f3;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.card.open .bd{display:block}
.k{color:#59636e;margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:8px;border-radius:6px;overflow-x:auto}
.res .d{white-space:pre-wrap;word-break:break-word}
.res .add{background:#e6ffec}.res .del{background:#ffebe9}.res .h{background:#eff5ff;color:#0550ae}
</style></head>
<body>
<header><h1>reproduce-gen: <code>__TITLE__</code></h1>
<div><span id="counts"></span></div></header>
<main id="list"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function cls(r){ return r.passed?'ok':(r.supported?'no':'un'); }
function label(r){ return r.passed?'PASS':(r.supported?'RESIDUAL':'UNSUPPORTED'); }
let out='';
DATA.results.forEach(r => {
  const c = cls(r);
  out += '<div class="card '+c+'"><div class="hd"><span class="badge '+c+'">'+label(r)+'</span>';
  out += '<code>'+esc(r.commit.slice(0,9))+'</code><span>'+esc(r.subject)+'</span></div><div class="bd">';
  if(r.notes.length) out += '<div class="k">notes:</div><pre>'+esc(r.notes.join('\\n'))+'</pre>';
  out += '<div class="k">repro script:</div><pre>'+esc(r.script)+'</pre>';
  if(r.residual){
    out += '<div class="k">residual diff:</div><div class="res">';
    r.residual.split('\\n').forEach(l => {
      const k = l.startsWith('@@')?'h':(l[0]==='+'?'add':(l[0]==='-'?'del':''));
      out += '<div class="d '+k+'">'+esc(l)+'</div>';
    });
    out += '</div>';
  }
  out += '</div></div>';
});
document.getElementById('counts').innerHTML =
  '<span class="chip ok">'+DATA.passed+' reproduced</span><span class="chip no">'+(DATA.total-DATA.passed)+' not</span>';
document.getElementById('list').innerHTML = out;
document.getElementById('list').addEventListener('click', e => {
  const hd = e.target.closest('.hd'); if(hd) hd.parentNode.classList.toggle('open');
});
</script></body></html>"""


def _main(argv: list[str]) -> int:
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    match = None
    if "--match" in argv:
        i = argv.index("--match")
        match = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) != 1:
        print(
            "usage: python3 mechanical_refactor_generate_proof.py <commit>\n"
            "       python3 mechanical_refactor_generate_proof.py <base>..<tip> "
            "[--match REGEX] --out DIR",
            file=sys.stderr,
        )
        return 2
    target = argv[0]
    if ".." in target:
        assert out_dir, "--out DIR is required for a range"
        results = generate_range(target, match=match, out_dir=out_dir)
        n = sum(1 for r in results if r.passed)
        print(f"{n}/{len(results)} reproduced; folder: {out_dir}")
        return 0
    root = _repo_root()
    recipe = infer_recipe(target, root)
    print(
        recipe_to_script(
            recipe, _git_output(["log", "-1", "--format=%s", target], root)
        )
    )
    if recipe.supported and (recipe.moves or recipe.extracts):
        build_repro(recipe).run()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
