"""TOP kernel inventory and version-bump check.

    python python/sglang/srt/true_on_policy/kernel_inventory.py generate   # rewrite kernel_inventory.md
    python python/sglang/srt/true_on_policy/kernel_inventory.py check      # what a bump changed under TOP's kernels
    python python/sglang/srt/true_on_policy/kernel_inventory.py update     # record the reviewed fingerprints

The inventory is rendered from the registry rows (``registry.py``), the references in ``kernel_manifest.py`` and the
code they point to: "Stands in for" docstrings, gate comments (or a startup refusal's message), pinned call arguments
and pin constants. The check fingerprints every stock symbol the manifest names (normalized source of the definition
and the same-module helpers it references; for a compiled symbol, the shared object holding it; the package's installed
version and its pin in ``python/pyproject.toml``) and compares them with ``kernel_fingerprints.json``.

Everything is read statically with the standard library: nothing is imported, so it runs on a CPU host, without
GPUs or the packages installed. A package that is not installed is compared by its pin only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import re
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import NamedTuple, Optional

_HERE = Path(__file__).resolve().parent
PY_ROOT = _HERE.parents[2]
INVENTORY_PATH = _HERE / "kernel_inventory.md"
FINGERPRINTS_PATH = _HERE / "kernel_fingerprints.json"
_REGISTRY_PATH = _HERE / "registry.py"
_PYPROJECT_PATH = PY_ROOT / "pyproject.toml"

# Import name -> the distribution SGLang pins it as.
DISTRIBUTIONS = {
    "deep_ep": "sgl-deep-ep",
    "deep_gemm": "sgl-deep-gemm",
    "flashinfer": "flashinfer_python",
    "sgl_flash_mla": "sgl-flash-mla",
    "sgl_kernel": "sglang-kernel",
    "tilelang": "tilelang",
    "torch": "torch",
}
# Packages whose Python wrapper is also in this repo; read when the package is not installed.
_REPO_COPIES = {"sgl_kernel": PY_ROOT / "sglang" / "kernels" / "aot" / "python"}
_STANDS_IN_FOR = "Stands in for"


def _load_manifest():
    spec = importlib.util.spec_from_file_location(
        "_top_kernel_manifest", _HERE / "kernel_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = _load_manifest()


# ---------------------------------------------------------------- registry


class RegistryOp(NamedTuple):
    name: str
    comment: str
    in_trainer: bool


class RegistryFamily(NamedTuple):
    name: str
    model_types: tuple
    ops: tuple
    fp8_ops: Optional[tuple]
    vision_ops: Optional[tuple]


def _comment_above(lines: list, lineno: int) -> str:
    """The contiguous ``#`` block ending right above 1-based ``lineno``, as one line of text."""
    block = []
    index = lineno - 2
    while index >= 0 and lines[index].strip().startswith("#"):
        block.append(lines[index].strip()[1:].strip())
        index -= 1
    return " ".join(reversed(block))


def _eval_names(node: ast.AST, names: dict):
    """Evaluate the registry's tuple-of-strings expressions (literals, names, ``+``)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_names(element, names) for element in node.elts)
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_names(node.left, names) + _eval_names(node.right, names)
    raise ValueError(
        f"registry.py:{node.lineno}: cannot read {ast.unparse(node)!r} statically"
    )


def read_registry(path: Path = _REGISTRY_PATH) -> tuple:
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    names, ops, families = {}, [], []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target = statement.target
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id == "OPS":
            for call in statement.value.elts:
                trainer = [k for k in call.keywords if k.arg == "in_trainer"]
                ops.append(
                    RegistryOp(
                        name=call.args[0].value,
                        comment=_comment_above(lines, call.lineno),
                        in_trainer=not trainer or bool(trainer[0].value.value),
                    )
                )
        elif target.id == "FAMILIES":
            for call in statement.value.elts:
                fields = {k.arg: _eval_names(k.value, names) for k in call.keywords}
                families.append(
                    RegistryFamily(
                        name=fields["name"],
                        model_types=fields["model_types"],
                        ops=fields["ops"],
                        fp8_ops=fields.get("fp8_ops"),
                        vision_ops=fields.get("vision_ops"),
                    )
                )
        else:
            try:
                names[target.id] = _eval_names(statement.value, names)
            except (ValueError, KeyError):
                pass
    return tuple(ops), tuple(families)


# ---------------------------------------------------------------- static symbol resolution


class Symbol(NamedTuple):
    ref: str
    # "repo" (this checkout), "installed", "repo-copy" (package absent, wrapper read from this repo),
    # "compiled" (found only in a shared object), "absent" (package not installed), "missing".
    status: str
    path: Optional[Path] = None
    tree: Optional[ast.Module] = None
    lines: tuple = ()
    # The definition or assignment; None for a module reference.
    node: Optional[ast.AST] = None
    detail: str = ""


_PARSED: dict = {}


def _parse(path: Path):
    if path not in _PARSED:
        source = path.read_text(errors="replace")
        _PARSED[path] = (ast.parse(source), tuple(source.splitlines()))
    return _PARSED[path]


def _package_base(top: str):
    """(directory holding package ``top``, origin) without importing it."""
    if top == "sglang":
        return PY_ROOT, "repo"
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]).parent, "installed"
    if spec is not None and spec.origin and spec.origin.endswith(".py"):
        return Path(spec.origin).parent, "installed"
    if top in _REPO_COPIES:
        return _REPO_COPIES[top], "repo-copy"
    return None, "absent"


def _module_paths(dotted: str):
    """(source file or None, compiled files, origin) of module ``dotted``."""
    parts = dotted.split(".")
    base, origin = _package_base(parts[0])
    if base is None:
        return None, [], origin
    stem = base.joinpath(*parts)
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate, [], origin
    compiled = sorted(stem.parent.glob(stem.name + ".*so")) + sorted(
        stem.parent.glob(stem.name + ".*pyd")
    )
    return None, compiled, origin


def _module_level(body):
    """Module statements, looking inside top-level if/try blocks (platform branches, optional imports)."""
    for statement in body:
        if isinstance(statement, ast.If):
            yield from _module_level(statement.body)
            yield from _module_level(statement.orelse)
        elif isinstance(statement, ast.Try):
            for block in (
                statement.body,
                *(h.body for h in statement.handlers),
                statement.orelse,
            ):
                yield from _module_level(block)
        else:
            yield statement


def _defines(statement, name: str) -> bool:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return statement.name == name
    if isinstance(statement, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return isinstance(statement.target, ast.Name) and statement.target.id == name
    return False


def _descend(node, parts):
    for name in parts:
        body = node.body if isinstance(node, ast.ClassDef) else []
        found = [s for s in _module_level(body) if _defines(s, name)]
        if not found:
            return None
        node = found[0]
    return node


def _import_target(dotted: str, path: Path, statement: ast.ImportFrom) -> str:
    if not statement.level:
        return statement.module
    package = (
        dotted.split(".") if path.name == "__init__.py" else dotted.split(".")[:-1]
    )
    package = package[: len(package) - (statement.level - 1)]
    return ".".join(package + ([statement.module] if statement.module else []))


def _compiled_holder(files, name: str):
    needle = name.encode()
    for file in files:
        if needle in file.read_bytes():
            return file
    return None


def resolve(ref: str, depth: int = 0) -> Symbol:
    module, _, qualname = ref.partition(":")
    parts = [p for p in qualname.split(".") if p]
    path, compiled, origin = _module_paths(module)
    if origin == "absent":
        return Symbol(ref=ref, status="absent")
    if path is None:
        holder = _compiled_holder(compiled, parts[-1]) if parts and compiled else None
        if holder is not None:
            return Symbol(ref=ref, status="compiled", path=holder)
        return Symbol(ref=ref, status="missing", detail=f"no module {module}")
    tree, lines = _parse(path)
    if not parts:
        return Symbol(ref=ref, status=origin, path=path, tree=tree, lines=lines)
    statements = list(_module_level(tree.body))
    for statement in statements:
        if _defines(statement, parts[0]):
            node = _descend(statement, parts[1:])
            if node is None:
                break
            return Symbol(
                ref=ref, status=origin, path=path, tree=tree, lines=lines, node=node
            )
    if depth < 6:
        for statement in statements:
            if not isinstance(statement, ast.ImportFrom):
                continue
            for alias in statement.names:
                if (alias.asname or alias.name) != parts[0] or alias.name == "*":
                    continue
                target = _import_target(module, path, statement)
                rest = ".".join(parts[1:])
                for candidate in (
                    f"{target}:{alias.name}{'.' + rest if rest else ''}",
                    f"{target}.{alias.name}:{rest}",
                ):
                    found = resolve(candidate, depth + 1)
                    if found.status != "missing":
                        return found._replace(ref=ref)
        for statement in statements:
            if isinstance(statement, ast.ImportFrom) and statement.names[0].name == "*":
                target = _import_target(module, path, statement)
                found = resolve(f"{target}:{qualname}", depth + 1)
                if found.status != "missing":
                    return found._replace(ref=ref)
    if origin == "installed" and depth == 0:
        holder = _compiled_holder(sorted(path.parent.rglob("*.so")), parts[-1])
        if holder is not None:
            return Symbol(ref=ref, status="compiled", path=holder)
    return Symbol(
        ref=ref,
        status="missing",
        path=path,
        detail=f"{qualname} not defined in {path.name}",
    )


# ---------------------------------------------------------------- fingerprints


def _segment(symbol: Symbol, node: ast.AST) -> str:
    definition = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    decorators = node.decorator_list if isinstance(node, definition) else []
    start = min([node.lineno] + [d.lineno for d in decorators])
    return textwrap.dedent("\n".join(symbol.lines[start - 1 : node.end_lineno]))


def _normalized(source: str) -> str:
    """The source without comments, blank lines or whitespace runs, so a comment edit is not a change.

    Comments are cut by token position and the rest stays text: Python 3.12 splits f-strings into several tokens,
    so token strings would hash differently across interpreter versions.
    """
    lines = source.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return re.sub(
        r"\s+", " ", "\n".join(line for line in lines if line.strip())
    ).strip()


def _local_closure(symbol: Symbol) -> list:
    """The definition plus the module-level definitions it references, transitively (jit kernels, helpers)."""
    top = {}
    for statement in _module_level(symbol.tree.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.setdefault(statement.name, statement)
    seen, queue, nodes = set(), [symbol.node], []
    while queue:
        node = queue.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in top:
                queue.append(top[child.id])
    return sorted(nodes, key=lambda n: n.lineno)


def _params(node) -> Optional[list]:
    if isinstance(node, ast.ClassDef):
        init = _descend(node, ["__init__"])
        node = init if init is not None else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def read_pins(path: Path = _PYPROJECT_PATH) -> dict:
    pins = {}
    for name, version in re.findall(
        r'^\s*"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^";\s]+)', path.read_text(), re.M
    ):
        pins.setdefault(name, version)
    return pins


def _installed_version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint(ref: str, pins: dict) -> dict:
    symbol = resolve(ref)
    top = ref.split(".")[0].split(":")[0]
    dist = DISTRIBUTIONS.get(top)
    record = {"status": symbol.status}
    if dist is not None:
        record.update(dist=dist, pin=pins.get(dist), installed=_installed_version(dist))
    if symbol.status == "compiled":
        record["shared_object"] = symbol.path.name
        record["hash"] = hashlib.sha256(symbol.path.read_bytes()).hexdigest()
    elif symbol.status not in ("absent", "missing"):
        if symbol.node is None:
            text = "\n".join(symbol.lines)
        else:
            text = "\n\n".join(
                _segment(symbol, node) for node in _local_closure(symbol)
            )
        record["hash"] = hashlib.sha256(_normalized(text).encode()).hexdigest()
        record["params"] = _params(symbol.node)
    return record


# ---------------------------------------------------------------- reasons and pins at the code


def first_paragraph(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split("\n\n")[0].split())


def top_kernel_reason(kernel: str) -> str:
    symbol = resolve(kernel)
    node = symbol.tree if symbol.node is None else symbol.node
    return first_paragraph(ast.get_docstring(node)) if node is not None else ""


def _line_comment(line: str) -> str:
    try:
        tokens = list(
            tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline)
        )
    except (tokenize.TokenError, IndentationError):
        return ""
    return " ".join(t.string[1:].strip() for t in tokens if t.type == tokenize.COMMENT)


def _enclosing_statement(node: ast.AST, lineno: int) -> ast.stmt:
    """The innermost statement under ``node`` whose lines contain ``lineno``."""
    best = node
    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and child.lineno <= lineno <= child.end_lineno:
            if child.end_lineno - child.lineno <= best.end_lineno - best.lineno:
                best = child
    return best


_BLOCKS = (ast.If, ast.For, ast.While, ast.With, ast.Try)
# A simple statement's comment covers an anchor this many lines into it, not a whole long call or literal.
_STATEMENT_HEAD_LINES = 3


def _locate_gate(gate) -> tuple:
    """(symbol, anchored line number, problem)."""
    symbol = resolve(gate.at)
    if symbol.node is None:
        problem = f"gate site {gate.at} not found ({symbol.status} {symbol.detail})"
        return symbol, 0, problem.strip()
    span = range(symbol.node.lineno, symbol.node.end_lineno + 1)
    hits = [n for n in span if gate.anchor in symbol.lines[n - 1]]
    if len(hits) != 1:
        return (
            symbol,
            0,
            f"gate anchor {gate.anchor!r} matches {len(hits)} lines in {gate.at}",
        )
    return symbol, hits[0], ""


def _in_header(statement: ast.stmt, lineno: int) -> bool:
    if isinstance(statement, _BLOCKS):
        return lineno < statement.body[0].lineno
    return lineno - statement.lineno < _STATEMENT_HEAD_LINES


def gate_reason(gate) -> tuple:
    """(reason, problem): the comment on the anchored line, above it, above its statement, or opening its block."""
    symbol, lineno, problem = _locate_gate(gate)
    if problem:
        return "", problem
    lines = list(symbol.lines)
    statement = _enclosing_statement(symbol.node, lineno)
    candidates = [_line_comment(lines[lineno - 1]), _comment_above(lines, lineno)]
    if statement is not symbol.node and _in_header(statement, lineno):
        candidates.append(_comment_above(lines, statement.lineno))
        if isinstance(statement, _BLOCKS):
            candidates.append(_comment_above(lines, statement.body[0].lineno))
    return next((c for c in candidates if c), ""), ""


def _message(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def gate_refusal(gate) -> str:
    """The refusal message when the gate refuses at startup instead of switching kernels, else "".

    Either the anchor sits in the condition of an ``if`` whose body raises, or in a ``("feature", flag)`` pair the
    function refuses by name.
    """
    symbol, lineno, problem = _locate_gate(gate)
    if problem:
        return ""
    statement = _enclosing_statement(symbol.node, lineno)
    if isinstance(statement, ast.If) and _in_header(statement, lineno):
        for child in statement.body:
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                if child.exc.args:
                    return _message(child.exc.args[0])
    if not any(isinstance(n, ast.Raise) for n in ast.walk(symbol.node)):
        return ""
    pairs = [
        n
        for n in ast.walk(statement)
        if isinstance(n, ast.Tuple)
        and n.lineno <= lineno <= n.end_lineno
        and n.elts
        and isinstance(n.elts[0], ast.Constant)
        and isinstance(n.elts[0].value, str)
    ]
    if pairs:
        return min(pairs, key=lambda n: n.end_lineno - n.lineno).elts[0].value
    return ""


def _matches(call: ast.Call, spellings) -> bool:
    text = ast.unparse(call.func)
    return any(text == s or text.endswith("." + s) for s in spellings)


def pinned_calls(stock) -> tuple:
    """(rendered calls at the pin site, keywords passed to the kernel, problem)."""
    if not stock.pinned_at:
        return (), set(), ""
    site = resolve(stock.pinned_at)
    if site.node is None:
        return (), set(), f"pin site {stock.pinned_at} not found ({site.status})"
    spellings = stock.calls or (stock.kernel.split(":")[1].split(".")[-1],)
    calls = [
        c
        for c in ast.walk(site.node)
        if isinstance(c, ast.Call) and _matches(c, spellings)
    ]
    if not calls:
        return (
            (),
            set(),
            f"pin site {stock.pinned_at} no longer calls {' / '.join(spellings)}",
        )
    kernel_calls = [
        c
        for c in calls
        if _matches(c, spellings[:1]) or ast.unparse(c.func).endswith(".update")
    ]
    keywords = {k.arg for c in kernel_calls for k in c.keywords if k.arg}
    rendered = tuple(
        dict.fromkeys(
            " ".join(ast.unparse(c).split())
            for c in sorted(calls, key=lambda c: c.lineno)
        )
    )
    return rendered, keywords, ""


def pin_constant(ref: str) -> tuple:
    symbol = resolve(ref)
    if symbol.node is None:
        return "", f"pin constant {ref} not found ({symbol.status})"
    return " ".join(ast.unparse(symbol.node).split()), ""


# ---------------------------------------------------------------- the inventory


def stock_refs() -> dict:
    """Every stock symbol the manifest names -> the (op, role) pairs that name it."""
    refs = {}

    def add(ref, op, role):
        refs.setdefault(ref, []).append((op, role))

    for op, entry in MANIFEST.OPS.items():
        for kernel in entry.kernels:
            if isinstance(kernel, MANIFEST.Stock):
                add(kernel.kernel, op, "stock")
            else:
                for target in kernel.stands_in_for:
                    add(target, op, f"stood in for by {kernel.kernel}")
        for gate in entry.gates:
            add(gate.fast_path, op, f"fast path off at {gate.at}")
    for gate in MANIFEST.CONTRACT_GATES:
        add(gate.fast_path, "(contract)", f"fast path off at {gate.at}")
    return refs


def problems() -> list:
    """References that no longer resolve, and registry ops the manifest does not cover (or names wrongly)."""
    ops, _ = read_registry()
    names = [op.name for op in ops]
    found = [
        f"op {n} has no kernel_manifest entry" for n in names if n not in MANIFEST.OPS
    ]
    found += [
        f"kernel_manifest names unknown op {n}" for n in MANIFEST.OPS if n not in names
    ]
    for ref in stock_refs():
        symbol = resolve(ref)
        if symbol.status == "missing":
            found.append(f"stock symbol {ref} does not exist: {symbol.detail}")
    for op, entry in MANIFEST.OPS.items():
        for kernel in entry.kernels:
            if isinstance(kernel, MANIFEST.TopKernel):
                if resolve(kernel.kernel).status in ("missing", "absent"):
                    found.append(f"{op}: TOP kernel {kernel.kernel} does not exist")
                continue
            found += [
                f"{op}: {p}"
                for p in [pinned_calls(kernel)[2]]
                + [pin_constant(r)[1] for r in kernel.pins]
                if p
            ]
        for gate in entry.gates:
            problem = gate_reason(gate)[1]
            if problem:
                found.append(f"{op}: {problem}")
        if entry.admitted_by and resolve(entry.admitted_by).node is None:
            found.append(f"{op}: admitting function {entry.admitted_by} not found")
    found += [p for p in (gate_reason(g)[1] for g in MANIFEST.CONTRACT_GATES) if p]
    for family, modules in MANIFEST.FAMILY_PROGRAMS.items():
        found += [
            f"{family}: program module {m} not found"
            for m in modules
            if resolve(m + ":").path is None
        ]
    return found


def signature_problems() -> list:
    """Keywords a pin site passes that the stock kernel no longer accepts."""
    found = []
    for op, entry in MANIFEST.OPS.items():
        for kernel in entry.kernels:
            if not isinstance(kernel, MANIFEST.Stock):
                continue
            keywords = pinned_calls(kernel)[1]
            params = _params(resolve(kernel.kernel).node)
            if (
                not keywords
                or params is None
                or any(p.startswith("**") for p in params)
            ):
                continue
            gone = sorted(keywords - set(params))
            if gone:
                found.append(
                    f"{op}: {kernel.kernel} no longer takes pinned {', '.join(gone)} (pinned at {kernel.pinned_at})"
                )
    return found


def _source_label(ref: str, pins: dict) -> str:
    top = ref.split(".")[0].split(":")[0]
    if top == "sglang":
        return "SGLang"
    dist = DISTRIBUTIONS.get(top, top)
    return f"{dist}=={pins[dist]}" if dist in pins else dist


def _render_kernel(kernel, pins: dict, unclassified: list, op: str) -> list:
    if isinstance(kernel, MANIFEST.TopKernel):
        reason = top_kernel_reason(kernel.kernel)
        targets = ", ".join(f"`{t}`" for t in kernel.stands_in_for)
        if not reason.startswith(_STANDS_IN_FOR):
            unclassified.append(
                f'`{op}`: TOP kernel `{kernel.kernel}` has no "Stands in for" docstring'
            )
            reason = f'**no "Stands in for" reason in its docstring** ({reason or "no docstring"})'
        return [f"- TOP `{kernel.kernel}`, standing in for {targets}: {reason}"]
    out = [f"- stock `{kernel.kernel}` ({_source_label(kernel.kernel, pins)})"]
    calls, _, _ = pinned_calls(kernel)
    if calls:
        out.append(
            f"  - pinned at `{kernel.pinned_at}`: " + "; ".join(f"`{c}`" for c in calls)
        )
    for ref in kernel.pins:
        out.append(f"  - pin `{ref}`: `{pin_constant(ref)[0]}`")
    return out


def _render_gate(gate, unclassified: list, op: str) -> str:
    reason = gate_reason(gate)[0]
    if reason:
        return f"- `{gate.fast_path}` off at `{gate.at}` (`{gate.anchor}`): {reason}"
    refusal = gate_refusal(gate)
    if refusal:
        return f"- `{gate.fast_path}` refused at `{gate.at}` (`{gate.anchor}`), refusal: {refusal}"
    unclassified.append(
        f"`{op}`: gate at `{gate.at}` (`{gate.anchor}`) states no reason"
    )
    return f"- `{gate.fast_path}` off at `{gate.at}` (`{gate.anchor}`): **no reason comment at the gate**"


def render() -> str:
    ops, families = read_registry()
    pins = read_pins()
    unclassified = []
    out = [
        "# TOP kernel inventory",
        "",
        "Generated by `python python/sglang/srt/true_on_policy/kernel_inventory.py generate`; do not edit. Ops and",
        "their programs come from `registry.py`, kernel references from `kernel_manifest.py`, reasons from the code",
        "they point to. `kernel_inventory.py check` lists what a version bump changed under these kernels.",
        "",
        "## Families",
        "",
        "| Family | model_type | ops | FP8 ops | vision ops | serving program modules |",
        "|---|---|---|---|---|---|",
    ]
    for family in families:
        modules = ", ".join(
            f"`{m.rsplit('.', 1)[1]}`"
            for m in MANIFEST.FAMILY_PROGRAMS.get(family.name, ())
        )
        fp8 = (
            "same"
            if family.fp8_ops == family.ops
            else ", ".join(family.fp8_ops or ()) or "-"
        )
        out.append(
            f"| {family.name} | {', '.join(family.model_types)} | {', '.join(family.ops)} | {fp8} | "
            f"{', '.join(family.vision_ops or ()) or '-'} | {modules} |"
        )
    out += ["", "## Family program modules", ""]
    for module in sorted({m for ms in MANIFEST.FAMILY_PROGRAMS.values() for m in ms}):
        out.append(
            f"- `{module}`: {first_paragraph(ast.get_docstring(resolve(module + ':').tree)) or '-'}"
        )
    out += ["", "## Ops", ""]
    for op in ops:
        entry = MANIFEST.OPS.get(op.name, MANIFEST.OpKernels())
        users = [
            f.name
            for f in families
            if op.name in f.ops + (f.fp8_ops or ()) + (f.vision_ops or ())
        ]
        out += [f"### {op.name}", "", op.comment or "**no registry row comment**", ""]
        out.append(
            f"Families: {', '.join(users) or '-'}."
            + ("" if op.in_trainer else " Serving only.")
        )
        out.append("")
        if entry.admitted_by:
            doc = first_paragraph(ast.get_docstring(resolve(entry.admitted_by).node))
            out += [
                f"Admitted by `{entry.admitted_by}`" + (f": {doc}" if doc else "."),
                "",
            ]
        if entry.kernels:
            out += ["Kernels that decide its bits:", ""]
            for kernel in entry.kernels:
                out += _render_kernel(kernel, pins, unclassified, op.name)
            out.append("")
        if entry.gates:
            out += ["Stock fast paths TOP turns off:", ""]
            out += [_render_gate(gate, unclassified, op.name) for gate in entry.gates]
            out.append("")
    out += ["## Contract-wide gates", ""]
    out += [
        _render_gate(gate, unclassified, "(contract)")
        for gate in MANIFEST.CONTRACT_GATES
    ]
    out += ["", "## Not classified: reason missing in code", ""]
    out += [f"- {item}" for item in dict.fromkeys(unclassified)] or ["None."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- bump check


def current_fingerprints() -> dict:
    pins = read_pins()
    used = {
        DISTRIBUTIONS[t]
        for t in {r.split(".")[0].split(":")[0] for r in stock_refs()}
        if t in DISTRIBUTIONS
    }
    return {
        "pins": {d: pins.get(d) for d in sorted(used)},
        "symbols": {ref: fingerprint(ref, pins) for ref in sorted(stock_refs())},
    }


def _compare(old: Optional[dict], new: dict) -> tuple:
    """(changes, note): what differs, and whether the sources could be compared at all."""
    if old is None:
        return [], "new reference; run update"
    if new["status"] == "missing":
        return ["symbol no longer exists"], ""
    changes = []
    for field, label in (("pin", "pin"), ("installed", "installed version")):
        if old.get(field) != new.get(field) and (
            field == "pin" or (old.get(field) and new.get(field))
        ):
            changes.append(f"{label} {old.get(field)} -> {new.get(field)}")
    if "hash" in old and "hash" in new:
        if old["hash"] != new["hash"]:
            changes.append("source changed")
        if old.get("params") != new.get("params"):
            changes.append(f"signature {old.get('params')} -> {new.get('params')}")
        return changes, ""
    return changes, f"source not compared ({old['status']} then, {new['status']} now)"


def check(stored: dict, current: dict) -> tuple:
    """(report lines, changed?)."""
    refs = stock_refs()
    sections = {"stock": [], "stood": [], "fast": []}
    notes = []
    for ref, new in current["symbols"].items():
        changes, note = _compare(stored.get("symbols", {}).get(ref), new)
        if note:
            notes.append(f"- `{ref}`: {note}")
        if not changes:
            continue
        for op, role in refs[ref]:
            key = (
                "stock"
                if role == "stock"
                else "stood"
                if role.startswith("stood")
                else "fast"
            )
            sections[key].append(f"- `{op}`: `{ref}` ({role}): {'; '.join(changes)}")
    signatures = [f"- {p}" for p in signature_problems()]
    missing = [f"- {p}" for p in problems()]
    report = []
    for title, lines in (
        ("Unresolved references (fix the manifest or the code)", missing),
        (
            "Stock kernels changed: re-check each op's pins and re-gate its families",
            sections["stock"],
        ),
        ("Pinned options the stock kernel no longer accepts", signatures),
        (
            "Stand-in targets changed: candidates to retire a TOP kernel",
            sections["stood"],
        ),
        (
            "Disabled fast paths changed: re-check whether the gate is still needed",
            sections["fast"],
        ),
    ):
        report += [f"## {title}", ""] + (sorted(set(lines)) or ["None."]) + [""]
    report += ["## Not compared", ""] + (notes or ["None."])
    changed = bool(missing or signatures or any(sections.values()))
    return report, changed


def _merge_for_update(stored: dict, current: dict) -> dict:
    """Keep a recorded source fingerprint when the package is absent here at the same pin: absence is not news."""
    merged = json.loads(json.dumps(current))
    for ref, new in merged["symbols"].items():
        old = stored.get("symbols", {}).get(ref)
        if (
            old
            and new["status"] == "absent"
            and "hash" in old
            and old.get("pin") == new.get("pin")
        ):
            merged["symbols"][ref] = old
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("generate", "check", "update"))
    args = parser.parse_args(argv)
    found = problems()
    if args.command == "generate":
        if found:
            print("\n".join(["Unresolved references:"] + found), file=sys.stderr)
            return 1
        INVENTORY_PATH.write_text(render())
        print(f"wrote {INVENTORY_PATH}")
        return 0
    stored = (
        json.loads(FINGERPRINTS_PATH.read_text()) if FINGERPRINTS_PATH.exists() else {}
    )
    current = current_fingerprints()
    if args.command == "update":
        if found:
            print("\n".join(["Unresolved references:"] + found), file=sys.stderr)
            return 1
        FINGERPRINTS_PATH.write_text(
            json.dumps(_merge_for_update(stored, current), indent=1, sort_keys=True)
            + "\n"
        )
        print(f"wrote {FINGERPRINTS_PATH}")
        return 0
    report, changed = check(stored, current)
    print("\n".join(report))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
