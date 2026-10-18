"""Layers that append stages are built inside an open layer stack."""

import ast
import unittest
from pathlib import Path

import sglang.srt.models
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PACKAGE = ("sglang", "srt", "models")
# Calls that build layers inside a stack of their own.
_STACK_BUILDERS = ("make_layers", "make_pp_layers")


def _call_name(func):
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _calls(node):
    return {
        _call_name(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)
    }


class _Classes:
    """The package's classes, resolved the way each file names them: its own
    class of that name, else the class its import names. A name resolved
    neither way stands for every class of that name, which fails closed."""

    def __init__(self, trees, root):
        self.by_name = {}
        modules = {}
        for path, tree in trees.items():
            parts = path.relative_to(root).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            modules[".".join(_PACKAGE + parts)] = path
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.by_name.setdefault(node.name, {})[path] = node
        self.imports = {}
        for path, tree in trees.items():
            package = path.relative_to(root).parent.parts
            table = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module
                if node.level:
                    up = node.level - 1
                    base = (
                        _PACKAGE + package[: len(package) - up]
                        if up <= len(package)
                        else None
                    )
                    module = base and ".".join(base + ((module,) if module else ()))
                for alias in node.names:
                    table[alias.asname or alias.name] = (
                        modules.get(module),
                        alias.name,
                    )
            self.imports[path] = table

    def resolve(self, path, name):
        defined = self.by_name.get(name, {})
        if path in defined:
            return {(path, name)}
        source, original = self.imports[path].get(name, (None, name))
        defined = self.by_name.get(original, {})
        if source in defined:
            return {(source, original)}
        return {(p, original) for p in defined}


def _stage_classes(trees, classes):
    """Classes whose construction appends stages: a method reaches
    append_stages, directly or through a module-level helper, or a base class
    does. Derived from the code, so a new model joins without being listed."""
    helpers, functions = {"append_stages"}, {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                functions.setdefault(node.name, []).append(node)
    changed = True
    while changed:
        changed = False
        for name, nodes in functions.items():
            if name not in helpers and any(_calls(fn) & helpers for fn in nodes):
                helpers.add(name)
                changed = True
    stage = set()
    changed = True
    while changed:
        changed = False
        for name, defs in classes.by_name.items():
            for path, node in defs.items():
                if (path, name) in stage:
                    continue
                bases = set()
                for base in node.bases:
                    bases |= classes.resolve(path, ast.unparse(base).split(".")[-1])
                if bases & stage or any(
                    isinstance(m, ast.FunctionDef) and _calls(m) & helpers
                    for m in node.body
                ):
                    stage.add((path, name))
                    changed = True
    return stage


class _Holders:
    """Which expressions can hold a stage class: the class itself, a
    collection of them, and the names they are bound to. Within a file that is
    a variable or attribute assigned one; across files it is a keyword or
    parameter given one, since a class handed to a shared model is called
    under the parameter's name in another file."""

    def __init__(self, trees, classes, stage):
        self.classes, self.stage = classes, stage
        self.names = {path: set() for path in trees}
        self.parameters = set()
        changed = True
        while changed:
            changed = False
            for path, tree in trees.items():
                for node in ast.walk(tree):
                    for target, value in self._bindings(node):
                        if value is None or not self.holds(path, value):
                            continue
                        names = self.parameters if target[0] else self.names[path]
                        if target[1] not in names:
                            names.add(target[1])
                            changed = True

    @staticmethod
    def _bindings(node):
        """(is_parameter, name) targets and the values bound to them."""
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in target.elts if isinstance(target, ast.Tuple) else [target]:
                    if isinstance(t, (ast.Name, ast.Attribute)):
                        yield (False, _call_name(t)), node.value
        elif isinstance(node, ast.keyword) and node.arg:
            yield (True, node.arg), node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
            for arg, default in zip(
                positional + args.kwonlyargs, defaults + args.kw_defaults
            ):
                yield (True, arg.arg), default

    def holds(self, path, expr):
        if isinstance(expr, ast.Name):
            return (
                expr.id in self.names[path]
                or expr.id in self.parameters
                or bool(self.classes.resolve(path, expr.id) & self.stage)
            )
        if isinstance(expr, ast.Attribute):
            return expr.attr in self.names[path]
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return any(self.holds(path, e) for e in expr.elts)
        if isinstance(expr, ast.Dict):
            return any(self.holds(path, v) for v in expr.values)
        if isinstance(expr, ast.Subscript):
            return self.holds(path, expr.value)
        if isinstance(expr, ast.IfExp):
            return self.holds(path, expr.body) or self.holds(path, expr.orelse)
        if isinstance(expr, ast.BoolOp):
            return any(self.holds(path, v) for v in expr.values)
        return False


def _opens_a_stack(node):
    if isinstance(node, ast.With):
        return any("layer_stack" in ast.unparse(i.context_expr) for i in node.items)
    return isinstance(node, ast.Call) and _call_name(node.func) in _STACK_BUILDERS


def _unstacked_constructions(trees, root):
    classes = _Classes(trees, root)
    stage = _stage_classes(trees, classes)
    holders = _Holders(trees, classes, stage)
    found = []
    for path, tree in trees.items():
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        stacked = [n for n in ast.walk(tree) if _opens_a_stack(n)]
        # A named factory used inside a stack (e.g. handed to make_layers).
        factories = {
            x.id for s in stacked for x in ast.walk(s) if isinstance(x, ast.Name)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and holders.holds(path, node.func)):
                continue
            chain, x = [], node
            while x in parents:
                x = parents[x]
                chain.append(x)
            if any(_opens_a_stack(c) for c in chain):
                continue
            enclosing = next(
                (c for c in chain if isinstance(c, (ast.FunctionDef, ast.Lambda))),
                None,
            )
            if (
                isinstance(enclosing, ast.FunctionDef)
                and enclosing.name != "__init__"
                and enclosing.name in factories
            ):
                continue
            found.append((str(path.relative_to(root)), node.lineno, node.func))
    return stage, [
        f"{p}:{line} {ast.unparse(func)}"
        for p, line, func in sorted(found, key=lambda f: f[:2])
    ]


class TestStageLayersBuiltInAStack(CustomTestCase):
    def test_every_stage_layer_is_built_inside_a_layer_stack(self):
        # append_stages raises outside a stack, but only when that model is
        # built: a draft or wrapper model nobody constructs in CI would fail
        # only in production. This checks every construction site instead.
        (root,) = map(Path, sglang.srt.models.__path__)
        trees = {p: ast.parse(p.read_text()) for p in sorted(root.rglob("*.py"))}
        stage, found = _unstacked_constructions(trees, root)
        self.assertGreater(len(stage), 40)
        self.assertEqual(found, [])

    def test_a_layer_built_outside_a_stack_is_reported(self):
        draft = (
            "def make_stage_boundary(norm):\n"
            "    return append_stages((decl, norm))\n"
            "class Layer:\n"
            "    def __init__(self):\n"
            "        self.boundary = make_stage_boundary(None)\n"
            "class DraftLayer(Layer):\n"
            "    pass\n"
            "class Draft:\n"
            "    def __init__(self):\n"
            "        self.decoder = DraftLayer()\n"
            "class Stacked:\n"
            "    def __init__(self):\n"
            "        with layer_stack():\n"
            "            self.decoder = DraftLayer()\n"
        )
        # A class of the same name that appends nothing, built by its own file.
        other = (
            "class Layer:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "class Model:\n"
            "    def __init__(self):\n"
            "        self.layer = Layer()\n"
        )
        # A class reached by a lookup, or handed to a shared model under a
        # parameter's name.
        lookup = (
            "from sglang.srt.models.draft import DraftLayer, Layer\n"
            "LAYERS = {'draft': DraftLayer}\n"
            "class Shared:\n"
            "    def __init__(self, layer_type=None):\n"
            "        self.layers = [layer_type() for _ in range(2)]\n"
            "class Variant(Shared):\n"
            "    def __init__(self):\n"
            "        super().__init__(layer_type=Layer)\n"
            "class Hybrid:\n"
            "    def __init__(self):\n"
            "        def get_layer(idx, prefix):\n"
            "            return LAYERS['draft']()\n"
            "        self.layers = make_layers(2, get_layer)\n"
            "        layer_class = LAYERS['draft']\n"
            "        self.extra = layer_class()\n"
        )
        root = Path("models")
        trees = {
            root / "draft.py": ast.parse(draft),
            root / "z.py": ast.parse(other),
            root / "lookup.py": ast.parse(lookup),
        }
        stage, found = _unstacked_constructions(trees, root)
        # Reached through a helper, inherited by a subclass, and not confused
        # with a same-named class elsewhere.
        self.assertEqual({name for _, name in stage}, {"Layer", "DraftLayer"})
        self.assertNotIn((root / "z.py", "Layer"), stage)
        self.assertEqual(
            found,
            [
                "draft.py:10 DraftLayer",
                "lookup.py:5 layer_type",
                "lookup.py:15 layer_class",
            ],
        )


if __name__ == "__main__":
    unittest.main()
