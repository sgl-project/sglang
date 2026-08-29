"""Resolution does not read the config bags, because they do not exist yet.

The bags are projected from what resolution decides, so anything the pipeline
calls has to read the resolving state instead — `resolved_view(server_args)`,
or the view a handler already holds. A bag read reached from resolution raises
`config namespace ... not published`, and only on the branch that reaches it:
the diffusion-LM page-size pass needed one model family, the Marlin LoRA
validation needed one MoE runner backend. Both were written, merged into a
branch, and stayed green for everything except the configuration that triggers
them.

`test_publish_precedes_bag_reads.py` is the same worry from the other side, but
it walks the *process entries* — it cannot see a helper the pipeline calls, and
neither of the two above appeared in it.

The walk starts from three places: the symbols the pipeline imports, the live
resolution registries (every pass and override provider, taken from the
registries themselves rather than from the decorator that put it there -- most
providers register through a helper call), and the passes named at a
`run_post_process_pass(sa, fn)` call site. From there it follows calls
in-module, one hop out, and matches an accessor whether it is spelled bare or
through an object.

What this still cannot see: a bag read reached through a method rather than a
module-level function, one behind an import the walk does not follow, and one
in a callable that reaches the pipeline through a variable no call site names.
It is a ratchet, not a proof.
"""

import ast
import functools
import inspect
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"


def _accessor_names():
    """Every bag accessor `runtime_context` exports, read from the module.

    Listing them by hand is how this went stale once already: the list had
    eighteen names while the module exported twenty-five, so a resolution-time
    `get_flags().x` or `get_resources().y` would have walked straight past.
    """
    tree = ast.parse((_SRT / "runtime_context.py").read_text(encoding="utf-8-sig"))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
    }
    # The context object itself is not a bag: it exists before anything is
    # published, and `declare_late_resolution` calls it deliberately to find
    # out whether the record it was handed has been published yet.
    return frozenset(names - {"get_context"})


_BAG_ACCESSORS = _accessor_names()

# `get_device` also names the device-string utility and the platform method,
# so only the bare spelling is the accessor.
_ATTRIBUTE_SPELLED = _BAG_ACCESSORS - {"get_device"}

# The pipeline itself and the mechanism it publishes through: `runtime_context`
# defines the accessors, and `arg_groups` is the pipeline's own code.
_OWN = ("server_args.py", "runtime_context.py")


def _pipeline_sources():
    """The record plus every module under `arg_groups/`.

    A handler that moved out of the record takes its imports with it, so
    seeding the walk from two files would stop covering it.
    """
    return [_SRT / "server_args.py", *sorted((_SRT / "arg_groups").rglob("*.py"))]


def _module_of(name):
    """`sglang.srt.a.b` -> the file, if it is one of ours."""
    if not name or not name.startswith("sglang.srt."):
        return None
    rel = name[len("sglang.srt.") :].replace(".", "/")
    for candidate in (_SRT / f"{rel}.py", _SRT / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imported_symbols(paths):
    """{module file: {symbol names imported from it}} across the given sources."""
    out = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _module_of(node.module)
            if target is None or target.name in _OWN:
                continue
            out.setdefault(target, set()).update(alias.name for alias in node.names)
    return out


def _registry_functions():
    """Every callable the resolution registries will call, from the registries.

    Not from decorator syntax: most model-override providers register through
    a `_register_for(...)` helper rather than a decorator, so a scan keyed on
    the decorator name walked past all of them -- 39 entries found where the
    registries hold 65. However a provider registers, it is in the registry
    once its module is imported, and `inspect` says where it came from.
    """
    from sglang.srt.arg_groups import overrides

    functions = list(overrides.POST_PROCESS_PASSES)
    functions += [fn for fns in overrides._MODEL_OVERRIDE_FNS.values() for fn in fns]
    functions += [fn for _predicate, fn in overrides._PREDICATE_OVERRIDE_FNS]
    return functions


@functools.lru_cache(maxsize=None)
def _registered_entries():
    """Entries the import map cannot reach: passes and override providers.

    A pass arrives at the pipeline as a value, and the registry calls its
    providers by dictionary lookup. Both run during resolution, so a bag read
    inside one raises exactly like a bag read in a handler -- and neither is
    named by an import the walk can follow.
    """
    entries = set()
    for fn in _registry_functions():
        target = inspect.unwrap(fn)
        name = getattr(target, "__name__", "")
        if not name or name == "<lambda>":
            continue
        source = inspect.getsourcefile(target)
        if source is None:
            continue
        path = pathlib.Path(source).resolve()
        if _SRT in path.parents:
            entries.add((path, name))
    # A pass handed over by value is in no registry, so its call sites are read
    # from the source. The entry carries the *defining* file: `_reaches_a_bag`
    # walks functions in the entry's file, so a call-site key walks nothing.
    by_value = set()
    sources = {
        path: path.read_text(encoding="utf-8-sig")
        for path in sorted(_SRT.rglob("*.py"))
    }
    for path, source in sources.items():
        if "run_post_process_pass" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_post_process_pass"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Name)
            ):
                by_value.add(node.args[1].id)
    trees = {}
    for path, source in sources.items():
        if not any(name in source for name in by_value):
            continue
        try:
            trees[path] = ast.parse(source)
        except SyntaxError:
            continue
    for name in sorted(by_value):
        defined_in = [
            path
            for path, tree in trees.items()
            if any(
                isinstance(node, ast.FunctionDef) and node.name == name
                for node in tree.body
            )
        ]
        if not defined_in:
            raise AssertionError(
                f"pass {name!r} is handed to run_post_process_pass by value "
                "but defined in no scanned module; the walk cannot see it"
            )
        for path in defined_in:
            entries.add((path, name))
    return entries


@functools.lru_cache(maxsize=None)
def _functions_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _locally_shadowed_accessors(path):
    """Accessor names this file imports from somewhere that is not the context.

    `get_device` is both the `device` bag accessor and the hardware probe in
    `utils.common`. Matching the bare name would report the probe as a bag read,
    so a name imported from elsewhere in this file is not the accessor.
    """
    shadowed = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("runtime_context"):
                continue
            for alias in node.names:
                name = alias.asname or alias.name
                if name in _BAG_ACCESSORS:
                    shadowed.add(name)
    return shadowed


def _reaches_a_bag(path, entry):
    """Does `entry` in `path` reach a bag accessor, following calls in-module?"""
    functions = _functions_in(path)
    shadowed = _locally_shadowed_accessors(path)
    seen = set()

    def walk(name):
        if name in seen or name not in functions:
            return None
        seen.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                # `rc.get_exec()`, `self.get_schedule()`: the same accessor
                # reached through a module alias or an object.
                if node.func.attr in _ATTRIBUTE_SPELLED:
                    return node.lineno
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id in _BAG_ACCESSORS and node.func.id not in shadowed:
                return node.lineno
            found = walk(node.func.id)
            if found is not None:
                return found
        return None

    return walk(entry)


class TestResolutionReadsNoBag(CustomTestCase):
    def test_the_accessor_set_is_derived_and_whole(self):
        """A shrunken accessor set would make every other check pass quietly."""
        self.assertGreaterEqual(
            len(_BAG_ACCESSORS),
            15,
            f"only {len(_BAG_ACCESSORS)} accessors were derived from "
            "runtime_context; the derivation broke",
        )
        # Spelled out so a rename that drops one fails here.
        for name in ("get_exec", "get_flags", "get_parallel", "get_resources"):
            self.assertIn(name, _BAG_ACCESSORS)

    def test_the_walk_finds_something_to_walk(self):
        """A collapsed import map would make the pin vacuous."""
        imported = _imported_symbols(_pipeline_sources())
        self.assertGreater(
            len(imported),
            20,
            f"the pipeline only imports from {len(imported)} of our modules; "
            "the scan broke",
        )

    def test_the_registered_entries_are_found(self):
        """The passes and providers are the half the import map cannot see."""
        entries = _registered_entries()
        self.assertGreater(
            len(entries),
            60,
            f"only {len(entries)} passes and providers were found; the scan broke",
        )
        # Every registered callable that lives in our tree has to appear: the
        # derivation reads the registries through `inspect`, so an interpreter
        # that imported a *different* checkout would resolve them outside
        # `_SRT` and quietly leave the walk with nothing to walk.
        missing = sorted(
            name
            for name in (
                getattr(inspect.unwrap(fn), "__name__", "")
                for fn in _registry_functions()
            )
            if name
            and name != "<lambda>"
            and name not in {entry for _path, entry in entries}
        )
        self.assertEqual(
            missing,
            [],
            "a registered pass or provider did not resolve to a file under "
            f"{_SRT}; the entry set is narrower than the registries:\n  "
            + "\n  ".join(missing),
        )

    def test_nothing_the_pipeline_calls_reads_a_bag(self):
        imported = _imported_symbols(_pipeline_sources())
        reachable = {
            (path, symbol) for path, symbols in imported.items() for symbol in symbols
        } | _registered_entries()
        found = []
        for path, symbol in sorted(reachable):
            line = _reaches_a_bag(path, symbol)
            if line is not None:
                found.append(
                    f"{path.relative_to(_SRT)}:{line} reached from "
                    f"{symbol}(), which resolution calls"
                )
        self.assertEqual(
            found,
            [],
            "resolution reaches a config-bag read, which raises on whichever "
            "branch gets there first; read the resolving state instead:\n  "
            + "\n  ".join(found),
        )


if __name__ == "__main__":
    unittest.main()
