"""A process entry publishes before it reads a config namespace.

The functions checked here are found by walking the package for `publish`
calls, not by listing them: a hand-kept list can name a function that no longer
exists and still pass, which is how the Ray actor entry went unchecked.

Every such function starts a process (or is the first thing a spawned worker
runs), so the runtime context it inherits is empty. A bag read placed above the
publish raises `config namespace ... not published` -- in a spawned worker,
which no unit test starts, so the failure only shows up as a server that never
comes up.

A process entry reaches its bag reads through what it calls -- `Scheduler(...)`,
`configure_scheduler_process(...)`, `self.init_tokenizer_and_processor()` -- not
by naming an accessor itself, so a scan of the entry's own body sees nothing to
order and passes whatever the code does. The read line is therefore taken over
what the entry calls: a call resolved inside the module, through a parameter's
default (`detokenizer_manager_class=DetokenizerManager`), or one hop out through
that module's import table, followed the same way at every depth. Following the
import table only out of the entry's own body would stop one call short of the
expert-backup read, which is reached as `ExpertBackupManager(...)` ->
`backup_weights_from_disk` -> imported loader code -> `get_model()`.

Reaching no read is not a pass. The walk is a static one, and every call it
cannot resolve -- a callable handed in as a parameter, an attribute off
something other than `self` -- turns into "reaches no accessor", which is also
what a defect looks like. So every publisher that reaches none is named in
`_UNREAD_ENTRIES` with the reason, and that map is asserted against the whole
set the walk finds: a publisher that stops reaching a read, or a new one that
never reached any, fails here instead of becoming an entry with nothing to
check. Restricting the comparison to `_KNOWN_ENTRIES` would exempt exactly the
newly discovered entry the derivation exists to catch.

What this cannot pin: a publish moving across code that reads only the handed
`server_args` instance. Such code names no accessor, so there is no read for the
walk to order it against.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.config_publishers import publisher_names
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

_ACCESSORS = frozenset(
    {
        "get_exec",
        "get_memory",
        "get_schedule",
        "get_model",
        "get_spec",
        "get_serving",
        "get_observability",
        "get_disagg",
        "get_lora",
        "get_mm",
        "get_device",
        "get_parallel",
    }
)

# The process entries that must be found by the walk. A derivation that stops
# matching -- an import rewritten, a publish moved behind a helper -- would
# otherwise leave this test green over an empty set.
_KNOWN_ENTRIES = frozenset(
    {
        ("srt/managers/scheduler.py", "run_scheduler_process"),
        ("srt/managers/detokenizer_manager.py", "run_detokenizer_process"),
        (
            "srt/managers/data_parallel_controller.py",
            "run_data_parallel_controller_process",
        ),
        ("srt/ray/scheduler_actor.py", "__init__"),
        ("srt/disaggregation/encoder/server.py", "__init__"),
        ("srt/disaggregation/encoder/http_server.py", "launch_server"),
        ("srt/managers/tokenizer_manager.py", "__init__"),
        ("srt/entrypoints/engine.py", "_launch_subprocesses"),
        (
            "srt/elastic_ep/expert_backup_manager.py",
            "run_expert_backup_manager_process",
        ),
        ("srt/weight_cache/daemon.py", "load"),
    }
)

# Every publisher the walk finds whose callees reach no bag accessor at this
# revision, and why. Asserted exactly against what the walk finds -- not
# intersected with `_KNOWN_ENTRIES`, which would drop a newly discovered entry
# and check no ordering for the one case the derivation exists to catch.
# "Reaches none" is also what the walk answers when it cannot resolve a call, so
# every one of them is named. An entry leaves this map in the commit that gives
# it a bag read.
_UNREAD_ENTRIES: dict = {
    # Not process entries: these publish to set up a context for themselves.
    ("kernels/aot/tests/test_fused_qk_norm_rope.py", "test_fused_qk_norm_rope"): (
        "a kernel test publishing its own context"
    ),
    ("multimodal_gen/test/unit/test_disagg_trace.py", "_srt_trace_server_args"): (
        "a trace fixture publishing its own context"
    ),
    (
        "multimodal_gen/runtime/managers/gpu_worker.py",
        "init_device_and_model",
    ): (
        "a worker installing a placeholder when its process has nothing "
        "published; it reads its own config, not the srt bags"
    ),
}

# `publish` itself and its named wrappers live here; a call inside them is the
# definition, not a process entry.
_PUBLISH_HOMES = frozenset({"srt/runtime_context.py", "srt/server_args.py"})

_CONFIG_MODULES = frozenset({"sglang.srt.runtime_context", "sglang.srt.server_args"})


def _module_path(dotted: str):
    """The package-relative file a `sglang.` import names, if it is one."""
    if not dotted.startswith("sglang."):
        return None
    parts = dotted.split(".")[1:]
    for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if (_PACKAGE_ROOT / candidate).exists():
            return candidate
    return None


_CALLS = {}


def _calls(fn):
    """(callee key, line) per call in this body.

    `self.f()` / `cls.f()` are keyed to the owning class; a bare name is a
    module-level def, an imported name, or a class -- for a class the call runs
    its `__init__`. `x.f()` for any other `x` yields two keys: the bare `x`,
    which resolves when `x` is a class (`PortArgs.init_new()`), and a
    module-qualified one that keeps `f`, which resolves when `x` is a module
    the file imported. Keeping only the bare name loses `f` entirely, so a
    helper called as `foo.initialize()` contributes nothing to the walk.
    Anything deeper (`a.b.c()`, a callable off an attribute) is not resolved.
    """
    if id(fn) in _CALLS:
        return _CALLS[id(fn)]
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            out.append((("name", func.id), node.lineno))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in ("self", "cls"):
                out.append((("self", func.attr), node.lineno))
            else:
                out.append((("name", func.value.id), node.lineno))
                out.append(((f"module:{func.value.id}", func.attr), node.lineno))
    _CALLS[id(fn)] = out
    return out


_PUBLISH_NAMES = publisher_names(_PACKAGE_ROOT / "srt")


class _Module:
    """One parsed module: what it calls the config API, and what it defines.

    The publisher and accessor names are resolved from the imports rather than
    matched by name: a model's ``index_topk_share.publish()`` and a platform's
    ``get_device()`` are unrelated methods that a name-only match reports as
    config calls.
    """

    def __init__(self, rel: str, tree):
        self.rel, self.tree = rel, tree
        self.publishers, self.accessors, self.qualified = set(), set(), set()
        self.imported = {}
        # Names bound to another sglang module rather than to a symbol in one:
        # `import sglang.srt.foo as foo` / `from sglang.srt import foo`. Without
        # these, `foo.initialize()` reaches nothing.
        self.modules = {}
        self.functions = {}
        self.classes = {}
        self.owner = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in _CONFIG_MODULES:
                    for alias in node.names:
                        local = alias.asname or alias.name
                        if alias.name in _PUBLISH_NAMES:
                            self.publishers.add(local)
                        elif alias.name in _ACCESSORS:
                            self.accessors.add(local)
                target = _module_path(node.module)
                if target is not None and target != rel:
                    for alias in node.names:
                        self.imported[alias.asname or alias.name] = (target, alias.name)
                for alias in node.names:
                    # `from sglang.srt import runtime_context` binds the module
                    # itself; `runtime_context.get_serving()` is the same read.
                    dotted = f"{node.module}.{alias.name}"
                    if dotted in _CONFIG_MODULES:
                        self.qualified.add(alias.asname or alias.name)
                    bound = _module_path(dotted)
                    if bound is not None and bound != rel:
                        self.modules[alias.asname or alias.name] = bound
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _CONFIG_MODULES:
                        self.qualified.add(alias.asname or alias.name.split(".")[0])
                    # Only `import a.b.c as name` binds a usable name; without
                    # the alias the call reads `a.b.c.f()`, which is deeper than
                    # this walk resolves.
                    bound = _module_path(alias.name) if alias.asname else None
                    if bound is not None and bound != rel:
                        self.modules[alias.asname] = bound
            elif isinstance(node, ast.ClassDef):
                methods = self.classes.setdefault(node.name, {})
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[stmt.name] = stmt
                        self.owner[id(stmt)] = node.name
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(node) not in self.owner
            ):
                self.functions.setdefault(node.name, node)
        self._direct = {}

    def is_publish(self, node) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name) and node.func.id in self.publishers:
            return True
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.qualified
        )

    def is_read(self, node) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name) and node.func.id in self.accessors:
            return True
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.qualified
            and node.func.attr in _ACCESSORS
        )

    def resolve(self, key, cls):
        """The def in this module a callee key names, if it names one.

        A module-qualified key names nothing here: its def lives in the module
        the alias is bound to, which `_targets` follows.
        """
        kind, name = key
        if kind == "self":
            return self.classes.get(cls, {}).get(name)
        if kind.startswith("module:"):
            return None
        if name in self.functions:
            return self.functions[name]
        return self.classes.get(name, {}).get("__init__")

    def direct_read(self, fn) -> bool:
        """Whether this body names an accessor itself."""
        if id(fn) not in self._direct:
            self._direct[id(fn)] = any(self.is_read(n) for n in ast.walk(fn))
        return self._direct[id(fn)]


_MODULES = {}


def _module(rel: str):
    # A module this walk cannot parse would resolve to "reaches no config",
    # which is the answer that hides a defect. utf-8-sig because a file in the
    # package carries a BOM; anything still unparsable fails the test.
    if rel not in _MODULES:
        _MODULES[rel] = _Module(
            rel, ast.parse((_PACKAGE_ROOT / rel).read_text(encoding="utf-8-sig"))
        )
    return _MODULES[rel]


def _defaulted_parameters(fn):
    """`{parameter: default name}` for the parameters this def gives a plain
    name as default. `run_detokenizer_process` reaches `DetokenizerManager`
    only this way -- the body calls the parameter, and the class it is really
    handed is written once, as that parameter's default."""
    if id(fn) in _DEFAULTS:
        return _DEFAULTS[id(fn)]
    arguments = fn.args
    positional = arguments.posonlyargs + arguments.args
    pairs = list(
        zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults)
    )
    pairs += zip(arguments.kwonlyargs, arguments.kw_defaults)
    _DEFAULTS[id(fn)] = {
        parameter.arg: default.id
        for parameter, default in pairs
        if isinstance(default, ast.Name)
    }
    return _DEFAULTS[id(fn)]


_DEFAULTS = {}


def _targets(mod, key, cls, fn=None):
    """The (module, def, owning class) a callee key names, here and one hop out.

    A bare name that is one of `fn`'s parameters resolves through that
    parameter's default as well, which is how a factory handed in as an
    argument is followed to the class the entry actually constructs.

    A module-qualified key (`foo.initialize()`) resolves in the module `foo` is
    bound to, so a helper reached that way joins the walk instead of dropping
    out of it.
    """
    keys = [key]
    if fn is not None and key[0] == "name":
        default = _defaulted_parameters(fn).get(key[1])
        if default is not None:
            keys.append(("name", default))
    for key in keys:
        target = mod.resolve(key, cls)
        if target is not None:
            yield mod, target, mod.owner.get(id(target), cls)
        if key[0].startswith("module:"):
            home = mod.modules.get(key[0][len("module:") :])
            if home is None:
                continue
            other = _module(home)
            target = other.resolve(("name", key[1]), None)
            if target is not None:
                yield other, target, other.owner.get(id(target))
            continue
        hop = mod.imported.get(key[1]) if key[0] == "name" else None
        if hop is None:
            continue
        other = _module(hop[0])
        target = other.resolve(("name", hop[1]), None)
        if target is not None:
            yield other, target, other.owner.get(id(target))


# The callee names from a def down to the read it reaches, for the defs a
# witness has been found for. Only "reaches a read" is carried between
# questions: a witness path stays one, while "reaches none" can be the answer a
# recursive call gets when it re-enters a def the walk is still inside, which
# is that call's answer and not the def's own.
_WITNESS = {}


def _witness(mod, fn, cls, seen):
    """The callees from this def down to a bag read, ending in the file that
    reads, or None for a def that reaches no read.

    `seen` belongs to one question. Skipping a def already on this walk keeps
    the answer for the def the question was asked about -- that def opened the
    skipped one, so a read below it still comes back up the path that opened
    it -- and bounds the walk on a call cycle.
    """
    key = (mod.rel, id(fn))
    if key in _WITNESS:
        return _WITNESS[key]
    if key in seen:
        return None
    seen.add(key)
    if mod.direct_read(fn):
        _WITNESS[key] = [mod.rel]
        return _WITNESS[key]
    for call, _ in _calls(fn):
        for target in _targets(mod, call, cls, fn):
            below = _witness(*target, seen)
            if below is not None:
                _WITNESS[key] = [call[1]] + below
                return _WITNESS[key]
    return None


def _publishing_functions():
    """(relative path, function node, module) per publisher."""
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel in _PUBLISH_HOMES:
            continue
        source = path.read_text(encoding="utf-8-sig")
        if not any(name in source for name in _PUBLISH_NAMES):
            continue
        mod = _module(rel)
        if mod is None or not mod.publishers:
            continue
        for fn in ast.walk(mod.tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(mod.is_publish(n) for n in ast.walk(fn)):
                yield rel, fn, mod


def _first_read(fn, mod):
    """(line, what read it) of the earliest config read this entry reaches."""
    cls = mod.owner.get(id(fn))
    marks = [(n.lineno, "a bag accessor") for n in ast.walk(fn) if mod.is_read(n)]
    for call, line in _calls(fn):
        for target in _targets(mod, call, cls, fn):
            if target[1] is fn:
                continue
            below = _witness(*target, set())
            if below is not None:
                chain = [call[1]] + below
                marks.append(
                    (line, " -> ".join(chain[:-1]) + f", which reads in {chain[-1]}")
                )
                break
    return min(marks) if marks else None


class TestPublishPrecedesBagReads(CustomTestCase):
    def test_every_publishing_entry_publishes_first(self):
        offenders = []
        found = set()
        unread = set()
        for rel, fn, mod in _publishing_functions():
            found.add((rel, fn.name))
            # ast.walk yields breadth-first, so the first match is not the
            # earliest line; take the minimum.
            publish_line = min(n.lineno for n in ast.walk(fn) if mod.is_publish(n))
            read = _first_read(fn, mod)
            if read is None:
                unread.add((rel, fn.name))
            elif read[0] < publish_line:
                offenders.append(
                    f"{rel}:{fn.name} reaches a config namespace at line "
                    f"{read[0]} through {read[1]}, before its publish at "
                    f"{publish_line}"
                )
        self.assertEqual(
            sorted(_KNOWN_ENTRIES - found),
            [],
            "the walk stopped finding known process entries; the derivation "
            "is broken, not the tree",
        )
        self.assertEqual(
            sorted(unread),
            sorted(_UNREAD_ENTRIES),
            "a publisher reaching no bag accessor checks nothing; either the "
            "walk stopped resolving a call, or a publisher appeared or moved "
            "its reads and _UNREAD_ENTRIES has to say so",
        )
        self.assertEqual(
            offenders,
            [],
            "a spawned worker starts with an empty context:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
