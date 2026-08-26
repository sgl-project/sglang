"""Launch paths read the configured parallel sizes, not the live ones.

`get_parallel().pp_size` and its four siblings are read-through properties over
the process groups, so they answer only after distributed init. The launcher
decides how many processes to spawn *before* that, and a live read there raises
`Distributed environment is not initialized` -- a startup crash no unit test
reaches, because nothing short of booting a server runs the launcher. The
configured answer is one hop away on the same object,
`get_parallel().config.pp_size`, which reads the published `parallel` bag.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent


def _live_shadowed() -> dict:
    """{name: remedy} for every name that is BOTH a live ParallelContext
    property and a `parallel` config leaf.

    Derived from the two sides themselves, so a new size that gains a live
    property (or a live property that gains a leaf) is watched without a second
    list here. ParallelContext shadows more properties than these -- every
    `_v(name, ...)` one raises the same "Distributed environment is not
    initialized" -- but only a shadowed name has a configured answer to point a
    launcher at.
    """
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.runtime_context import ParallelContext
    from sglang.srt.server_args import ServerArgs

    live = {
        name
        for name, value in vars(ParallelContext).items()
        if isinstance(value, property)
    }
    leaves = {
        field for field, path in namespace_of(ServerArgs).items() if path == "parallel"
    }
    shadowed = live & leaves
    assert shadowed, (
        "no live-shadowed parallel size found; the derivation is broken, not "
        "the tree"
    )
    return {name: f"get_parallel().config.{name}" for name in sorted(shadowed)}


_LIVE_SHADOWED = _live_shadowed()

# Launch paths that decide how many children to spawn are derived below
# from the spawn itself. These launch without a size-driven spawn, so no
# derivation reaches them and they are carried by hand.
_HAND_CARRIED = (
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/sidecar.py",
    "srt/ray/data_parallel_controller.py",
    "srt/ray/engine.py",
    "srt/ray/http_server.py",
)


def _multiprocessing_names(tree):
    """Names bound to multiprocessing, to one of its start contexts, or to the
    process constructors themselves."""
    modules, constructors = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "multiprocessing" or a.name.startswith("multiprocessing."):
                    modules.add(a.asname or a.name.split(".")[0])
                elif a.name == "torch.multiprocessing":
                    modules.add(a.asname or "torch")
        elif isinstance(node, ast.ImportFrom):
            if node.module in (
                "multiprocessing",
                "multiprocessing.context",
                "torch.multiprocessing",
            ):
                constructors |= {
                    a.asname or a.name for a in node.names if a.name == "Process"
                }
            elif node.module == "concurrent.futures":
                constructors |= {
                    a.asname or a.name
                    for a in node.names
                    if a.name == "ProcessPoolExecutor"
                }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get_context"
                and isinstance(func.value, ast.Name)
                and func.value.id in modules
            ):
                modules |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return modules, constructors


def _spawns_from_a_size(tree) -> bool:
    """Does a function here spawn a child *and* read a live-shadowed size?

    Both tiers count: deriving on the live read alone drops a launcher from the
    scan the moment it is converted, so the guard would only watch the ones
    that already fail it.
    """
    modules, constructors = _multiprocessing_names(tree)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        spawns = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in (
                "Process",
                "ProcessPoolExecutor",
                "Popen",
                "spawn",
            ):
                # `mp.Process(`, `mp.get_context("spawn").Process(` and
                # `subprocess.Popen(` all reach a child process; the receiver of
                # a chained call is itself a call, so this cannot require a bare
                # Name.
                spawns = True
            elif isinstance(func, ast.Name) and func.id in constructors:
                spawns = True
        if not spawns:
            continue
        # A record read (`server_args.tp_size`) sizes a spawn too, but it cannot
        # raise pre-dist; only a bag read is this guard's subject.
        live, configured = _shadowed_size_reads(tree, scope=fn)
        if live or configured:
            return True
    return False


def _parallel_bag_names(tree):
    """What this module calls `get_parallel`, plus any runtime_context alias.

    A literal-name match reads only one spelling; an aliased import or a
    module-qualified call is the same read with a different surface.
    """
    names, modules = set(), set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("runtime_context")
        ):
            names |= {
                a.asname or a.name for a in node.names if a.name == "get_parallel"
            }
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from sglang.srt import runtime_context as rc` binds the module,
            # so `rc.get_parallel()` is the same call under another spelling.
            for a in node.names:
                if f"{node.module}.{a.name}".endswith("runtime_context"):
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("runtime_context"):
                    modules.add(a.asname or a.name.split(".")[0])
    return names, modules


def _is_parallel_bag_call(node, names, modules) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in names
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_parallel"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in modules
    )


def _bag_aliases(tree, names, qualified):
    """Locals bound to either tier: `p = get_parallel()` then `p.pp_size` is the
    same live read one line later, and `cfg = get_parallel().config` then
    `cfg.pp_size` is the same configured read."""
    live, config = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if _is_parallel_bag_call(value, names, qualified):
            bucket = live
        elif (
            isinstance(value, ast.Attribute)
            and value.attr == "config"
            and _is_parallel_bag_call(value.value, names, qualified)
        ):
            bucket = config
        else:
            continue
        bucket |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return live, config


def _shadowed_size_reads(module_tree, scope=None):
    """(live, configured) reads of a live-shadowed size in `scope`.

    `<parallel bag>.tp_size` is the live group; `<parallel bag>.config.tp_size`
    is the published leaf. Both spellings are reported so a caller can tell a
    launcher that reads the topology at all from one that reads it live.

    What binds the bag -- the import, a module-level alias -- lives at module
    scope, so those names always come from `module_tree` even when only one
    function is being walked. Deriving them from the function alone finds no
    import, reports no reads, and quietly answers "this launcher reads nothing".
    """
    names, qualified = _parallel_bag_names(module_tree)
    live_aliases, config_aliases = _bag_aliases(module_tree, names, qualified)

    def is_live_bag(node):
        return _is_parallel_bag_call(node, names, qualified) or (
            isinstance(node, ast.Name) and node.id in live_aliases
        )

    def is_config_bag(node):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "config"
            and is_live_bag(node.value)
        ) or (isinstance(node, ast.Name) and node.id in config_aliases)

    live, configured = [], []
    for node in ast.walk(scope if scope is not None else module_tree):
        if isinstance(node, ast.Attribute) and node.attr in _LIVE_SHADOWED:
            base, name, spelling = node.value, node.attr, "attribute"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _LIVE_SHADOWED
        ):
            base, name, spelling = node.args[0], node.args[1].value, "getattr"
        else:
            continue
        if is_config_bag(base):
            configured.append((node.lineno, name, spelling))
        elif is_live_bag(base):
            live.append((node.lineno, name, spelling))
    return live, configured


def _launch_paths():
    """(relative path, tree) per module that runs before its process groups.

    A module that sizes a spawn loop from a parallel-bag size is derived from
    the spawn itself; `_HAND_CARRIED` holds the launch entries that spawn
    nothing, which no derivation can reach.
    """
    seen = {}
    sizes = frozenset(_LIVE_SHADOWED)
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text()
        # Every spawn shape below names Process, ProcessPoolExecutor or Popen.
        if not any(name in source for name in ("Process", "Popen", "spawn")):
            continue
        if not any(name in source for name in sizes):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if _spawns_from_a_size(tree):
            seen[str(path.relative_to(_PACKAGE_ROOT))] = tree
    for rel in _HAND_CARRIED:
        seen.setdefault(rel, ast.parse((_PACKAGE_ROOT / rel).read_text()))
    return sorted(seen.items())


class TestLaunchPathsReadConfiguredSizes(CustomTestCase):
    def test_configured_sizes_hold_when_the_live_topology_disagrees(self):
        """The other direction: groups exist and answer something else.

        The check above proves nobody reads a live size too early. It says
        nothing about what `.config.<size>` returns once the groups *are* up and
        answering a different number -- which is not hypothetical: elastic EP
        scales the live topology away from what the operator configured, and
        that divergence is the entire reason the two tiers are separate. With
        only the early-read direction covered, a `config` hop that quietly
        delegated to the live property would look correct.
        """
        import json
        import os
        import tempfile
        from unittest.mock import patch

        from sglang.srt.runtime_context import (
            ParallelContext,
            get_parallel,
            publish,
            reset_context,
        )
        from sglang.srt.server_args import ServerArgs

        directory = tempfile.mkdtemp(prefix="configured_sizes_")
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 2,
                    "vocab_size": 128,
                    "max_position_embeddings": 2048,
                },
                handle,
            )
        # No resolve_once() here: `tp_size` is raw input, so the configured
        # value is 2 either way.
        server_args = ServerArgs(model_path=directory, device="cuda", tp_size=2)
        self.addCleanup(reset_context)
        publish(server_args, role="scheduler")

        # The live getter behind each property, read out of ParallelContext
        # rather than listed here.
        context_source = ast.parse(
            (_PACKAGE_ROOT / "srt" / "runtime_context.py").read_text(
                encoding="utf-8-sig"
            )
        )
        parallel_class = next(
            node
            for node in ast.walk(context_source)
            if isinstance(node, ast.ClassDef) and node.name == "ParallelContext"
        )
        live_getter = {}
        for method in parallel_class.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for call in ast.walk(method):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_v"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                ):
                    continue
                getter = call.args[1]
                if isinstance(getter, ast.Attribute):
                    live_getter[call.args[0].value] = getter.attr
        state = "sglang.srt.distributed.parallel_state"
        missing = sorted(set(_LIVE_SHADOWED) - set(live_getter))
        self.assertEqual(
            missing,
            [],
            f"these sizes no longer have a live property to diverge from: {missing}",
        )
        for name in sorted(_LIVE_SHADOWED):
            with self.subTest(size=name):
                target = f"{state}.{live_getter[name]}"
                configured = getattr(get_parallel().config, name)
                with patch(target, return_value=configured + 41):
                    self.assertEqual(
                        get_parallel().__getattribute__(name),
                        configured + 41,
                        f"{name} no longer follows the live topology",
                    )
                    self.assertEqual(
                        getattr(get_parallel().config, name),
                        configured,
                        f"get_parallel().config.{name} followed the live topology "
                        "instead of the published configuration",
                    )
        # A leaf with no live property reads bare: there is no live value it
        # could be confused with. Compared against the declaration the bag was
        # projected from, so the two sides do not come from the same read.
        from sglang.srt.arg_groups.overrides import resolution_result

        self.assertEqual(
            resolution_result(server_args, "nccl_port"),
            getattr(get_parallel(), "nccl_port"),
            "a config-only leaf read bare disagreed with what resolution decided",
        )
        reset_context()

        # Before publish there is no value to hand back, and the error has to
        # name that rather than look like a misspelled attribute.
        with self.assertRaisesRegex(ValueError, r"'parallel' not published"):
            getattr(ParallelContext(), "nccl_port")
        with self.assertRaisesRegex(AttributeError, r"has no 'not_a_leaf'"):
            getattr(ParallelContext(), "not_a_leaf")

    def test_no_live_topology_read_before_distributed_init(self):
        offenders = []
        for rel, tree in _launch_paths():
            live, _ = _shadowed_size_reads(tree)
            for lineno, name, spelling in live:
                through = " through getattr" if spelling == "getattr" else ""
                offenders.append(
                    f"{rel}:{lineno} reads the live {name}{through}; "
                    f"use {_LIVE_SHADOWED[name]}"
                )
        self.assertEqual(
            offenders,
            [],
            "launch paths run before distributed init:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
