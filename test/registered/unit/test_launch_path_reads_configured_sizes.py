"""Launch paths read the configured parallel sizes, not the live ones.

`get_parallel().pp_size` and its four siblings are read-through properties over
the process groups, so they answer only after distributed init. The launcher
decides how many processes to spawn *before* that, and a live read there raises
`Distributed environment is not initialized` -- a startup crash no unit test
reaches, because nothing short of booting a server runs the launcher.
"""

import ast
import functools
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent

# Live-shadowed sizes a launch path is known to have read. ParallelContext
# shadows more properties than these (every `_v(name, ...)` one raises the same
# "Distributed environment is not initialized"); this dict carries the ones a
# `configured_*` accessor answers, so it is a remedy map, not a census.
_LIVE_SHADOWED = {
    "tp_size": "configured_tp_size()",
    "pp_size": "configured_pp_size()",
    "moe_dp_size": "configured_moe_dp_size()",
    "attn_cp_size": "configured_attn_cp_size()",
    "dcp_size": "configured_dcp_size()",
}

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


@functools.lru_cache(maxsize=None)
def _configured_accessors() -> frozenset:
    """The `configured_*_size()` names `runtime_context` exports.

    Derived from that module, so a new accessor keeps its launcher watched
    without a second list here.
    """
    tree = ast.parse(
        (_PACKAGE_ROOT / "srt/runtime_context.py").read_text(encoding="utf-8-sig")
    )
    names = frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("configured_")
        and node.name.endswith("_size")
    )
    assert names, (
        "no configured_*_size accessors found in runtime_context; the "
        "derivation is broken, not the tree"
    )
    return names


def _spawns_from_a_size(tree) -> bool:
    """Does any function here construct a child process *and* read one of the
    five sizes -- live off the parallel bag, or through its `configured_*_size()`
    answer? That is a spawn count decided from the topology.

    Counting the configured read too is what keeps a launcher watched after it
    is converted. Deriving on the live read alone means the file drops out of
    the scan the moment it stops offending, so the guard would only ever watch
    the launchers that already fail it.
    """
    configured = _configured_accessors()
    modules, constructors = _multiprocessing_names(tree)
    names, qualified = _parallel_bag_names(tree)
    aliases = _bag_aliases(tree, names, qualified)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        spawns = reads = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in (
                    "Process",
                    "ProcessPoolExecutor",
                    "Popen",
                    "spawn",
                ):
                    # `mp.Process(`, `mp.get_context("spawn").Process(` and
                    # `subprocess.Popen(` all reach a child process; the
                    # receiver of a chained call is itself a call, so this
                    # cannot require a bare Name.
                    spawns = True
                elif isinstance(func, ast.Name) and func.id in constructors:
                    spawns = True
                if (isinstance(func, ast.Name) and func.id in configured) or (
                    isinstance(func, ast.Attribute) and func.attr in configured
                ):
                    reads = True
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in _LIVE_SHADOWED
                and (
                    _is_parallel_bag_call(node.value, names, qualified)
                    or (isinstance(node.value, ast.Name) and node.value.id in aliases)
                )
            ):
                # A record read (`server_args.tp_size`) sizes a spawn too, but
                # it cannot raise pre-dist; only the bag read is this guard's
                # subject, so only it forces a module into _PRE_DIST.
                reads = True
        if spawns and reads:
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
    """Locals bound to the parallel bag: `p = get_parallel()` then `p.pp_size`
    is the same read one line later."""
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and _is_parallel_bag_call(node.value, names, qualified)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _launch_paths():
    """(relative path, tree) per module that runs before its process groups.

    A module that sizes a spawn loop from a parallel-bag size is derived from
    the spawn itself; `_HAND_CARRIED` holds the launch entries that spawn
    nothing, which no derivation can reach.
    """
    seen = {}
    sizes = frozenset(_LIVE_SHADOWED) | _configured_accessors()
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
        nothing about what `configured_*()` returns once the groups *are* up
        and answering a different number -- which is not hypothetical: elastic
        EP scales the live topology away from what the operator configured, and
        that divergence is the entire reason these five helpers exist. With
        only the early-read direction covered, a helper that quietly delegated
        to the live property would look correct.
        """
        import json
        import os
        import tempfile
        from unittest.mock import patch

        from sglang.srt.runtime_context import (
            configured_attn_cp_size,
            configured_dcp_size,
            configured_moe_dp_size,
            configured_pp_size,
            configured_tp_size,
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
        helpers = {
            "tp_size": configured_tp_size,
            "pp_size": configured_pp_size,
            "moe_dp_size": configured_moe_dp_size,
            "attn_cp_size": configured_attn_cp_size,
            "dcp_size": configured_dcp_size,
        }
        missing = sorted(set(helpers) - set(live_getter))
        self.assertEqual(
            missing,
            [],
            f"these sizes no longer have a live property to diverge from: {missing}",
        )
        cases = tuple(
            (name, helper, f"{state}.{live_getter[name]}")
            for name, helper in helpers.items()
        )
        for name, helper, target in cases:
            with self.subTest(size=name):
                configured = helper()
                with patch(target, return_value=configured + 41):
                    self.assertEqual(
                        get_parallel().__getattribute__(name),
                        configured + 41,
                        f"{name} no longer follows the live topology",
                    )
                    self.assertEqual(
                        helper(),
                        configured,
                        f"configured_{name}() followed the live topology instead "
                        "of the published configuration",
                    )
        reset_context()

    def test_no_live_topology_read_before_distributed_init(self):
        offenders = []
        for rel, tree in _launch_paths():
            names, modules = _parallel_bag_names(tree)
            aliases = _bag_aliases(tree, names, modules)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in _LIVE_SHADOWED:
                    base = node.value
                    if _is_parallel_bag_call(base, names, modules) or (
                        isinstance(base, ast.Name) and base.id in aliases
                    ):
                        offenders.append(
                            f"{rel}:{node.lineno} reads the live {node.attr}; "
                            f"use {_LIVE_SHADOWED[node.attr]}"
                        )
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in _LIVE_SHADOWED
                    and (
                        _is_parallel_bag_call(node.args[0], names, modules)
                        or (
                            isinstance(node.args[0], ast.Name)
                            and node.args[0].id in aliases
                        )
                    )
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} reads the live "
                        f"{node.args[1].value} through getattr; "
                        f"use {_LIVE_SHADOWED[node.args[1].value]}"
                    )
        self.assertEqual(
            offenders,
            [],
            "launch paths run before distributed init:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
