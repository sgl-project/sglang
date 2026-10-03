"""Contract test for the SGLang surface the external Foundry plugin uses.

Foundry (CUDA graph save/restore, enabled by FOUNDRY_GRAPH_EXTENSION_CONFIG)
wraps existing functions on their dotted paths and reads a few runtime
context fields. Nothing in the tree exercises that, so a rename, a signature
change or a call that stops dispatching through the patched attribute would
only fail at a plugin user's startup. The test reads source only: no GPU, no
Foundry.

Run:  python3 test/registered/unit/plugins/test_foundry_plugin_contract.py
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = Path(sglang.__file__).parent / "srt"
_DECODE = "model_executor/runner/decode_cuda_graph_runner.py"
_PREFILL = "model_executor/runner/prefill_cuda_graph_runner.py"
_MR = "model_executor/model_runner.py"

# Functions Foundry wraps: (defined in, qualname, parameters with "*" before
# keyword-only ones, a caller, the call expression that must reach the
# patched attribute).
_WRAPPED = [
    ("distributed/bootstrap.py", "init_parallel_runtime",
     ["*", "server_args", "device", "dist_port"],
     "managers/scheduler.py", "bootstrap.init_parallel_runtime"),
    (_MR, "ModelRunner.init_torch_distributed", ["self"],
     _MR, "self.init_torch_distributed"),
    (_MR, "ModelRunner.alloc_memory_pool", ["self", "memory_pool_config"],
     "managers/tp_worker.py", "self.model_runner.alloc_memory_pool"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     ["self", "pre_model_load_memory"],
     "mem_cache/kv_cache_configurator.py", "self._resolve_memory_pool_config"),
    (_DECODE, "DecodeCudaGraphRunner.capture", ["self"], _DECODE, "self.capture"),
    (_DECODE, "DecodeCudaGraphRunner._resolve_shared_read_ends",
     ["self", "attn_backend", "forward_mode"],
     _DECODE, "self._resolve_shared_read_ends"),
    (_PREFILL, "PrefillCudaGraphRunner.capture", ["self"], _PREFILL, "self.capture"),
    ("model_executor/runner_backend/full_cuda_graph_backend.py",
     "FullCudaGraphBackend.capture_one",
     ["self", "shape_key", "forward_fn", "capture_inputs", "post_warmup_hook"],
     _DECODE, "self.backend.capture_one"),
    ("managers/data_parallel_controller.py",
     "DataParallelController.launch_tensor_parallel_group",
     ["self", "server_args", "port_args", "base_gpu_id", "dp_rank", "worker_ports"],
     "managers/data_parallel_controller.py", "self.launch_tensor_parallel_group"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     ["cls", "server_args", "port_args", "run_scheduler_process_func", "*",
      "placement_group"],
     "entrypoints/engine.py", "cls._launch_scheduler_processes"),
]  # fmt: skip

# Names Foundry reads: get_parallel().<name>, get_device().<name>,
# get_flags().dp.<name>, get_context().<name>() and the accessors themselves.
_READ = [
    ("arg_groups/fields/parallel.py", "Parallel",
     {"tp_size", "pp_size", "enable_dp_attention", "tp_rank", "pp_rank",
      "dp_rank", "attn_dp_rank"}),
    ("arg_groups/fields/device.py", "Device", {"device", "gpu_id"}),
    ("runtime_context.py", "DpFlags", {"prefill_graph_has_dp_gather"}),
    ("runtime_context.py", "RuntimeContext", {"override", "overrides_log"}),
    ("runtime_context.py", None,
     {"get_parallel", "get_device", "get_flags", "get_context"}),
]  # fmt: skip

# Hooks apply in the processes that run load_plugins(); the scheduler must
# load them before it builds the Scheduler (parallel bring-up, capture).
_LOAD_PLUGINS_SITES = [
    ("entrypoints/engine.py", "Engine.__init__"),
    ("entrypoints/engine.py", "Engine._launch_subprocesses"),
    ("managers/scheduler.py", "run_scheduler_process"),
]

# Resolution steps Foundry wraps with register_resolution_hook, in run order.
_RESOLUTION_STEPS = (
    "apply_inkling_prefill_cuda_graph_default",
    "handle_cuda_graph_config",
    "handle_other_validations",
)


def _tree(relpath):
    return ast.parse((_SRT / relpath).read_text(encoding="utf-8"))


def _find(relpath, qualname):
    node = _tree(relpath)
    for part in qualname.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part
        )
    return node


def _params(fn):
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg or a.kwonlyargs:
        names.append(f"*{a.vararg.arg}" if a.vararg else "*")
    return names + [p.arg for p in a.kwonlyargs]


def _calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _members(node):
    names = set()
    for n in node.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


class TestFoundryPluginContract(CustomTestCase):
    def test_wrapped_functions_keep_signature_and_dispatch(self):
        """A wrapper forwards these parameters; a caller that stops going
        through the attribute (e.g. a from-import) bypasses the wrapper."""
        for path, qualname, params, caller, call in _WRAPPED:
            with self.subTest(target=qualname):
                self.assertEqual(_params(_find(path, qualname)), params)
                called = {ast.unparse(c.func) for c in _calls(_tree(caller))}
                self.assertIn(call, called, f"{caller} no longer calls {call}")
        launch = _find("entrypoints/engine.py", "Engine._launch_scheduler_processes")
        self.assertIn("classmethod", [ast.unparse(d) for d in launch.decorator_list])

    def test_read_names_exist(self):
        """Renamed fields would give Foundry a wrong workspace rank or device."""
        for path, cls, names in _READ:
            with self.subTest(owner=cls or path):
                owner = _find(path, cls) if cls else _tree(path)
                self.assertLessEqual(names, _members(owner))

    def test_plugin_loading_sites_and_group(self):
        """Wrappers exist only in processes that ran load_plugins()."""
        for path, qualname in _LOAD_PLUGINS_SITES:
            with self.subTest(site=qualname):
                calls = [ast.unparse(c.func) for c in _calls(_find(path, qualname))]
                self.assertIn("load_plugins", calls)
        sched = _find("managers/scheduler.py", "run_scheduler_process")
        call_lines = {}
        for c in _calls(sched):
            call_lines.setdefault(ast.unparse(c.func), []).append(c.lineno)
        self.assertLess(min(call_lines["load_plugins"]), min(call_lines["Scheduler"]))
        groups = {
            t.id: n.value.value
            for n in _tree("plugins/__init__.py").body
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
            for t in n.targets
        }
        self.assertEqual(groups["GENERAL_PLUGINS_GROUP"], "sglang.srt.plugins")

    def test_resolution_steps_stay_overridable_in_order(self):
        """Foundry's graph-config pins must land before the prefill default
        and the CUDA graph config parse."""
        overridable = next(
            n.value
            for n in _tree("arg_groups/resolution_hooks.py").body
            if isinstance(n, ast.AnnAssign) and n.target.id == "_OVERRIDABLE_HOOKS"
        )
        overridable = {
            n.value for n in ast.walk(overridable) if isinstance(n, ast.Constant)
        }
        src = ast.unparse(_find("arg_groups/pipeline.py", "run_resolution_pipeline"))
        positions = []
        for name in _RESOLUTION_STEPS:
            self.assertIn(name, overridable)
            positions.append(src.index(f"run_hook({name}, server_args)"))
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
