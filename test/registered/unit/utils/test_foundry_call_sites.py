"""Contract test for the Foundry call sites (--cuda-graph-persistence).

Foundry (CUDA graph save/restore) is an optional dependency that SGLang calls
from fixed call sites through sglang.srt.utils.foundry_adapter. Its archive
is only valid if those calls stay at their sequence points (before or after
the work they bracket), so a refactor that moves or drops one fails here
instead of at a user's LOAD. The test reads source only: no GPU, no Foundry.

Run:  python -m pytest test/registered/unit/utils/test_foundry_call_sites.py -v
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

# (file, function qualname, adapter call, where): "first" = the first
# statement after the docstring, "last" = the last statement before the
# return, "before:X" / "after:X" = before the next / after the last use of X,
# "any" = somewhere in the body.
_CALL_SITES = [
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().before_parallel_init", "before:_resolve_backend"),
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().after_parallel_init", "after:_init_parallel_groups"),
    (_MR, "ModelRunner.init_torch_distributed",
     "get_foundry_adapter().after_runner_distributed_init", "last"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().before_alloc_memory_pool", "first"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().after_alloc_memory_pool", "last"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().replay_saved_memory_pool_config",
     "before:self._profile_available_bytes"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().record_memory_pool_overrides", "last"),
    (_DECODE, "DecodeCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    (_PREFILL, "PrefillCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    (_DECODE, "DecodeCudaGraphRunner._resolve_shared_read_ends",
     "get_foundry_adapter().shared_read_ends_override",
     "before:attn_backend.shared_read_ends"),
    ("model_executor/runner_backend/full_cuda_graph_backend.py",
     "FullCudaGraphBackend.capture_one", "foundry.capture_one",
     "before:self._cuda_graph_runner"),
    ("managers/scheduler.py", "run_scheduler_process", "activate_foundry",
     "before:publish"),
    ("managers/data_parallel_controller.py", "run_data_parallel_controller_process",
     "activate_foundry", "before:publish"),
    ("managers/data_parallel_controller.py",
     "DataParallelController.launch_tensor_parallel_group",
     "get_foundry_adapter().configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "foundry_adapter.configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "activate_foundry(server_args).configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "_set_envs_and_config",
     "activate_foundry(server_args).apply_env_pins", "before:is_mnnvl_fabric_device"),
]  # fmt: skip

# Resolution steps, each right after the step named (None: right before the
# next one), in pipeline order.
_RESOLUTION_STEPS = [
    ("handle_cuda_graph_persistence", "before", "apply_inkling_prefill_cuda_graph_default"),
    ("validate_cuda_graph_persistence_graph_config", "after", "handle_cuda_graph_config"),
    ("validate_cuda_graph_persistence", "after", "handle_other_validations"),
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

# Attributes Foundry reads on the runners it is handed (capture scope, decode
# runner checks). The backend's fields are not among them: capture_one gets
# the pool, stream and prefill request slots as arguments and SGLang stores the
# returned graph and output.
_ATTRIBUTES = [
    (_PREFILL, "PrefillCudaGraphRunner.__init__",
     {"_capture_req_slots", "_is_full_backend", "prefill_backend_name"}),
    (_DECODE, "DecodeCudaGraphRunner.__init__", {"backend", "in_graph_metadata_prep_done"}),
]  # fmt: skip


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


def _call_lines(fn, prefix):
    """Lines of calls whose callee source starts with ``prefix``."""
    return sorted(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith(prefix)
    )


def _name_lines(fn, prefix):
    return sorted(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, (ast.Call, ast.Attribute, ast.Name))
        and ast.unparse(n).startswith(prefix)
    )


def _body(fn):
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return body


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


class TestFoundryCallSites(CustomTestCase):
    def test_call_sites_stay_at_their_sequence_points(self):
        for path, qualname, call, where in _CALL_SITES:
            with self.subTest(site=qualname, call=call):
                fn = _find(path, qualname)
                lines = _call_lines(fn, call)
                self.assertTrue(lines, f"{qualname} no longer calls {call}")
                body = _body(fn)
                if where == "first":
                    self.assertIn(call, ast.unparse(body[0]))
                elif where == "last":
                    tail = [s for s in body if not isinstance(s, ast.Return)][-1]
                    self.assertIn(call, ast.unparse(tail))
                elif where.startswith("before:"):
                    other = _name_lines(fn, where.split(":", 1)[1])
                    self.assertTrue(any(o > lines[0] for o in other), where)
                elif where.startswith("after:"):
                    other = _name_lines(fn, where.split(":", 1)[1])
                    self.assertTrue(other, where)
                    self.assertGreater(lines[-1], other[-1])

    def test_capture_one_passes_arguments_and_stores_the_result(self):
        """Foundry gets what it needs as arguments and SGLang assigns the
        returned graph and output into its own backend fields."""
        fn = _find(
            "model_executor/runner_backend/full_cuda_graph_backend.py",
            "FullCudaGraphBackend.capture_one",
        )
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "foundry.capture_one"
        )
        self.assertEqual(
            [ast.unparse(a) for a in call.args], ["shape_key", "forward_fn"]
        )
        self.assertEqual(
            {k.arg: ast.unparse(k.value) for k in call.keywords},
            {
                "pool": "self._pool",
                "stream": "self._capture_stream",
                "prefill_req_slots": "self._prefill_req_slots()",
            },
        )
        assign = next(
            n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is call
        )
        self.assertEqual(
            [ast.unparse(t) for t in assign.targets[0].elts],
            ["self._graphs[shape_key]", "self._outputs[shape_key]"],
        )
        # Nothing hands Foundry the backend itself.
        self.assertNotIn("self", [ast.unparse(a) for a in call.args])

    def test_capture_scope_wraps_the_whole_capture_loop(self):
        """Foundry's scope must bracket warmup and every shape: the runner's
        capture() is only the scope around _capture_graphs()."""
        for path, cls in (
            (_DECODE, "DecodeCudaGraphRunner"),
            (_PREFILL, "PrefillCudaGraphRunner"),
        ):
            with self.subTest(runner=cls):
                body = _body(_find(path, f"{cls}.capture"))
                self.assertEqual(len(body), 1)
                self.assertIsInstance(body[0], ast.With)
                inner = ast.unparse(body[0].body[0])
                self.assertEqual(inner, "self._capture_graphs()")
                graphs = _find(path, f"{cls}._capture_graphs")
                self.assertTrue(_call_lines(graphs, "self.warmup"))

    def test_resolution_steps_keep_their_positions(self):
        overridable = next(
            n.value
            for n in _tree("arg_groups/resolution_hooks.py").body
            if isinstance(n, ast.AnnAssign) and n.target.id == "_OVERRIDABLE_HOOKS"
        )
        overridable = {
            n.value for n in ast.walk(overridable) if isinstance(n, ast.Constant)
        }
        steps = [
            ast.unparse(n.args[0])
            for n in ast.walk(
                _find("arg_groups/pipeline.py", "run_resolution_pipeline")
            )
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "run_hook"
        ]
        for name, relation, anchor in _RESOLUTION_STEPS:
            with self.subTest(step=name):
                self.assertIn(name, overridable)
                offset = 1 if relation == "after" else -1
                self.assertEqual(steps[steps.index(anchor) + offset], name)
        order = [steps.index(name) for name, _, _ in _RESOLUTION_STEPS]
        self.assertEqual(order, sorted(order))

    def test_flags_exist(self):
        fields = _members(_find("arg_groups/fields/exec_.py", "ExecGraph"))
        self.assertLessEqual(
            {"cuda_graph_persistence", "cuda_graph_persistence_config"}, fields
        )

    def test_read_names_exist(self):
        """Renamed fields would give Foundry a wrong workspace rank or device."""
        for path, cls, names in _READ:
            with self.subTest(owner=cls or path):
                owner = _find(path, cls) if cls else _tree(path)
                self.assertLessEqual(names, _members(owner))

    def test_attributes_foundry_touches_exist(self):
        for path, qualname, names in _ATTRIBUTES:
            with self.subTest(owner=qualname):
                assigned = {
                    n.attr
                    for n in ast.walk(_find(path, qualname))
                    if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "self"
                    and isinstance(n.ctx, ast.Store)
                }
                self.assertLessEqual(names, assigned)


if __name__ == "__main__":
    unittest.main()
