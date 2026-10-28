"""Reference structure of the scheduler's components.

`Scheduler` composes the components and runs the loop; every reference
points one way, from `Scheduler` down:

- R1: no component imports or annotates `Scheduler`.
- R2: `Scheduler` passes no lambda that captures it, no `self`, and no bound
  method of its own into a component.
- R3: the declared dependency graph has a topological order, and each built
  component takes exactly its declared dependencies; no code reaches a node
  through the workers (`tp_worker.model_runner.<node>`) instead of taking it.
- R4: nothing below `ModeOwners` takes a component that holds `ModeOwners`
  (they are replaced on a PD role switch).

The allowlists record the sites still to be removed: a new site fails, and a
removed site fails until its entry is deleted, so they only shrink.
"""

import ast
import unittest
import warnings
from collections import Counter
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_SCHEDULER = "managers/scheduler.py"
_ASSEMBLY = "managers/scheduler_assembly.py"
assert (_SRT_DIR / _SCHEDULER).is_file(), f"scheduler not found under {_SRT_DIR}"

# ---------------------------------------------------------------------------
# The declared graph: node -> the nodes it takes. Shared live state and
# frozen bundles take nothing.
# ---------------------------------------------------------------------------
_SHARED = ["CoreLists", "LaunchCounters", "EngineFlags", "PendingSteps"]
_FROZEN = [
    "KvResources",
    "KvCapacity",
    "RequestLimits",
    "Streams",
    "IpcChannels",
    "Workers",
]
_DECLARED = {
    **{node: [] for node in _SHARED + _FROZEN},
    "StageMetrics": [],
    "SchedulerServices": ["MetricsSink", "OutputStreamer", "StageMetrics"],
    "MetricsSink": [],
    "StepCounters": [],
    "SessionController": ["KvResources"],
    "FutureMap": ["KvResources"],
    "BeamCoordinator": ["KvResources", "FutureMap", "CoreLists"],
    "GrammarManager": [],
    "SchedulePolicy": ["KvResources"],
    "NewTokenRatioTracker": [],
    "HiSparse": ["KvResources"],
    "DecodeOffloadManager": ["KvResources"],
    "PoolStatsObserver": ["KvResources", "SessionController", "HiSparse", "KvCapacity"],
    "InvariantChecker": [
        "PoolStatsObserver",
        "KvResources",
        "KvCapacity",
        "RequestRegistry",
        "PendingSteps",
        "LaunchCounters",
        "StageMetrics",
    ],
    "KvEventsPublisher": ["KvResources", "IpcChannels", "MetricsSink", "KvCapacity"],
    "LoadPublisher": [],
    "ProfilerManager": ["LaunchCounters"],
    "OutputStreamer": ["IpcChannels", "KvResources"],
    "MMReceiver": ["Workers", "MetricsSink"],
    # The invariant checker is attached once both are built.
    "Watchdog": ["LaunchCounters", "EngineFlags", "InvariantChecker"],
    "PPSpecForward": ["Workers", "FutureMap", "KvResources", "Streams"],
    "DeferredResponses": ["OutputStreamer", "RequestRetirer"],
    "RequestRetirer": [
        "BeamCoordinator",
        "HiSparse",
        "DecodeOffloadManager",
        "PendingSteps",
        "OutputStreamer",
        "IpcChannels",
        "KvResources",
    ],
    "LoraAdmission": ["Workers"],
    "StepBuilder": [
        "KvResources",
        "FutureMap",
        "Workers",
        "LaunchCounters",
        "HiSparse",
    ],
    "DPAttnAdapter": ["Workers", "EngineFlags", "StepBuilder"],
    "BatchResultProcessor": [
        "OutputStreamer",
        "MetricsSink",
        "BeamCoordinator",
        "DecodeOffloadManager",
        "PendingSteps",
        "Workers",
        "KvResources",
        "RequestRetirer",
    ],
    "DisaggPrefill": [
        "KvResources",
        "KvCapacity",
        "Streams",
        "SchedulerServices",
        "BatchResultProcessor",
        "LaunchCounters",
        "RequestRetirer",
        "Workers",
        "CoreLists",
    ],
    "DisaggDecode": [
        "KvResources",
        "KvCapacity",
        "Streams",
        "SchedulerServices",
        "HiSparse",
        "LoraAdmission",
        "SchedulePolicy",
        "PoolStatsObserver",
        "DecodeOffloadManager",
        "EngineFlags",
        "CoreLists",
        "FutureMap",
        "RequestRetirer",
        "Workers",
    ],
    "DllmManager": [
        "KvResources",
        "SchedulerServices",
        "SchedulePolicy",
        "NewTokenRatioTracker",
        "BeamCoordinator",
        "StepBuilder",
        "RequestRetirer",
        "RequestLimits",
    ],
    "ModeOwners": ["DisaggPrefill", "DisaggDecode", "DllmManager", "HiSparse"],
    "RequestIntake": [
        "CoreLists",
        "LaunchCounters",
        "KvResources",
        "KvCapacity",
        "IpcChannels",
        "SchedulerServices",
        "BeamCoordinator",
        "SessionController",
        "GrammarManager",
        "RequestRetirer",
        "ModeOwners",
        "Workers",
        "RequestLimits",
    ],
    "RequestRegistry": [
        "CoreLists",
        "ModeOwners",
        "DeferredResponses",
        "GrammarManager",
        "RequestRetirer",
        "IpcChannels",
        "KvResources",
        "MMReceiver",
        "PendingSteps",
        "DecodeOffloadManager",
    ],
    # Prefill admission, decode selection and retraction, and the retract/reset
    # half of pause and flush.
    "StepPlanner": [
        "CoreLists",
        "PendingSteps",
        "EngineFlags",
        "LaunchCounters",
        "KvResources",
        "KvCapacity",
        "SchedulePolicy",
        "NewTokenRatioTracker",
        "PoolStatsObserver",
        "LoraAdmission",
        "RequestIntake",
        "RequestRegistry",
        "BeamCoordinator",
        "StepBuilder",
        "RequestRetirer",
        "GrammarManager",
        "MetricsSink",
        "IpcChannels",
        "DecodeOffloadManager",
        "ModeOwners",
        "Workers",
        "RequestLimits",
    ],
    "StepLauncher": [
        "Workers",
        "FutureMap",
        "BeamCoordinator",
        "Streams",
        "LaunchCounters",
        "EngineFlags",
        "SchedulerServices",
        "ProfilerManager",
        "PendingSteps",
        "CoreLists",
        "KvResources",
        "IpcChannels",
        "PPSpecForward",
    ],
    "MetricsReporter": [
        "MetricsSink",
        "RequestRegistry",
        "LaunchCounters",
        "PoolStatsObserver",
        "GrammarManager",
        "MMReceiver",
        "KvEventsPublisher",
        "ModeOwners",
        "KvResources",
        "Workers",
        "KvCapacity",
    ],
    "LoadInquirer": [
        "RequestRegistry",
        "ModeOwners",
        "StepCounters",
        "MetricsSink",
        "PoolStatsObserver",
        "SchedulePolicy",
        "Workers",
        "KvResources",
        "KvCapacity",
    ],
    "InternalState": [
        "RequestRegistry",
        "MetricsSink",
        "Workers",
        "KvResources",
        "KvCapacity",
    ],
    "WeightUpdater": [
        "StepPlanner",
        "RequestRegistry",
        "EngineFlags",
        "ModeOwners",
        "Workers",
        "MetricsSink",
    ],
    "RequestReceiver": ["IpcChannels", "MMReceiver", "OutputStreamer", "StageMetrics"],
    "FlushWrapper": ["StepPlanner", "RequestRegistry", "IpcChannels"],
    "PPRing": [
        "PendingSteps",
        "StepLauncher",
        "LaunchCounters",
        "EngineFlags",
        "PPSpecForward",
        "Workers",
        "FutureMap",
        "Streams",
        "KvResources",
    ],
}

# The constructor parameter (or dataclass field) names that take a node.
_PARAM = {
    "CoreLists": ("core_lists",),
    "LaunchCounters": ("counters",),
    "EngineFlags": ("flags",),
    "PendingSteps": ("pending",),
    "KvResources": ("kv",),
    "KvCapacity": ("capacity",),
    "RequestLimits": ("limits",),
    "Workers": ("workers",),
    "Streams": ("streams",),
    "SchedulerServices": ("services",),
    "IpcChannels": ("ipc_channels",),
    "StageMetrics": ("scheduler_stage_metrics",),
    "MetricsSink": ("metrics_sink",),
    "HiSparse": ("hisparse_coordinator",),
    "DisaggPrefill": ("disagg_prefill",),
    "DisaggDecode": ("disagg_decode",),
    "DllmManager": ("dllm",),
    "GrammarManager": ("grammar_manager",),
    "MMReceiver": ("mm_receiver",),
    "KvEventsPublisher": ("kv_events_publisher",),
    "OutputStreamer": ("output_streamer",),
    "BeamCoordinator": ("beam_coordinator",),
    "FutureMap": ("future_map",),
    "PoolStatsObserver": ("pool_stats_observer",),
    "SessionController": ("session_controller",),
    "InvariantChecker": ("invariant_checker",),
    "DecodeOffloadManager": ("decode_offload_manager",),
    "SchedulePolicy": ("policy",),
    "NewTokenRatioTracker": ("new_token_ratio_tracker",),
    "BatchResultProcessor": ("batch_result_processor",),
    "MetricsReporter": ("metrics_reporter",),
    "RequestRetirer": ("request_retirer",),
    "DeferredResponses": ("deferred_responses",),
    "RequestRegistry": ("request_registry", "abort_request"),
    "ModeOwners": ("mode_owners",),
    "RequestIntake": ("request_intake",),
    "StepBuilder": ("step_builder",),
    "StepCounters": ("step_counters",),
    "ProfilerManager": ("profiler_manager",),
    "PPSpecForward": ("pp_spec_forward",),
    "LoraAdmission": ("lora_admission",),
    "StepPlanner": ("step_planner",),
}

# Components in their final form: node -> (module under srt/, class).
_BUILT = {
    "CoreLists": ("managers/request_lists.py", "CoreLists"),
    "LaunchCounters": (
        "managers/scheduler_components/engine_state.py",
        "LaunchCounters",
    ),
    "EngineFlags": ("managers/scheduler_components/engine_state.py", "EngineFlags"),
    "ProfilerManager": (
        "managers/scheduler_components/profiler_manager.py",
        "SchedulerProfilerManager",
    ),
    "PendingSteps": ("managers/scheduler_components/pending_steps.py", "PendingSteps"),
    "IpcChannels": (
        "managers/scheduler_components/ipc_channels.py",
        "SchedulerIpcChannels",
    ),
    "NewTokenRatioTracker": (
        "managers/scheduler_components/new_token_ratio_tracker.py",
        "NewTokenRatioTracker",
    ),
    "LoadPublisher": (
        "managers/scheduler_components/load_publisher.py",
        "SchedulerLoadPublisher",
    ),
    "MetricsSink": ("managers/scheduler_components/metrics_reporter.py", "MetricsSink"),
    "MetricsReporter": (
        "managers/scheduler_components/metrics_reporter.py",
        "SchedulerMetricsReporter",
    ),
    "KvResources": ("managers/scheduler_components/bundles.py", "KvResources"),
    "KvCapacity": ("managers/scheduler_components/bundles.py", "KvCapacity"),
    "Streams": ("managers/scheduler_components/bundles.py", "Streams"),
    "SchedulerServices": (
        "managers/scheduler_components/bundles.py",
        "SchedulerServices",
    ),
    "GrammarManager": ("constrained/grammar_manager.py", "GrammarManager"),
    "RequestRetirer": (
        "managers/scheduler_components/request_retirer.py",
        "RequestRetirer",
    ),
    "DeferredResponses": (
        "managers/scheduler_components/deferred_responses.py",
        "DeferredResponses",
    ),
    "StepBuilder": ("managers/scheduler_components/step_builder.py", "StepBuilder"),
    "StepCounters": ("managers/scheduler_components/step_counters.py", "StepCounters"),
    "StepLauncher": (
        "managers/scheduler_components/step_launcher.py",
        "StepLauncher",
    ),
    "LoraAdmission": (
        "managers/scheduler_components/lora_admission.py",
        "LoraAdmission",
    ),
    "DisaggDecode": ("disaggregation/decode.py", "DisaggDecode"),
    "DisaggPrefill": ("disaggregation/prefill.py", "DisaggPrefill"),
    "DllmManager": ("dllm/manager.py", "DllmManager"),
    "ModeOwners": ("managers/scheduler_components/mode_owners.py", "ModeOwners"),
    "RequestRegistry": (
        "managers/scheduler_components/request_registry.py",
        "RequestRegistry",
    ),
    "RequestIntake": (
        "managers/scheduler_components/request_intake.py",
        "RequestIntake",
    ),
    "StepPlanner": (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner",
    ),
}

# The modules whose classes sit below ModeOwners.
_BELOW_MODE_OWNERS = (
    "disaggregation/decode.py",
    "disaggregation/prefill.py",
    "dllm/manager.py",
    "managers/hisparse_coordinator.py",
    "managers/scheduler_components/batch_result_processor.py",
    "managers/scheduler_components/output_streamer.py",
)

# ---------------------------------------------------------------------------
# Allowlists: {site: count}.
# ---------------------------------------------------------------------------
# R1: (module under srt/, scope, kind).
_SCHEDULER_REFS = {
    # The supervisor and handlers of a PD role switch drive the scheduler.
    ("disaggregation/role_switch.py", "", "import"): 1,
    ("disaggregation/role_switch.py", "_reject_reason", "annotation"): 1,
    ("disaggregation/role_switch.py", "handle_pd_role_switch", "annotation"): 1,
    ("disaggregation/role_switch.py", "run_event_loop_supervisor", "annotation"): 2,
    # The actor process that hosts a scheduler.
    ("ray/scheduler_actor.py", "SchedulerActor.__init__", "import"): 1,
}

# R2: (Scheduler method, callee, keyword, kind).
_LAMBDA, _BOUND, _SELF = "lambda", "bound method", "self"
_CALLBACK_SITES = {
    # Stage tables: the loop's own stages.
    ("loop_schedules", "LoopSchedules", "admit_prefill", _BOUND): 1,
    ("loop_schedules", "LoopSchedules", "admit_prefill_after_interval", _BOUND): 1,
    ("loop_schedules", "LoopSchedules", "admit_dllm", _BOUND): 1,
    ("loop_schedules", "LoopSchedules", "select_decode", _BOUND): 1,
    ("loop_schedules", "LoopSchedules", "ring_intake_disagg_prefill", _BOUND): 1,
    ("loop_schedules", "LoopSchedules", "ring_release_disagg_prefill", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "ingest_requests", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "launch_step", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "commit_step", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "on_idle", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "mark_idle", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "paused_pass", _BOUND): 1,
    ("init_pp_ring", "DriverOps", "schedule_step", _BOUND): 1,
    # RPC dispatch and the PD role switch, which name the scheduler itself.
    ("handle_rpc_request", "getattr", "", _SELF): 1,
    ("handle_pd_role_switch", "role_switch.handle_pd_role_switch", "", _SELF): 1,
    ("run_event_loop", "dispatch_event_loop", "", _SELF): 1,
    # The scripted runtime's test hook.
    ("__init__", "hook.attach", "", _SELF): 1,
}

# R3: (module under srt/, scope, attribute chain) reaching a node through the
# workers or their model runner.
_WORKER_ROOTS = (
    "tp_worker",
    "model_worker",
    "draft_worker",
    "target_worker",
    "model_runner",
)
_THROUGH_WORKERS = {
    # The model runner builds the coordinator before graph capture.
    (
        "managers/scheduler_assembly.py",
        "init_hisparse_coordinator",
        "tp_worker.model_runner.hisparse_coordinator",
    ): 1,
}

# R4: (module under srt/, class, parameter).
_HUB_PARAMS_BELOW_MODE_OWNERS = {}


# ---------------------------------------------------------------------------
# Scans.
# ---------------------------------------------------------------------------
def _parse(rel: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse((_SRT_DIR / rel).read_text(encoding="utf-8-sig"))


def _scoped_nodes(tree: ast.AST):
    """Yield (node, dotted Class.method scope) for every node."""
    scope_of = {}

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        scope_of[node] = scope
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "")
    return scope_of.items()


def _names_scheduler(annotation: ast.AST) -> bool:
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == "Scheduler":
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode="eval")
            except SyntaxError:
                continue
            if _names_scheduler(inner):
                return True
    return False


def _scan_scheduler_refs() -> Counter:
    """R1: imports and annotations of Scheduler outside the scheduler module."""
    found = Counter()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        if rel == _SCHEDULER:
            continue
        for node, scope in _scoped_nodes(_parse(rel)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "sglang.srt.managers.scheduler"
                and any(alias.name == "Scheduler" for alias in node.names)
            ):
                found[(rel, scope, "import")] += 1
            annotation = None
            if isinstance(node, ast.arg):
                annotation = node.annotation
            elif isinstance(node, ast.AnnAssign):
                annotation = node.annotation
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = node.returns
            if annotation is not None and _names_scheduler(annotation):
                found[(rel, scope, "annotation")] += 1
    return found


def _scheduler_class() -> ast.ClassDef:
    return next(
        node
        for node in _parse(_SCHEDULER).body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )


def _scan_callback_sites() -> Counter:
    """R2: callables and self that Scheduler methods pass into calls."""
    cls = _scheduler_class()
    # A property read passes a value, not a callable.
    methods = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "property" not in map(ast.unparse, node.decorator_list)
    }

    def is_self(node):
        return isinstance(node, ast.Name) and node.id == "self"

    found = Counter()
    for method in cls.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(method):
            if not isinstance(call, ast.Call):
                continue
            callee = ast.unparse(call.func)
            args = [("", arg) for arg in call.args]
            args += [(kw.arg or "", kw.value) for kw in call.keywords]
            for keyword, arg in args:
                if isinstance(arg, ast.Lambda) and any(
                    is_self(node) for node in ast.walk(arg)
                ):
                    kind = _LAMBDA
                elif is_self(arg):
                    kind = _SELF
                elif (
                    isinstance(arg, ast.Attribute)
                    and is_self(arg.value)
                    and arg.attr in methods
                ):
                    kind = _BOUND
                else:
                    continue
                found[(method.name, callee, keyword, kind)] += 1
    return found


def _constructor_params(rel: str, class_name: str) -> list:
    """The __init__ parameters of a class, or its dataclass fields."""
    cls = next(
        node
        for node in ast.walk(_parse(rel))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args
            return [arg.arg for arg in args.args[1:] + args.kwonlyargs]
    return [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and "ClassVar" not in ast.unparse(node.annotation)
    ]


def _scanned_modules() -> list:
    """The scheduler, its assembly, the components and the classes below
    ModeOwners."""
    components = _SRT_DIR / "managers" / "scheduler_components"
    return sorted(
        {
            _SCHEDULER,
            _ASSEMBLY,
            *(rel for rel, _ in _BUILT.values()),
            *_BELOW_MODE_OWNERS,
        }
        | {path.relative_to(_SRT_DIR).as_posix() for path in components.glob("*.py")}
    )


def _scan_through_workers() -> Counter:
    """R3: attribute chains that start at a worker and end at a node's parameter."""
    targets = {name for name, node in _node_of_param().items() if node != "Workers"}
    found = Counter()
    for rel in _scanned_modules():
        for node, scope in _scoped_nodes(_parse(rel)):
            if not isinstance(node, ast.Attribute) or node.attr not in targets:
                continue
            names, cur = [], node.value
            while isinstance(cur, ast.Attribute):
                names.insert(0, cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name):
                continue
            if cur.id != "self":
                names.insert(0, cur.id)
            if names and names[0] in _WORKER_ROOTS:
                found[(rel, scope, ast.unparse(node))] += 1
    return found


def _node_of_param() -> dict:
    return {name: node for node, names in _PARAM.items() for name in names}


def _below(node: str) -> set:
    seen, stack = set(), list(_DECLARED[node])
    while stack:
        dep = stack.pop()
        if dep not in seen:
            seen.add(dep)
            stack.extend(_DECLARED[dep])
    return seen


def _hub() -> set:
    """ModeOwners and every node that holds it."""
    return {"ModeOwners"} | {n for n in _DECLARED if "ModeOwners" in _below(n)}


def _scan_hub_params_below_mode_owners() -> Counter:
    """R4: hub-level parameters taken by classes below ModeOwners."""
    hub_params = {name for node in _hub() for name in _PARAM.get(node, ())}
    hub_params.add("scheduler")
    found = Counter()
    for rel in _BELOW_MODE_OWNERS:
        for cls in ast.walk(_parse(rel)):
            if not isinstance(cls, ast.ClassDef):
                continue
            for param in _constructor_params(rel, cls.name):
                if param in hub_params:
                    found[(rel, cls.name, param)] += 1
    return found


def _diff(found: Counter, allow: dict, what: str) -> str:
    allow = Counter(allow)
    new, gone = found - allow, allow - found
    msg = []
    if new:
        msg.append(
            f"New {what} beyond the allowlist:\n  "
            + "\n  ".join(f"{site} x{n}" for site, n in sorted(new.items()))
        )
    if gone:
        msg.append(
            f"Allowlisted {what} no longer present (delete the entries):\n  "
            + "\n  ".join(f"{site} x{n}" for site, n in sorted(gone.items()))
        )
    return "\n\n".join(msg)


class TestComponentGraph(CustomTestCase):
    def test_r1_no_component_references_scheduler(self):
        msg = _diff(_scan_scheduler_refs(), _SCHEDULER_REFS, "Scheduler reference(s)")
        self.assertFalse(msg, msg)

    def test_r2_no_callback_into_scheduler(self):
        msg = _diff(
            _scan_callback_sites(), _CALLBACK_SITES, "scheduler callback site(s)"
        )
        self.assertFalse(msg, msg)

    def test_r3_declared_graph_is_acyclic(self):
        state = {}

        def visit(node, path):
            state[node] = "open"
            for dep in _DECLARED[node]:
                self.assertIn(dep, _DECLARED, f"{node} takes undeclared {dep}")
                self.assertNotEqual(
                    state.get(dep), "open", f"cycle: {' -> '.join(path + [dep])}"
                )
                if dep not in state:
                    visit(dep, path + [dep])
            state[node] = "done"

        for node in sorted(_DECLARED):
            if node not in state:
                visit(node, [node])

    def test_r3_built_components_take_their_declared_deps(self):
        node_of = _node_of_param()
        for node, (rel, class_name) in sorted(_BUILT.items()):
            params = _constructor_params(rel, class_name)
            taken = {node_of[p] for p in params if p in node_of}
            self.assertEqual(
                taken, set(_DECLARED[node]), f"{class_name} ({rel}) dependencies"
            )

    def test_r3_no_node_reached_through_the_workers(self):
        msg = _diff(
            _scan_through_workers(), _THROUGH_WORKERS, "node(s) reached through workers"
        )
        self.assertFalse(msg, msg)

    def test_r4_nothing_below_mode_owners_holds_the_hub(self):
        self.assertFalse(_below("ModeOwners") & _hub())
        msg = _diff(
            _scan_hub_params_below_mode_owners(),
            _HUB_PARAMS_BELOW_MODE_OWNERS,
            "hub parameter(s) below ModeOwners",
        )
        self.assertFalse(msg, msg)


if __name__ == "__main__":
    unittest.main()
