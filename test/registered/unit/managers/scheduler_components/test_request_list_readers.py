"""Who reads the scheduler's request lists.

Every question about the live requests (abort, idle, pause, load, metrics,
LoRA, invariant checks) is asked of `RequestRegistry`, through the flags each
owner declares on its `RequestList`s; a reader that picks lists itself drifts
from the others as lists are added.

- L1: a request list field (one `lists()` registers) is read only by the
  registry, by its owner, and by the loop's scheduling code named below.
- L2: outside the registry nobody selects a `RequestList` by its name.
- L3: every field of a list owner that holds requests is a registered list,
  so no request hides from the registry's readers.

The allowlists record the sites still to be removed: a new site fails, and a
removed site fails until its entry is deleted, so they only shrink.
"""

import ast
import functools
import re
import unittest
import warnings
from collections import Counter
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_REGISTRY = "managers/scheduler_components/request_registry.py"
assert (_SRT_DIR / _REGISTRY).is_file(), f"registry not found under {_SRT_DIR}"

# ---------------------------------------------------------------------------
# The list owners: each declares its lists in `lists()` (the registry declares
# the core lists and the grammar queue); the classes of one owner read each
# other's lists freely.
# ---------------------------------------------------------------------------
# owner -> (module under srt/, the class whose method declares its lists,
# the classes that make up the owner).
_OWNERS = {
    "core": (
        _REGISTRY,
        "RequestRegistry.request_lists",
        {"managers/request_lists.py": ["CoreLists"]},
    ),
    "grammar": (
        _REGISTRY,
        "RequestRegistry.request_lists",
        {"constrained/grammar_manager.py": ["GrammarManager"]},
    ),
    "disagg_prefill": (
        "disaggregation/prefill.py",
        "DisaggPrefill.lists",
        {
            "disaggregation/prefill.py": [
                "DisaggPrefill",
                "PrefillBootstrapQueue",
                "_HandoffOutcome",
            ]
        },
    ),
    "disagg_decode": (
        "disaggregation/decode.py",
        "DisaggDecode.lists",
        {
            "disaggregation/decode.py": [
                "DisaggDecode",
                "DecodePreallocQueue",
                "DecodeTransferQueue",
            ],
            "disaggregation/decode_hicache_mixin.py": [
                "DecodeHiCachePreallocMixin",
                "DecodeHiCacheTransferMixin",
            ],
        },
    ),
    "dllm": (
        "dllm/manager.py",
        "DllmManager.lists",
        {"dllm/manager.py": ["DllmManager"]},
    ),
    "hisparse": (
        "managers/hisparse_coordinator.py",
        "HiSparseCoordinator.lists",
        {"managers/hisparse_coordinator.py": ["HiSparseCoordinator"]},
    ),
}

# The class each attribute of a declaring class names, for the lists declared
# through it: (declaring class, attribute) -> class.
_HOLDER_OF = {
    ("RequestRegistry", "core_lists"): "CoreLists",
    ("RequestRegistry", "grammar_manager"): "GrammarManager",
    ("DisaggPrefill", "bootstrap_queue"): "PrefillBootstrapQueue",
    ("DisaggDecode", "prealloc_queue"): "DecodePreallocQueue",
    ("DisaggDecode", "transfer_queue"): "DecodeTransferQueue",
}

# The mode owners `ModeOwners` holds; the registry takes each one's lists.
_MODE_OWNER_FIELDS = {
    "disagg_prefill": "disagg_prefill",
    "disagg_decode": "disagg_decode",
    "dllm": "dllm",
    "hisparse_coordinator": "hisparse",
}

# ---------------------------------------------------------------------------
# L1: the loop's scheduling code, which reads the lists it schedules from:
# (module under srt/, Class.method).
# ---------------------------------------------------------------------------
_LOOP_SCHEDULING = {
    # The loop: intake, schedule, launch, commit, and the chunked request's
    # abort and stash, in each mode.
    ("managers/scheduler.py", "Scheduler.run_flat_loop"),
    ("managers/scheduler.py", "Scheduler.join_launched_step"),
    ("managers/scheduler.py", "Scheduler.leave_finished_reqs"),
    ("managers/scheduler.py", "Scheduler.process_pending_chunked_abort"),
    ("managers/scheduler.py", "Scheduler.stash_chunked_request"),
    ("managers/scheduler.py", "Scheduler.schedule_step"),
    ("managers/scheduler.py", "Scheduler._admit_dllm_step"),
    ("managers/scheduler.py", "Scheduler.process_prefill_transfer_queue"),
    ("managers/scheduler.py", "Scheduler.process_batch_result"),
    # Prefill admission (and the interval that defers it), decode selection
    # and retraction.
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner.should_defer_prefill",
    ),
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner.admission_skip_reason",
    ),
    ("managers/scheduler_components/step_planner.py", "StepPlanner._room_for_one"),
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner._get_new_batch_prefill_raw",
    ),
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner._select_decode_reqs",
    ),
    ("managers/scheduler_components/step_planner.py", "StepPlanner.get_decode_step"),
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner._retract_decode_reqs",
    ),
    # PD decode's admission stages budget against the decoding requests.
    ("disaggregation/decode.py", "DisaggDecode.intake"),
    ("disaggregation/decode.py", "DisaggDecode.apply_retract_consensus"),
    ("disaggregation/decode.py", "DisaggDecode.apply_prealloc_consensus"),
    ("disaggregation/decode.py", "DisaggDecode.apply_transfer_consensus"),
    ("disaggregation/decode.py", "DisaggDecode.pop_reqs_to_admit"),
    # PD prefill's extend commit tells the request still chunking from a
    # yielded one.
    ("disaggregation/prefill.py", "_HandoffOutcome.__init__"),
}

# L1 allowlist: (module under srt/, scope, field read) -> count.
_DIRECT_READS = {
    # Admission budget: the rows the decoding beam groups still claim.
    (
        "beam_search/coordinator.py",
        "BeamCoordinator.row_budget",
        "CoreLists.decoding",
    ): 1,
    # Writers of the core waiting queue (the queue-limit door evicts from it;
    # the L3 retries follow its order); they go with the loop as the only
    # writer.
    (
        "managers/scheduler_components/request_intake.py",
        "RequestIntake._abort_on_queued_limit",
        "CoreLists.waiting_queue",
    ): 1,
    (
        "managers/scheduler_components/request_intake.py",
        "RequestIntake.process_storage_prefetch_retries",
        "CoreLists.waiting_queue",
    ): 1,
}

# L2 allowlist: (module under srt/, scope, list name) -> count.
_NAME_LOOKUPS = {
    # A writer: the retract half of pause clears the chunked holder when the
    # chunked list retracts; it goes with the loop as the only writer.
    (
        "managers/scheduler_components/step_planner.py",
        "StepPlanner.retract_for_pause",
        "chunked",
    ): 1,
}

# L3: owner fields that hold requests but are no list: (class, field) -> why.
_NOT_LISTS = {
    ("CoreLists", "_pending_chunked_abort_req"): (
        "marks the chunked request, which the chunked list holds, for abort"
    ),
    ("DecodePreallocQueue", "pending_reqs"): (
        "the members of `queue` still resolving their prefill DP rank"
    ),
    # One PD prefill commit's rows, sorted while the commit runs.
    ("_HandoffOutcome", "concluded"): "the commit's rows that leave prefilling",
    ("_HandoffOutcome", "aborted"): "the commit's rows it retired, to stream",
    ("_HandoffOutcome", "requeued"): "the commit's rows that yield to waiting",
}

# L3: unannotated container fields of the owners that hold no requests; any
# other must be annotated, so the scan sees what it holds.
_UNANNOTATED_CONTAINERS = {
    ("GrammarManager", "grammar_pp_sync_work_list"),
    ("HiSparseCoordinator", "_is_shared_index_layer"),
    ("HiSparseCoordinator", "_prefetch_events"),
}


# ---------------------------------------------------------------------------
# Scans.
# ---------------------------------------------------------------------------
_PARSED = {}


def _parse(rel: str) -> ast.Module:
    if rel not in _PARSED:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            _PARSED[rel] = ast.parse((_SRT_DIR / rel).read_text(encoding="utf-8-sig"))
    return _PARSED[rel]


def _scoped_nodes(tree: ast.AST):
    """Yield (node, dotted Class.method scope, enclosing class) for every node."""
    out = []

    def visit(node, scope, cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
            if isinstance(node, ast.ClassDef):
                cls = node.name
        out.append((node, scope, cls))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, cls)

    visit(tree, "", None)
    return out


def _function(rel: str, dotted: str) -> ast.FunctionDef:
    cls_name, func_name = dotted.split(".")
    for node in ast.walk(_parse(rel)):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func_name:
                    return item
    raise AssertionError(f"{dotted} not found in {rel}")


def _chain(node: ast.AST):
    """`a.b.c` as ['a', 'b', 'c'], or None if it is not a plain chain."""
    names = []
    while isinstance(node, ast.Attribute):
        names.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id] + names


@functools.lru_cache(maxsize=None)
def _declared_lists():
    """The registered list fields, {(class, field): owner}, and the list
    names, from each declaring method's `RequestList(reqs=...)` calls."""
    fields, names = {}, set()
    for owner, (rel, dotted, _) in _OWNERS.items():
        func = _function(rel, dotted)
        declaring_cls = dotted.split(".")[0]
        # Local aliases of self's attributes (`queue = self.prealloc_queue`).
        aliases = {}
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and (chain := _chain(node.value)) is not None
                and chain[0] == "self"
            ):
                aliases[node.targets[0].id] = chain[1:]
        for call in ast.walk(func):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "RequestList"
            ):
                continue
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            names.add(kwargs["name"].value)
            for node in ast.walk(kwargs["reqs"]):
                chain = _chain(node) if isinstance(node, ast.Attribute) else None
                if chain is None:
                    continue
                if chain[0] == "self":
                    path = chain[1:]
                elif chain[0] in aliases:
                    path = aliases[chain[0]] + chain[1:]
                else:
                    continue
                holder = _HOLDER_OF.get((declaring_cls, path[0]))
                if holder is not None and len(path) == 2:
                    field = (holder, path[1])
                elif holder is None and len(path) == 1:
                    field = (declaring_cls, path[0])
                else:
                    continue
                if _owner_of_class(field[0]) == owner:
                    fields[field] = owner
    return fields, names


def _owner_of_class(cls_name: str):
    for owner, (_, _, classes) in _OWNERS.items():
        if any(cls_name in names for names in classes.values()):
            return owner
    return None


@functools.lru_cache(maxsize=None)
def _annotated_handles() -> dict:
    """The names that hold each owner class across srt: parameters, fields and
    `self.<name>` attributes annotated with it, and the registry's own
    handles."""
    handles = {"CoreLists": {"core_lists", "lists"}}
    classes = {
        cls
        for _, _, by_rel in _OWNERS.values()
        for names in by_rel.values()
        for cls in names
    }
    pattern = re.compile(r"\b(" + "|".join(sorted(classes)) + r")\b")
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        for node in ast.walk(_parse(rel)):
            if isinstance(node, ast.arg) and node.annotation is not None:
                name, annotation = node.arg, node.annotation
            elif isinstance(node, ast.AnnAssign):
                chain = _chain(node.target)
                if chain is None:
                    continue
                name, annotation = chain[-1], node.annotation
            else:
                continue
            for cls in pattern.findall(ast.unparse(annotation)):
                handles.setdefault(cls, set()).add(name)
    return handles


def _is_mutation(node: ast.Attribute, parents: dict) -> bool:
    """A write through the field (`x.f = ...`, `x.f.append(...)`), not a read."""
    if not isinstance(node.ctx, ast.Load):
        return True
    parent = parents.get(node)
    if (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and parent.attr in ("append", "extend", "remove", "insert", "clear", "pop")
        and isinstance(parents.get(parent), ast.Call)
    ):
        return True
    return False


def _scan_direct_reads(fields: dict, loop_scheduling=_LOOP_SCHEDULING) -> Counter:
    """L1: reads of a registered field outside the registry, its owner and the
    loop's scheduling code."""
    handles = _annotated_handles()
    by_name = {}
    for (holder, field), owner in fields.items():
        by_name.setdefault(field, []).append((holder, owner))
    found = Counter()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        if rel == _REGISTRY:
            continue
        tree = _parse(rel)
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node, scope, cls in _scoped_nodes(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in by_name:
                continue
            if _is_mutation(node, parents):
                continue
            receiver = _chain(node.value)
            if receiver is None:
                continue
            for holder, owner in by_name[node.attr]:
                if receiver == ["self"]:
                    matched = cls == holder
                else:
                    matched = receiver[-1] in handles.get(holder, ())
                if not matched:
                    continue
                if _owner_of_class(cls) == owner or (rel, scope) in loop_scheduling:
                    continue
                found[(rel, scope, f"{holder}.{node.attr}")] += 1
    return found


def _scan_name_lookups(names: set) -> Counter:
    """L2: a list picked by its name: a subscript or a comparison with it."""
    found = Counter()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        if rel == _REGISTRY:
            continue
        source = (_SRT_DIR / rel).read_text(encoding="utf-8-sig")
        if "request_lists" not in source and "RequestList" not in source:
            continue
        for node, scope, _ in _scoped_nodes(_parse(rel)):
            if isinstance(node, ast.Subscript):
                key = node.slice
            elif isinstance(node, ast.Compare) and any(
                isinstance(side, ast.Attribute) and side.attr == "name"
                for side in [node.left, *node.comparators]
            ):
                key = next(
                    (
                        s
                        for s in [node.left, *node.comparators]
                        if isinstance(s, ast.Constant)
                    ),
                    None,
                )
            else:
                continue
            if isinstance(key, ast.Constant) and key.value in names:
                found[(rel, scope, key.value)] += 1
    return found


def _owner_fields():
    """L3: {(class, field): annotation or None} of each owner class's `self`
    fields that are annotated or start as a container."""
    out = {}
    for _, _, by_rel in _OWNERS.values():
        for rel, classes in by_rel.items():
            for cls in ast.walk(_parse(rel)):
                if not isinstance(cls, ast.ClassDef) or cls.name not in classes:
                    continue
                for node in ast.walk(cls):
                    if isinstance(node, ast.AnnAssign):
                        chain = _chain(node.target)
                        if chain is not None and chain[0] == "self" and len(chain) == 2:
                            out[(cls.name, chain[1])] = ast.unparse(node.annotation)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            chain = _chain(target)
                            if chain is None or chain[0] != "self" or len(chain) != 2:
                                continue
                            value = node.value
                            container = isinstance(
                                value, (ast.List, ast.ListComp, ast.Dict, ast.Set)
                            ) or (
                                isinstance(value, ast.Call)
                                and isinstance(value.func, ast.Name)
                                and value.func.id
                                in ("list", "dict", "set", "deque", "defaultdict")
                            )
                            if container:
                                out.setdefault((cls.name, chain[1]), None)
    return out


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


class TestRequestListReaders(CustomTestCase):
    def test_l1_only_owners_and_the_loop_read_the_lists(self):
        fields, _ = _declared_lists()
        msg = _diff(
            _scan_direct_reads(fields), _DIRECT_READS, "direct request list read(s)"
        )
        self.assertFalse(msg, msg)

    def test_l1_loop_scheduling_scopes_read_the_lists(self):
        fields, _ = _declared_lists()
        reading = {(rel, scope) for rel, scope, _ in _scan_direct_reads(fields, ())}
        unused = _LOOP_SCHEDULING - reading
        self.assertFalse(unused, f"Loop scopes that read no list (delete): {unused}")

    def test_l2_no_list_picked_by_name(self):
        _, names = _declared_lists()
        msg = _diff(_scan_name_lookups(names), _NAME_LOOKUPS, "list name lookup(s)")
        self.assertFalse(msg, msg)

    def test_l3_every_request_field_of_an_owner_is_a_list(self):
        fields, _ = _declared_lists()
        holds_reqs = re.compile(r"\b(Req|DecodeRequest)\b")
        unregistered, unannotated = [], []
        for field, annotation in sorted(_owner_fields().items()):
            if annotation is None:
                if field not in _UNANNOTATED_CONTAINERS:
                    unannotated.append(field)
            elif holds_reqs.search(annotation):
                if field not in fields and field not in _NOT_LISTS:
                    unregistered.append((field, annotation))
        self.assertFalse(
            unregistered,
            f"Owner fields holding requests that no RequestList registers: "
            f"{unregistered}",
        )
        self.assertFalse(
            unannotated,
            f"Unannotated container fields of a list owner (annotate them): "
            f"{unannotated}",
        )
        stale = (set(_NOT_LISTS) | _UNANNOTATED_CONTAINERS) - set(_owner_fields())
        self.assertFalse(stale, f"Allowlisted fields no longer present: {stale}")

    def test_l3_the_registry_takes_every_mode_owners_lists(self):
        fields = _constructor_fields("managers/scheduler_components/mode_owners.py")
        self.assertEqual(set(fields), set(_MODE_OWNER_FIELDS))
        source = ast.unparse(_function(_REGISTRY, "RequestRegistry.request_lists"))
        for field, owner in _MODE_OWNER_FIELDS.items():
            self.assertIn(f"self.mode_owners.{field}.lists()", source)
            self.assertIn(owner, _OWNERS)


def _constructor_fields(rel: str) -> list:
    cls = next(
        node
        for node in _parse(rel).body
        if isinstance(node, ast.ClassDef) and node.name == "ModeOwners"
    )
    return [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


if __name__ == "__main__":
    unittest.main()
