import ast
import pathlib
import textwrap
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
IO_STRUCT_PATH = REPO_ROOT / "python/sglang/srt/managers/io_struct.py"
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
CONTROLLER_PATH = REPO_ROOT / "scripts/playground/disaggregation/pd_flip_controller.py"


def _load_top_level_definition(path, name, namespace):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definition = next(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@dataclass
class MigrationOutput:
    success: bool
    message: str
    status: Dict[str, Any]
    manifests: Optional[List[Dict[str, Any]]] = None


class DisaggregationMode:
    @staticmethod
    def to_engine_type(value):
        return value


def _load_scheduler_method(name):
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        (
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if method is None:
        raise AttributeError(f"Scheduler has no method {name}")
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Req": object,
        "PDFlipMigrationSourceStartReq": object,
        "PDFlipMigrationReqOutput": MigrationOutput,
        "DisaggregationMode": DisaggregationMode,
        "time": time,
    }
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    return namespace[name]


class BaseReq:
    pass


PDFlipMigrationSourceStartReq = _load_top_level_definition(
    IO_STRUCT_PATH,
    "PDFlipMigrationSourceStartReq",
    {
        "dataclass": dataclass,
        "BaseReq": BaseReq,
        "List": List,
        "Optional": Optional,
    },
)


class FakeReq:
    def __init__(
        self,
        rid,
        *,
        finished=False,
        req_pool_idx=0,
        kv_committed_len=1,
    ):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = kv_committed_len
        self._finished = finished

    def finished(self):
        return self._finished


def make_scheduler(*, running=(), waiting=()):
    scheduler = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=list(running)),
        waiting_queue=list(waiting),
    )
    for method_name in (
        "_pd_flip_waiting_req_skip_reason",
        "_pd_flip_classify_waiting_reqs",
        "_pd_flip_select_source_batch",
    ):
        setattr(
            scheduler,
            method_name,
            types.MethodType(_load_scheduler_method(method_name), scheduler),
        )
    return scheduler


def test_source_start_request_contract_defaults_to_running_only():
    request = PDFlipMigrationSourceStartReq()

    assert request.rids is None
    assert request.include_waiting is False
    assert not hasattr(request, "max_reqs")


def test_source_start_selects_exact_unfinished_running_prefix():
    request = PDFlipMigrationSourceStartReq(rids=["r0", "r1"], include_waiting=False)
    scheduler = make_scheduler(
        running=[
            FakeReq("r0"),
            FakeReq("finished", finished=True),
            FakeReq("r1"),
            FakeReq("r2"),
        ]
    )

    selected = scheduler._pd_flip_select_source_batch(request)

    assert [req.rid for req in selected] == ["r0", "r1"]


def test_source_start_distinguishes_empty_rids_from_all_running_compatibility():
    scheduler = make_scheduler(running=[FakeReq("r0"), FakeReq("r1")])

    empty = scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=[])
    )
    all_running = scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=None)
    )

    assert empty == []
    assert [req.rid for req in all_running] == ["r0", "r1"]


def test_source_start_builds_manifests_for_only_the_selected_running_prefix():
    running = [FakeReq("r0"), FakeReq("r1"), FakeReq("r2")]
    scheduler = make_scheduler(running=running)
    scheduler.disaggregation_mode = types.SimpleNamespace(value="decode")
    scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
    scheduler.ps = types.SimpleNamespace(attn_dp_rank=0, dp_rank=0)
    scheduler._pd_flip_migration_status_dict = lambda: {}
    scheduler._pd_flip_build_migration_manifest = lambda req: {"rid": req.rid}
    scheduler._pd_flip_migration_room_for_req = lambda req: f"room-{req.rid}"
    captured = {}

    def start_entries(reqs, manifests):
        captured["reqs"] = reqs
        captured["manifests"] = manifests
        return {}, ""

    scheduler._pd_flip_start_source_entries = start_entries
    start = types.MethodType(
        _load_scheduler_method("start_pd_flip_migration_source"), scheduler
    )

    output = start(PDFlipMigrationSourceStartReq(rids=["r0", "r1"]))

    assert output.success
    assert [req.rid for req in captured["reqs"]] == ["r0", "r1"]
    assert [manifest["rid"] for manifest in captured["manifests"]] == [
        "r0",
        "r1",
    ]


def test_source_start_rejects_non_prefix_or_out_of_order_selection():
    scheduler = make_scheduler(running=[FakeReq("r0"), FakeReq("r1")])

    for rids in (["r1"], ["r1", "r0"], ["r0", "missing"]):
        with pytest.raises(ValueError, match="running-batch prefix"):
            scheduler._pd_flip_select_source_batch(
                PDFlipMigrationSourceStartReq(rids=rids)
            )


def test_source_start_rejects_duplicate_requested_running_rid():
    scheduler = make_scheduler(running=[FakeReq("r0"), FakeReq("r1")])

    with pytest.raises(ValueError, match="duplicate rid.*r0"):
        scheduler._pd_flip_select_source_batch(
            PDFlipMigrationSourceStartReq(rids=["r0", "r0"])
        )


def test_source_start_with_none_selects_all_unfinished_running_requests():
    scheduler = make_scheduler(
        running=[FakeReq("r0"), FakeReq("done", finished=True), FakeReq(2)]
    )

    selected = scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=None)
    )

    assert [req.rid for req in selected] == ["r0", 2]


def test_source_start_excludes_waiting_by_default():
    scheduler = make_scheduler(running=[FakeReq("r0")], waiting=[FakeReq("waiting")])

    selected = scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=["r0"])
    )

    assert [req.rid for req in selected] == ["r0"]


def test_source_start_includes_migratable_waiting_and_ignores_finished_waiting():
    scheduler = make_scheduler(
        running=[FakeReq("r0")],
        waiting=[FakeReq("waiting"), FakeReq("finished", finished=True)],
    )

    selected = scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=["r0"], include_waiting=True)
    )

    assert [req.rid for req in selected] == ["r0", "waiting"]


@pytest.mark.parametrize(
    "running, waiting",
    [
        ([FakeReq("same")], [FakeReq("same")]),
        ([], [FakeReq("same"), FakeReq("same")]),
    ],
    ids=["running-waiting", "within-waiting"],
)
def test_source_start_rejects_duplicate_rid_in_combined_selection(running, waiting):
    scheduler = make_scheduler(running=running, waiting=waiting)

    with pytest.raises(ValueError, match="duplicate rid.*same"):
        scheduler._pd_flip_select_source_batch(
            PDFlipMigrationSourceStartReq(rids=None, include_waiting=True)
        )


def test_source_selection_does_not_modify_waiting_queue():
    waiting = [FakeReq("waiting"), FakeReq("finished", finished=True)]
    scheduler = make_scheduler(running=[FakeReq("r0")], waiting=waiting)

    scheduler._pd_flip_select_source_batch(
        PDFlipMigrationSourceStartReq(rids=["r0"], include_waiting=True)
    )

    assert len(scheduler.waiting_queue) == 2
    assert all(
        actual is expected for actual, expected in zip(scheduler.waiting_queue, waiting)
    )


def test_source_start_rejects_nonfinished_skipped_waiting_request():
    scheduler = make_scheduler(
        running=[FakeReq("r0")],
        waiting=[FakeReq("missing-pool", req_pool_idx=None)],
    )

    with pytest.raises(ValueError, match="remaining waiting requests"):
        scheduler._pd_flip_select_source_batch(
            PDFlipMigrationSourceStartReq(rids=["r0"], include_waiting=True)
        )


def test_source_start_rejects_duplicate_rids_before_manifest_or_sender_state():
    scheduler = make_scheduler(running=[FakeReq("r0"), FakeReq("r1")])
    scheduler.disaggregation_mode = types.SimpleNamespace(value="decode")
    scheduler._pd_flip_migration_status_dict = lambda: {}
    scheduler._pd_flip_build_migration_manifest = lambda req: pytest.fail(
        "duplicate selection must fail before manifest creation"
    )
    scheduler._pd_flip_start_source_entries = lambda reqs, manifests: pytest.fail(
        "duplicate selection must fail before sender creation"
    )
    start = types.MethodType(
        _load_scheduler_method("start_pd_flip_migration_source"), scheduler
    )

    output = start(PDFlipMigrationSourceStartReq(rids=["r0", "r0"]))

    assert not output.success
    assert "duplicate rid" in output.message
    assert not hasattr(scheduler, "pd_flip_migration_session")


def test_source_start_classifies_waiting_once_and_uses_same_identity_snapshot():
    finished = FakeReq("finished", finished=True)
    selected_waiting = FakeReq("waiting")
    drift = FakeReq("drift")
    scheduler = make_scheduler(running=[], waiting=[finished, selected_waiting])
    scheduler.disaggregation_mode = types.SimpleNamespace(value="decode")
    scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
    scheduler.ps = types.SimpleNamespace(attn_dp_rank=0, dp_rank=0)
    scheduler._pd_flip_migration_status_dict = lambda: {}
    scheduler._pd_flip_build_migration_manifest = lambda req: {"rid": req.rid}
    scheduler._pd_flip_migration_room_for_req = lambda req: f"room-{req.rid}"
    classify_inputs = []

    def classify(waiting_snapshot):
        classify_inputs.append(list(waiting_snapshot))
        if len(classify_inputs) > 1:
            return [(0, drift)], []
        scheduler.waiting_queue[:] = [drift]
        return [(1, selected_waiting)], [
            {"rid": "finished", "queue_index": 0, "reason": "finished"}
        ]

    scheduler._pd_flip_classify_waiting_reqs = classify
    captured = {}

    def start_entries(reqs, manifests):
        captured["reqs"] = reqs
        captured["manifests"] = manifests
        return {}, ""

    scheduler._pd_flip_start_source_entries = start_entries
    start = types.MethodType(
        _load_scheduler_method("start_pd_flip_migration_source"), scheduler
    )

    output = start(PDFlipMigrationSourceStartReq(rids=[], include_waiting=True))

    assert output.success
    assert len(classify_inputs) == 1
    assert classify_inputs[0][0] is finished
    assert classify_inputs[0][1] is selected_waiting
    assert captured["reqs"] == [selected_waiting]
    assert captured["manifests"][0]["rid"] == "waiting"
    assert captured["manifests"][0]["pd_flip_waiting_queue_index"] == 1


def test_controller_source_start_payload_sends_exact_rids():
    payload_builder = _load_top_level_definition(
        CONTROLLER_PATH,
        "_migration_source_start_payload",
        {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional},
    )

    assert payload_builder(
        "session-1", "http://target", ("r0", "r1"), include_waiting=True
    ) == {
        "session_id": "session-1",
        "target_url": "http://target",
        "rids": ["r0", "r1"],
        "include_waiting": True,
    }


def test_controller_source_start_payload_preserves_all_running_compatibility():
    payload_builder = _load_top_level_definition(
        CONTROLLER_PATH,
        "_migration_source_start_payload",
        {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional},
    )

    assert payload_builder("session-1", "http://target", None) == {
        "session_id": "session-1",
        "target_url": "http://target",
        "rids": None,
        "include_waiting": False,
    }
