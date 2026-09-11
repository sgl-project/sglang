"""CPU admission tests, loading production methods without GPU runtime imports."""

import ast
import logging
import runpy
import threading
import time
from abc import ABC
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import msgspec
import pytest

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load(namespace, path, selections):
    tree = ast.parse((SRT / path).read_text())
    nodes = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if node.name not in selections:
            continue
        if isinstance(node, ast.ClassDef) and selections[node.name] is not None:
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name in selections[node.name]
            ] or [ast.Pass()]
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                child.decorator_list = [
                    d
                    for d in child.decorator_list
                    if isinstance(d, ast.Name) and d.id in ("staticmethod", "property")
                ]
                # FINISH_ABORT is supplied from the real schedule_batch AST.
                child.body = [
                    statement
                    for statement in child.body
                    if not isinstance(statement, ast.ImportFrom)
                ]
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), namespace)


@pytest.fixture
def code():
    ns = {
        "ABC": ABC,
        "HTTPStatus": HTTPStatus,
        "array": array,
        "logger": logging.getLogger(__name__),
        "time": time,
        "get_parallel": lambda: SimpleNamespace(tp_rank=0, dp_size=1),
        "get_disagg": lambda: SimpleNamespace(optimistic_prefill_attempts=2),
        "should_force_retry": lambda req: False,
        "release_kv_cache": Mock(),
        "StagingManagerMixin": type("StagingManagerMixin", (), {}),
        "ReqDllmMixin": type("ReqDllmMixin", (), {}),
        "DecodeHiCachePreallocMixin": type("DecodeHiCachePreallocMixin", (), {}),
        "_is_fake_transfer": lambda req: False,
        "KVTransferMetric": SimpleNamespace,
        "NetworkAddress": lambda host, port: SimpleNamespace(
            to_host_port_str=lambda: f"{host}:{port}"
        ),
    }
    _load(
        ns,
        "managers/schedule_batch.py",
        {
            "BaseFinishReason": None,
            "FINISH_ABORT": None,
            "Req": {
                "set_finish_with_abort",
                "update_finish_state",
                "finished",
                "output_ids_through_stop",
                "init_incremental_detokenize",
            },
        },
    )
    _load(
        ns,
        "disaggregation/base/conn.py",
        {
            "KVPoll": None,
            "BaseKVManager": set(),
            "BaseKVSender": {"abort", "clear", "abort_before_send"},
            "BaseKVReceiver": set(),
        },
    )
    _load(
        ns,
        "disaggregation/utils.py",
        {
            "is_aborted": None,
            "prepare_abort": None,
            "poll_and_all_reduce_pp": None,
        },
    )
    _load(
        ns,
        "disaggregation/common/conn.py",
        {
            "KVTransferError": None,
            "CommonKVManager": {
                "check_status",
                "update_status",
                "record_failure",
                "_prefill_unique_rank",
                "notify_bootstrap_failure",
            },
            "CommonKVSender": {
                "__init__",
                "init",
                "_prepare_send_indices",
                "abort",
                "clear",
                "abort_before_send",
                "_check_bootstrap_timeout",
            },
            "CommonKVReceiver": {"abort", "clear", "_check_waiting_timeout"},
        },
    )
    _load(
        ns,
        "disaggregation/mooncake/conn.py",
        {
            "MooncakeKVManager": {"notify_bootstrap_failure", "start_prefill_thread"},
            "MooncakeFailureExceptionMixin": {"failure_exception"},
            "MooncakeKVSender": {"abort", "poll"},
            "MooncakeKVReceiver": {"poll"},
        },
    )
    _load(
        ns,
        "disaggregation/prefill.py",
        {
            "maybe_release_metadata_buffer": None,
            "PrefillBootstrapQueue": {"pop_bootstrapped"},
            "SchedulerDisaggregationPrefillMixin": {
                "clear_pending_chunk_send",
                "handle_bootstrap_failure",
            },
        },
    )
    _load(
        ns,
        "disaggregation/decode.py",
        {
            "DecodePreallocQueue": {
                "add",
                "pop_preallocated",
                "_update_handshake_waiters",
                "_resolve_pending_reqs",
                "prefetch_prefill_dp_rank_queries",
            },
            "_bootstrap_addr": None,
        },
    )
    _load(
        ns,
        "managers/scheduler_pp_mixin.py",
        {
            "SchedulerPPMixin": {
                "_pp_pd_get_bootstrapped_ids",
                "_pp_pd_get_prealloc_ids",
                "_route_aborts_to_bad",
                "get_rids",
            },
        },
    )
    ns["poll_and_all_reduce"] = lambda pollers, *args: [p.poll() for p in pollers]
    ns["poll_and_all_reduce_attn_cp_tp_group"] = ns["poll_and_all_reduce"]
    return SimpleNamespace(**ns)


def _req(code, rid="rejected"):
    req = code.Req.__new__(code.Req)
    req.rid = rid
    req.bootstrap_room = 42
    req.bootstrap_host = "prefill"
    req.bootstrap_port = 8000
    req.finished_reason = None
    req.finished_output = False
    req.to_finish = None
    req.multimodal_inputs = None
    req.session = None
    req.output_ids = []
    req.origin_input_ids = [1, 2, 3]
    req.time_stats = Mock()
    req.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
    req.metadata_buffer_index = -1
    req.prefill_attempt_count = 0
    req.pending_bootstrap = True
    req.is_retracted = False
    req.set_finish_with_abort("original admission error")
    assert req.finished_reason is None
    assert list(req.origin_input_ids) == [0]
    assert req.to_finish.status_code == 400
    return req


def _manager(code):
    manager = code.MooncakeKVManager.__new__(code.MooncakeKVManager)
    manager.request_status = {42: code.KVPoll.WaitingForInput}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.req_to_decode_prefix_len = {42: 64}
    manager.transfer_infos = {
        42: {
            "a": SimpleNamespace(endpoint="decode-a", dst_port=8001, is_dummy=False),
            "dummy": SimpleNamespace(endpoint="dummy", dst_port=8002, is_dummy=True),
            "b": SimpleNamespace(endpoint="decode-b", dst_port=8003, is_dummy=False),
        }
    }
    manager._deferred_ack_targets = {}
    manager._staging_outstanding = defaultdict(int)
    manager.attn_tp_rank = 1
    manager.attn_cp_rank = 1
    manager.attn_cp_size = 2
    manager.pp_rank = 1
    manager.pp_size = 2
    manager.bootstrap_timeout = 10
    manager.sync_status_to_decode_endpoint = Mock()
    return manager


def _decode_queue(code, req):
    manager = _manager(code)
    manager.required_prefill_response_num_table = {42: 1}
    manager.prefill_response_tracker = {42: set()}
    receiver = code.MooncakeKVReceiver.__new__(code.MooncakeKVReceiver)
    receiver.kv_mgr = manager
    receiver.bootstrap_room = 42
    receiver.conclude_state = None
    receiver.init_time = None
    receiver.abort_notified = False
    receiver.bootstrap_infos = [{}]
    receiver._send_abort_notification = Mock()
    receiver.init = Mock()
    receiver.send_metadata = Mock()
    decode_req = SimpleNamespace(req=req, kv_receiver=receiver, waiting_for_input=False)
    queue = code.DecodePreallocQueue.__new__(code.DecodePreallocQueue)
    queue.queue = [decode_req]
    queue.pending_reqs = [decode_req]
    queue._prefill_dp_rank_queries = {}
    queue._cancel_prefill_dp_rank_queries = Mock()
    queue._ensure_prefill_info = Mock(side_effect=lambda groups: (groups, []))
    queue._resolve_prefill_dp_rank = Mock(return_value=0)
    queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
    queue._create_receiver_and_enqueue = Mock(return_value=decode_req)
    queue._uses_swa_tail_prealloc = Mock(return_value=False)
    queue._allocatable_token_budgets = Mock(return_value=0)
    queue._hicache_pending_restore_tokens = Mock(return_value=0)
    queue._pre_alloc = Mock()
    queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 0)
    queue.pp_size = 1
    queue.tp_rank = 0
    queue.gloo_group = None
    queue.scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[]),
        enable_priority_scheduling=False,
        enable_hisparse=False,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        output_streamer=Mock(),
    )
    return queue, decode_req, receiver


def _prefill_queue(code, req, poll):
    sender = code.MooncakeKVSender.__new__(code.MooncakeKVSender)
    manager = _manager(code)
    manager.enable_all_cp_ranks_for_transfer = False
    manager.is_dummy_cp_rank = False
    code.CommonKVSender.__init__(sender, manager, "prefill:8000", 42, [0], 0)
    sender.kv_mgr.request_status[42] = poll
    sender.init_time = time.time()
    sender.trace_ctx = Mock()
    req.disagg_kv_sender = sender
    scheduler = code.SchedulerDisaggregationPrefillMixin()
    scheduler.ps = SimpleNamespace(tp_rank=0)
    scheduler.disagg_prefill_pending_chunk_rids = set()
    scheduler.tree_cache = Mock()
    scheduler.req_to_metadata_buffer_idx_allocator = Mock()
    scheduler.output_streamer = Mock()
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.enable_hicache_storage = True
    scheduler.attn_cp_cpu_group = None
    scheduler.attn_tp_cpu_group = None
    scheduler.processed_tokens_counter = 0
    queue = code.PrefillBootstrapQueue.__new__(code.PrefillBootstrapQueue)
    queue.queue = [req]
    queue.scheduler = scheduler
    queue.pp_size = 1
    queue.ensure_metadata_buffer = Mock(return_value=True)
    queue.finalize_bootstrap = Mock(return_value=True)
    return queue, sender


@pytest.mark.parametrize("status", [0, 1, 2])
def test_decode_pending_admission_abort_preserves_error_and_cleans_once(code, status):
    req = _req(code)
    original = req.to_finish
    queue, decode_req, receiver = _decode_queue(code, req)
    receiver.kv_mgr.request_status[42] = status

    assert queue.pop_preallocated() == ([], [decode_req])
    assert req.finished_reason is original
    assert req.to_finish is None
    assert queue.queue == queue.pending_reqs == []
    assert decode_req.kv_receiver is None
    receiver.init.assert_not_called()
    receiver.send_metadata.assert_not_called()
    receiver._send_abort_notification.assert_called_once_with()
    queue._pre_alloc.assert_not_called()
    code.release_kv_cache.assert_not_called()
    assert receiver.kv_mgr.request_status == receiver.kv_mgr.failure_records == {}
    assert queue.pop_preallocated() == ([], [])
    queue.scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


def test_decode_add_rejects_without_receiver_or_capacity_check(code):
    req = _req(code)
    original = req.to_finish
    queue, _, _ = _decode_queue(code, req)
    queue.queue = []
    queue.pending_reqs = []
    queue.add(req)
    assert req.finished_reason is original
    queue._check_if_req_exceed_kv_capacity.assert_not_called()
    queue._create_receiver_and_enqueue.assert_not_called()
    queue.scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


@pytest.mark.parametrize("pp", [False, True])
def test_decode_abort_respects_queue_filter(code, pp):
    req = _req(code)
    queue, decode_req, receiver = _decode_queue(code, req)
    queue.pp_size = 2 if pp else 1
    kwargs = {"pp_good_rids": [], "pp_bad_rids": []} if pp else {"rids_to_check": []}
    assert queue.pop_preallocated(**kwargs) == ([], [])
    assert queue.queue == queue.pending_reqs == [decode_req]
    assert req.finished_reason is None
    receiver.init.assert_not_called()
    receiver._send_abort_notification.assert_not_called()
    kwargs = (
        {"pp_good_rids": [], "pp_bad_rids": [req.rid]}
        if pp
        else {"rids_to_check": [req.rid]}
    )
    assert queue.pop_preallocated(**kwargs) == ([], [decode_req])
    assert req.finished_reason.status_code == 400


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pp_consensus_keeps_ready_prefill_admission_abort_in_ready_set(code, role):
    req = _req(code)
    scheduler = code.SchedulerPPMixin()
    scheduler.pp_group = SimpleNamespace(is_first_rank=True)
    scheduler.get_rids = Mock(return_value=([req.rid], []))
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[req])
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[SimpleNamespace(req=req)]
    )
    method = (
        scheduler._pp_pd_get_bootstrapped_ids
        if role == "prefill"
        else scheduler._pp_pd_get_prealloc_ids
    )
    expected = [[req.rid], []] if role == "prefill" else [[], [req.rid]]
    assert method() == expected


@pytest.mark.parametrize("status", [0, 2])
def test_prefill_admission_abort_notifies_known_peers_without_forward(code, status):
    req = _req(code)
    original = req.to_finish
    queue, sender = _prefill_queue(code, req, status)
    assert queue.pop_bootstrapped(return_failed_reqs=True) == ([], [req])
    assert req.finished_reason is original
    assert not req.pending_bootstrap
    queue.ensure_metadata_buffer.assert_not_called()
    queue.finalize_bootstrap.assert_not_called()
    code.release_kv_cache.assert_not_called()
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_any_call(
        "decode-a", 8001, 42, code.KVPoll.Failed, 7
    )
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_any_call(
        "decode-b", 8003, 42, code.KVPoll.Failed, 7
    )
    assert sender.kv_mgr.sync_status_to_decode_endpoint.call_count == 2
    assert sender.kv_mgr.request_status == sender.kv_mgr.transfer_infos == {}
    assert sender.kv_mgr.failure_records == sender.kv_mgr.req_to_decode_prefix_len == {}
    assert queue.pop_bootstrapped() == []
    queue.scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


def test_prefill_pending_abort_does_not_enter_optimistic_prefill(code):
    req = _req(code)
    queue, sender = _prefill_queue(code, req, code.KVPoll.Bootstrapping)
    assert queue.pop_bootstrapped() == []
    assert queue.queue == [req]
    queue.ensure_metadata_buffer.assert_not_called()
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()


@pytest.fixture
def real_streamer(code):
    ns = code.Req.finished.__globals__
    ns.update(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        ClassVar=ClassVar,
        msgspec=msgspec,
        wrap_as_pickle=lambda value: value,
        INIT_INCREMENTAL_DETOKENIZATION_OFFSET=5,
        DEFAULT_FORCE_STREAM_INTERVAL=50,
        get_serving=lambda: SimpleNamespace(stream_interval=1, weight_version="test"),
        get_observability=lambda: SimpleNamespace(
            enable_request_time_stats_logging=False
        ),
        envs=SimpleNamespace(
            SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS=SimpleNamespace(get=lambda: 0)
        ),
    )
    _load(
        ns, "managers/io_struct.py", {"BaseBatchReq": None, "BatchTokenIDOutput": None}
    )
    _load(
        ns,
        "utils/weight_versions.py",
        {"WeightVersionSpan": None, "compute_weight_version_spans": None},
    )
    _load(
        ns,
        "managers/scheduler_components/output_streamer.py",
        {"SchedulerOutputStreamer": None, "_GenerationStreamAccumulator": None},
    )
    payloads = []
    streamer = ns["SchedulerOutputStreamer"](
        send_to_detokenizer=SimpleNamespace(send_output=payloads.append),
        tree_cache=None,
        ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
        server_args=SimpleNamespace(),
        is_generation=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        disaggregation_mode="prefill",
        enable_hicache_storage=lambda: False,
    )
    return streamer, payloads


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("cleanup", ["metadata", "timeout"])
def test_prefill_admission_response_precedes_cleanup(
    code, real_streamer, monkeypatch, pp_size, cleanup
):
    req = _req(code)
    original = req.to_finish
    req.output_ids = array("q")
    req.origin_input_ids_unpadded = array("q", [1, 2, 3])
    req.finished_len = req.surr_offset = req.read_offset = None
    req.beam_group = req.http_worker_ipc = req.customized_info = None
    req.decoded_text = ""
    req.weight_version_events = []
    req.sampling_params = SimpleNamespace(
        skip_special_tokens=True, spaces_between_special_tokens=True, no_stop_trim=False
    )
    req.return_hidden_states = req.return_routed_experts = False
    req.return_indexer_topk = req.return_sampling_mask = False
    req.send_token_offset = req.send_output_token_logprobs_offset = 0
    req.send_decode_id_offset = req.reasoning_tokens = req.retraction_count = 0
    req.cached_tokens = req.cached_tokens_device = 0
    req.cached_tokens_host = req.cached_tokens_storage = 0
    req.mm_image_tokens = req.mm_audio_tokens = req.mm_video_tokens = 0
    queue, sender = _prefill_queue(code, req, code.KVPoll.Bootstrapping)
    queue.pp_size = pp_size
    queue.scheduler.output_streamer, payloads = real_streamer
    sender.kv_mgr.transfer_infos.clear()
    sender.kv_mgr.req_to_decode_prefix_len.clear()

    waiting = {"pp_good_rids": [], "pp_bad_rids": []} if pp_size > 1 else {}
    for _ in range(2):
        assert queue.pop_bootstrapped(**waiting) == []
        assert queue.queue == [req]
        assert req.pending_bootstrap
        assert req.finished_output is (pp_size == 1)
        assert req.finished_reason is (original if pp_size == 1 else None)
        assert req.to_finish is (None if pp_size == 1 else original)
        assert len(payloads) == (1 if pp_size == 1 else 0)
        sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()
        queue.scheduler.tree_cache.release_aborted_request.assert_not_called()
        assert 42 in sender.kv_mgr.request_status

    if cleanup == "metadata":
        _deliver_prefill_metadata(sender.kv_mgr, monkeypatch)
    else:
        sender.init_time -= sender.kv_mgr.bootstrap_timeout + 1
        monkeypatch.setattr(
            code.logger, "warning_once", code.logger.warning, raising=False
        )
    kwargs = (
        {
            "pp_good_rids": [req.rid] if cleanup == "metadata" else [],
            "pp_bad_rids": [req.rid] if cleanup == "timeout" else [],
        }
        if pp_size > 1
        else {}
    )
    assert queue.pop_bootstrapped(return_failed_reqs=True, **kwargs) == ([], [req])
    assert queue.pop_bootstrapped(**kwargs) == []
    assert queue.queue == []
    assert not req.pending_bootstrap
    assert req.finished_reason is original
    assert req.finished_output
    assert len(req.output_ids) == 0
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.rids == [req.rid]
    assert payload.finished_reasons == [original.to_json()]
    assert payload.finished_reasons[0]["status_code"] == 400
    assert payload.output_ids == [array("q")]
    assert payload.completion_tokens == [0]
    assert payload.decode_ids == [req.origin_input_ids_unpadded]
    assert not sender._send_started
    queue.ensure_metadata_buffer.assert_not_called()
    queue.finalize_bootstrap.assert_not_called()
    queue.scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
    code.release_kv_cache.assert_not_called()
    assert sender.kv_mgr.request_status == sender.kv_mgr.transfer_infos == {}
    assert sender.kv_mgr.failure_records == sender.kv_mgr.req_to_decode_prefix_len == {}
    assert sender.kv_mgr.sync_status_to_decode_endpoint.call_count == (
        cleanup == "metadata"
    )


def test_prefill_notification_failure_still_cleans_up(code):
    req = _req(code)
    queue, sender = _prefill_queue(code, req, code.KVPoll.WaitingForInput)
    sender.kv_mgr.sync_status_to_decode_endpoint.side_effect = RuntimeError("peer down")
    assert queue.pop_bootstrapped() == []
    assert req.finished_reason.status_code == 400
    assert sender.kv_mgr.sync_status_to_decode_endpoint.call_count == 2
    assert (
        sender.kv_mgr.request_status
        == sender.kv_mgr.transfer_infos
        == sender.kv_mgr.failure_records
        == {}
    )


@pytest.mark.parametrize("pages", [[], [1]])
def test_before_send_abort_refuses_even_metadata_only_send(code, pages):
    queue, sender = _prefill_queue(code, _req(code), code.KVPoll.WaitingForInput)
    sender.init(len(pages), 3)
    sender._prepare_send_indices(pages)
    with pytest.raises(RuntimeError, match="send"):
        sender.abort_before_send()
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()
    assert 42 in sender.kv_mgr.request_status


def test_general_sender_abort_does_not_notify_or_clear_inflight_room(code):
    queue, sender = _prefill_queue(code, _req(code), code.KVPoll.Transferring)
    sender._send_started = True
    sender.kv_mgr._staging_outstanding[42] = 1
    sender.abort()
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()
    assert 42 in sender.kv_mgr.transfer_infos
    assert sender.kv_mgr._staging_outstanding[42] == 1


def test_prefill_timeout_cleans_current_room_state(code, monkeypatch):
    req = _req(code)
    original = req.to_finish
    queue, sender = _prefill_queue(code, req, code.KVPoll.Bootstrapping)
    sender.kv_mgr.transfer_infos.clear()
    sender.init_time -= sender.kv_mgr.bootstrap_timeout + 1
    monkeypatch.setattr(code.logger, "warning_once", code.logger.warning, raising=False)

    assert queue.pop_bootstrapped(return_failed_reqs=True) == ([], [req])
    assert req.finished_reason is original
    assert sender.kv_mgr.request_status == sender.kv_mgr.failure_records == {}
    assert sender.kv_mgr.transfer_infos == sender.kv_mgr.req_to_decode_prefix_len == {}
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()


def test_prefill_abort_waits_for_pp_consensus_and_frees_reserved_metadata_once(code):
    req = _req(code)
    req.metadata_buffer_index = 7
    queue, sender = _prefill_queue(code, req, code.KVPoll.WaitingForInput)
    queue.pp_size = 2
    assert queue.pop_bootstrapped(pp_good_rids=[], pp_bad_rids=[]) == []
    assert queue.queue == [req]
    sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()
    assert queue.pop_bootstrapped(pp_good_rids=[], pp_bad_rids=[req.rid]) == []
    assert queue.queue == []
    assert req.metadata_buffer_index == -1
    assert queue.pop_bootstrapped(pp_good_rids=[], pp_bad_rids=[]) == []
    queue.scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    queue.scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


def test_decode_add_in_pp_enqueues_without_initializing_receiver(code):
    req = _req(code)
    queue, _, receiver = _decode_queue(code, req)
    queue.pp_size = 2
    queue.add(req)
    queue._create_receiver_and_enqueue.assert_called_once_with(
        req, is_rebootstrap=False
    )
    receiver.init.assert_not_called()
    queue._check_if_req_exceed_kv_capacity.assert_not_called()
    queue.scheduler.output_streamer.stream_output.assert_not_called()
    assert req.finished_reason is None


def test_decode_pending_abort_cancels_prefetched_query(code):
    queue, _, receiver = _decode_queue(code, _req(code))
    future = Mock()
    queue._prefill_dp_rank_queries = {"prefill:8000": ((42,), future)}
    queue.prefetch_prefill_dp_rank_queries()
    future.cancel.assert_called_once_with()
    assert queue._prefill_dp_rank_queries == {}
    receiver.init.assert_not_called()


def test_decode_valid_request_still_initializes_receiver(code):
    req = _req(code)
    req.to_finish = None
    queue, decode_req, receiver = _decode_queue(code, req)
    assert queue.pop_preallocated() == ([], [])
    assert queue.queue == [decode_req]
    assert queue.pending_reqs == []
    receiver.init.assert_called_once_with(0)
    receiver._send_abort_notification.assert_not_called()
    assert req.finished_reason is None


def test_decode_abort_does_not_repeat_finished_output(code):
    req = _req(code)
    queue, decode_req, receiver = _decode_queue(code, req)
    req.finished_output = True
    assert queue.pop_preallocated() == ([], [decode_req])
    assert req.finished_reason.status_code == 400
    queue.scheduler.output_streamer.stream_output.assert_not_called()
    receiver._send_abort_notification.assert_called_once_with()


def _pp_prefill_queues(code):
    queues = []
    for rank in range(3):
        queue, sender = _prefill_queue(code, _req(code), code.KVPoll.Bootstrapping)
        queue.pp_size = 3
        sender.kv_mgr.pp_rank = rank
        sender.kv_mgr.pp_size = 3
        sender.kv_mgr.transfer_infos.clear()
        sender.kv_mgr.req_to_decode_prefix_len.clear()
        queues.append(queue)
    return queues


def _pp_prefill_round(code, queues):
    consensus = None
    for rank, queue in enumerate(queues):
        scheduler = code.SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=rank == 0)
        scheduler.attn_cp_cpu_group = None
        scheduler.attn_tp_cpu_group = None
        scheduler.disagg_prefill_bootstrap_queue = queue
        scheduler._pp_recv_pyobj_from_prev_stage = Mock(return_value=consensus)
        consensus = scheduler._pp_pd_get_bootstrapped_ids()
    results = [
        queue.pop_bootstrapped(
            return_failed_reqs=True,
            pp_good_rids=consensus[0],
            pp_bad_rids=consensus[1],
        )
        for queue in queues
    ]
    return consensus, results


def _deliver_prefill_metadata(manager, monkeypatch):
    # Run the actual registration loop once; only transport and parsing are mocked.
    namespace = manager.start_prefill_thread.__globals__
    monkeypatch.setitem(
        namespace,
        "threading",
        SimpleNamespace(Thread=lambda target: SimpleNamespace(start=target)),
    )
    info = SimpleNamespace(
        endpoint="decode", dst_port=8001, is_dummy=False, decode_prefix_len=64
    )
    monkeypatch.setitem(
        namespace, "TransferInfo", SimpleNamespace(from_zmq=Mock(return_value=info))
    )
    manager.resolve_kv_replica_factor = Mock()
    manager.server_socket = SimpleNamespace(
        recv_multipart=Mock(
            side_effect=[
                [b"42", b"decode", b"8001", b"session", b"", b"", b"", b"1"],
                StopIteration,
            ]
        )
    )
    with pytest.raises(StopIteration):
        manager.start_prefill_thread()


@pytest.mark.parametrize("arrival_order", [(0, 1, 2), (2, 1, 0)])
def test_pp_admission_abort_waits_for_metadata_on_every_stage(
    code, monkeypatch, arrival_order
):
    queues = _pp_prefill_queues(code)
    reqs = [queue.queue[0] for queue in queues]
    original_errors = [req.to_finish for req in reqs]
    senders = [req.disagg_kv_sender for req in reqs]

    for rank in arrival_order:
        assert _pp_prefill_round(code, queues) == ([[], []], [([], [])] * 3)
        for queue, req in zip(queues, reqs):
            assert queue.queue == [req]
            assert req.finished_reason is None
            assert req.prefill_attempt_count == 0
            queue.ensure_metadata_buffer.assert_not_called()
            queue.finalize_bootstrap.assert_not_called()
            queue.scheduler.output_streamer.stream_output.assert_not_called()
            req.disagg_kv_sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()
        _deliver_prefill_metadata(senders[rank].kv_mgr, monkeypatch)

    consensus, results = _pp_prefill_round(code, queues)
    assert consensus == [[reqs[0].rid], []]
    assert results == [([], [req]) for req in reqs]
    for queue, req, error, sender in zip(queues, reqs, original_errors, senders):
        assert req.finished_reason is error
        assert error.status_code == 400
        assert not sender._send_started
        assert not req.pending_bootstrap
        manager = sender.kv_mgr
        manager.sync_status_to_decode_endpoint.assert_called_once_with(
            "decode", 8001, 42, code.KVPoll.Failed, manager._prefill_unique_rank()
        )
        assert manager.request_status == manager.transfer_infos == {}
        assert manager.req_to_decode_prefix_len == manager.failure_records == {}
    assert _pp_prefill_round(code, queues) == ([[], []], [([], [])] * 3)
    for queue, req in zip(queues, reqs):
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )
    code.release_kv_cache.assert_not_called()


@pytest.mark.parametrize("completed_rank", [0, 1, 2])
def test_pp_completed_abort_keeps_immediate_bad_union(code, completed_rank):
    queues = _pp_prefill_queues(code)
    reqs = [queue.queue[0] for queue in queues]
    reqs[completed_rank].update_finish_state()
    consensus, results = _pp_prefill_round(code, queues)
    assert consensus == [[], [reqs[0].rid]]
    assert results == [([], [req]) for req in reqs]
    assert all(not queue.queue for queue in queues)


def test_pp_admission_timeout_cleans_known_state_but_late_metadata_can_recreate_room(
    code, monkeypatch
):
    queues = _pp_prefill_queues(code)
    reqs = [queue.queue[0] for queue in queues]
    senders = [req.disagg_kv_sender for req in reqs]
    _deliver_prefill_metadata(senders[0].kv_mgr, monkeypatch)
    assert _pp_prefill_round(code, queues) == ([[], []], [([], [])] * 3)

    monkeypatch.setattr(code.logger, "warning_once", code.logger.warning, raising=False)
    senders[1].init_time -= senders[1].kv_mgr.bootstrap_timeout + 1
    consensus, results = _pp_prefill_round(code, queues)
    assert consensus == [[], [reqs[0].rid]]
    assert results == [([], [req]) for req in reqs]
    senders[0].kv_mgr.sync_status_to_decode_endpoint.assert_called_once()
    for sender in senders:
        manager = sender.kv_mgr
        assert manager.request_status == manager.transfer_infos == {}
        assert manager.req_to_decode_prefix_len == manager.failure_records == {}
    for sender in senders[1:]:
        sender.kv_mgr.sync_status_to_decode_endpoint.assert_not_called()

    late_manager = senders[2].kv_mgr
    _deliver_prefill_metadata(late_manager, monkeypatch)
    assert _pp_prefill_round(code, queues) == ([[], []], [([], [])] * 3)
    # Existing timeout behavior: late metadata has no scheduler owner to retire it.
    assert late_manager.request_status[42] == code.KVPoll.WaitingForInput
    assert 42 in late_manager.transfer_infos
    assert late_manager.req_to_decode_prefix_len == {42: 64}
    late_manager.sync_status_to_decode_endpoint.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
