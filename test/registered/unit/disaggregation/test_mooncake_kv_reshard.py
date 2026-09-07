from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mooncake.reshard.kv_cache import (
    kv_cache_logical_plan_from_json,
    kv_cache_part_to_json,
    kv_cache_placement_from_json,
    kv_cache_placement_to_json,
    kv_cache_runtime_binding_to_json,
)
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.srt.disaggregation.mooncake.kv_reshard import (
    KV_RESHARD_PROTOCOL,
    KVReshardCompatibilityError,
    KVReshardRuntime,
    encode_wire_json,
    record_writer_completion,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.utils import get_pp_indices


def _server_args() -> SimpleNamespace:
    return SimpleNamespace(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        revision=None,
        weight_version="test-generation",
    )


def _kv_args(
    *,
    total_layers: int,
    pp_rank: int,
    pp_size: int,
    tp_size: int,
    total_heads: int = 4,
    page_size: int = 16,
    base_address: int = 1_000_000_000,
    capacity_tokens: int = 64,
) -> SimpleNamespace:
    start, end = get_pp_indices(total_layers, pp_rank, pp_size)
    layer_count = end - start
    head_count = total_heads // tp_size if total_heads >= tp_size else 1
    head_dim = 8
    itemsize = 2
    row_bytes = head_count * head_dim * itemsize
    ptrs = tuple(base_address + index * 1_000_000 for index in range(layer_count * 2))
    return SimpleNamespace(
        kv_data_ptrs=list(ptrs),
        kv_data_lens=[capacity_tokens * row_bytes] * len(ptrs),
        kv_item_lens=[page_size * row_bytes] * len(ptrs),
        kv_layer_ids=[],
        total_kv_layers=total_layers,
        total_kv_head_num=total_heads,
        kv_head_num=head_count,
        kv_head_dim=head_dim,
        kv_value_head_dim=head_dim,
        kv_itemsize=itemsize,
        kv_storage_dtype_str="float16",
        kv_cache_layout="nhd",
        kv_is_quantized=False,
        page_size=page_size,
        prefill_start_layer=start,
        prefill_end_layer=end,
        gpu_id=0,
    )


def _runtime(
    role: str,
    *,
    total_layers: int,
    pp_rank: int,
    pp_size: int,
    tp_rank: int,
    tp_size: int,
    total_heads: int = 4,
    page_size: int = 16,
    base_address: int = 1_000_000_000,
    capacity_tokens: int = 64,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> KVReshardRuntime:
    return KVReshardRuntime(
        kv_args=_kv_args(
            total_layers=total_layers,
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_size=tp_size,
            total_heads=total_heads,
            page_size=page_size,
            base_address=base_address,
            capacity_tokens=capacity_tokens,
        ),
        server_args=_server_args(),
        role=role,
        dp_rank=dp_rank,
        dp_size=dp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def _source_placement(
    *,
    total_layers: int,
    pp_size: int,
    tp_size: int,
    total_heads: int = 4,
    page_size: int = 16,
    dp_rank: int = 0,
    dp_size: int = 1,
):
    runtimes = [
        _runtime(
            "prefill",
            total_layers=total_layers,
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            total_heads=total_heads,
            page_size=page_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        for pp_rank in range(pp_size)
        for tp_rank in range(tp_size)
    ]
    placement = KVReshardRuntime.assemble_placement(
        (kv_cache_part_to_json(runtime.local_part) for runtime in runtimes),
        dp_size=dp_size,
        pp_size=pp_size,
        tp_size=tp_size,
    )
    routes = {
        runtime.participant_id: {
            "rank_ip": "127.0.0.1",
            "rank_port": 20000 + index,
            "participant_id": runtime.participant_id,
        }
        for index, runtime in enumerate(runtimes)
    }
    return placement, routes


def test_pp_2_to_3_and_tp_1_to_2_routes_come_only_from_planner() -> None:
    source, routes = _source_placement(total_layers=5, pp_size=2, tp_size=1)
    target = _runtime(
        "decode",
        total_layers=5,
        pp_rank=1,
        pp_size=3,
        tp_rank=1,
        tp_size=2,
    )

    route_plan = target.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )

    assert route_plan.expected_writer_ids == (
        "prefill:dp0:pp0:tp0",
        "prefill:dp0:pp1:tp0",
    )
    assert {info["participant_id"] for info in route_plan.bootstrap_infos} == set(
        route_plan.expected_writer_ids
    )
    assert {info["required_dst_info_num"] for info in route_plan.bootstrap_infos} == {4}
    edge_layers = {
        edge.global_layer_id
        for info in route_plan.bootstrap_infos
        for edge in kv_cache_logical_plan_from_json(
            info["kv_reshard_plan_json"]
        ).edges
    }
    assert edge_layers == {1, 2}


def test_pd_routes_allow_different_source_and_target_dp_pp_tp() -> None:
    source, routes = _source_placement(
        total_layers=6,
        pp_size=3,
        tp_size=2,
        dp_rank=1,
        dp_size=2,
    )
    target = _runtime(
        "decode",
        total_layers=6,
        pp_rank=0,
        pp_size=2,
        tp_rank=3,
        tp_size=4,
        dp_rank=3,
        dp_size=4,
    )

    route_plan = target.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )

    assert route_plan.expected_writer_ids == (
        "prefill:dp1:pp0:tp1",
        "prefill:dp1:pp1:tp1",
    )
    for info in route_plan.bootstrap_infos:
        plan = kv_cache_logical_plan_from_json(info["kv_reshard_plan_json"])
        assert plan.source_dp_rank == 1
        assert plan.target_part.rank.dp == 3


def test_gqa_replica_selects_exact_matching_writer() -> None:
    source, routes = _source_placement(
        total_layers=1, pp_size=1, tp_size=4, total_heads=2
    )
    target = _runtime(
        "decode",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=3,
        tp_size=4,
        total_heads=2,
    )

    route_plan = target.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )

    assert route_plan.expected_writer_ids == ("prefill:dp0:pp0:tp3",)


def test_incomplete_source_collection_cannot_form_a_placement() -> None:
    placement, _ = _source_placement(total_layers=2, pp_size=2, tp_size=2)

    with pytest.raises(ValueError, match="topology_id differs"):
        KVReshardRuntime.assemble_placement(
            (kv_cache_part_to_json(part) for part in placement.parts[:-1]),
            dp_size=1,
            pp_size=2,
            tp_size=2,
        )


class _FakeEngine:
    def __init__(self, identity: int):
        self.identity = identity
        self.calls = []

    def batch_transfer_sync(self, *args):
        self.calls.append(args)
        return 0


class _FailingEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_sync(self, *args):
        self.calls.append(args)
        return -1


def test_submit_chunk_uses_legacy_batch_transfer_sync_contract() -> None:
    runtime = _runtime(
        "prefill",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
    )
    engine = _FakeEngine(0xA000)
    runtime.bind_runtime(session_id="source:12345", transfer_engine=engine)
    execution = SimpleNamespace(
        batches=(
            SimpleNamespace(
                endpoint="target:12346",
                source_addresses=(100,),
                target_addresses=(200,),
                sizes=(16,),
            ),
        )
    )

    assert runtime.submit_chunk(execution.batches) == 0
    assert engine.calls == [("target:12346", [100], [200], [16])]
    failing_engine = _FailingEngine()
    runtime.bind_runtime(session_id="source:12345", transfer_engine=failing_engine)
    assert runtime.submit_chunk(execution.batches) == -1


def test_prepared_request_lowers_partial_page_ranges_to_native_batch() -> None:
    source_runtime = _runtime(
        "prefill",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        total_heads=2,
    )
    target_runtime = _runtime(
        "decode",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=1,
        tp_size=2,
        total_heads=2,
        base_address=2_000_000_000,
    )
    source_engine = _FakeEngine(0xA001)
    target_engine = _FakeEngine(0xA002)
    source_runtime.bind_runtime(
        session_id="source:12345", transfer_engine=source_engine
    )
    target_runtime.bind_runtime(
        session_id="target:12346", transfer_engine=target_engine
    )
    assert source_runtime.binding is not None
    assert (
        source_runtime.binding.buffers[0].fragment.worker_id
        == source_runtime.participant_id
    )
    source, routes = _source_placement(
        total_layers=1, pp_size=1, tp_size=1, total_heads=2
    )
    route_plan = target_runtime.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )

    prepared = source_runtime.prepare_transfer(
        logical_plan_json=route_plan.bootstrap_infos[0]["kv_reshard_plan_json"],
        target_binding_json=kv_cache_runtime_binding_to_json(target_runtime.binding),
    )
    batches = source_runtime.lower_chunk(
        prepared_plan=prepared,
        source_page_ids=(1,),
        target_page_ids=(2,),
        token_start=3,
        token_count=5,
    )
    assert source_runtime.submit_chunk(batches) == 0

    assert sum(len(batch.sizes) for batch in batches) == 10
    assert sum(sum(batch.sizes) for batch in batches) == 5 * 2 * 8 * 2
    assert len(source_engine.calls) == 1
    assert source_engine.calls[0][0] == "target:12346"


@pytest.mark.parametrize("page_size", [1, 16])
def test_full_rows_coalesce_contiguous_source_and_target_slots(
    page_size: int,
) -> None:
    capacity_tokens = 8192 if page_size == 1 else 512
    source_runtime = _runtime(
        "prefill",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        total_heads=2,
        page_size=page_size,
        capacity_tokens=capacity_tokens,
    )
    target_runtime = _runtime(
        "decode",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        total_heads=2,
        page_size=page_size,
        base_address=2_000_000_000,
        capacity_tokens=capacity_tokens,
    )
    source_runtime.bind_runtime(
        session_id="source:12345", transfer_engine=_FakeEngine(0xB001)
    )
    target_runtime.bind_runtime(
        session_id="target:12346", transfer_engine=_FakeEngine(0xB002)
    )
    source, routes = _source_placement(
        total_layers=1,
        pp_size=1,
        tp_size=1,
        total_heads=2,
        page_size=page_size,
    )
    route_plan = target_runtime.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )
    prepared = source_runtime.prepare_transfer(
        logical_plan_json=route_plan.bootstrap_infos[0]["kv_reshard_plan_json"],
        target_binding_json=kv_cache_runtime_binding_to_json(target_runtime.binding),
    )
    if page_size == 1:
        token_count = 1200
        source_pages = tuple(range(3, 2403, 2))
        target_pages = tuple(range(4000, 6400, 2))
        expected_runs = token_count
    else:
        token_count = 18
        source_pages = (1, 4)
        target_pages = (5, 9)
        expected_runs = 2
    batches = source_runtime.lower_chunk(
        prepared_plan=prepared,
        source_page_ids=source_pages,
        target_page_ids=target_pages,
        token_start=3,
        token_count=token_count,
    )

    assert sum(len(batch.sizes) for batch in batches) == 2 * expected_runs
    assert len(batches) == 1
    assert sum(sum(batch.sizes) for batch in batches) == token_count * 2 * 2 * 8 * 2


def test_reshard_queue_preserves_legacy_session_affinity() -> None:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.request_status = {room: KVPoll.WaitingForInput for room in range(8)}
    manager.transfer_infos = {
        room: {"decode:12346": object()} for room in manager.request_status
    }
    manager.transfer_queues = [MagicMock() for _ in range(4)]
    manager.enable_kv_reshard = True

    for room in range(4):
        manager.add_transfer_request(
            room,
            np.asarray([room], dtype=np.int32),
            slice(0, 1),
            is_last_chunk=False,
            num_kv_tokens=1,
        )

    assert [queue.put.call_count for queue in manager.transfer_queues] == [0, 0, 4, 0]

    for queue in manager.transfer_queues:
        queue.reset_mock()
    manager.enable_kv_reshard = False

    for room in range(4, 8):
        manager.add_transfer_request(
            room,
            np.asarray([room], dtype=np.int32),
            slice(0, 1),
            is_last_chunk=False,
        )

    assert [queue.put.call_count for queue in manager.transfer_queues] == [0, 0, 4, 0]


def test_tp_slice_vectorization_caps_native_batches_at_1024_operations() -> None:
    source_runtime = _runtime(
        "prefill",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        total_heads=2,
        capacity_tokens=4096,
    )
    target_runtime = _runtime(
        "decode",
        total_layers=1,
        pp_rank=0,
        pp_size=1,
        tp_rank=1,
        tp_size=2,
        total_heads=2,
        base_address=2_000_000_000,
        capacity_tokens=4096,
    )
    source_runtime.bind_runtime(
        session_id="source:12345", transfer_engine=_FakeEngine(0xC001)
    )
    target_runtime.bind_runtime(
        session_id="target:12346", transfer_engine=_FakeEngine(0xC002)
    )
    source, routes = _source_placement(
        total_layers=1, pp_size=1, tp_size=1, total_heads=2
    )
    route_plan = target_runtime.plan_target_routes(
        source_placement_json=kv_cache_placement_to_json(source), routes=routes
    )
    prepared = source_runtime.prepare_transfer(
        logical_plan_json=route_plan.bootstrap_infos[0]["kv_reshard_plan_json"],
        target_binding_json=kv_cache_runtime_binding_to_json(target_runtime.binding),
    )
    batches = source_runtime.lower_chunk(
        prepared_plan=prepared,
        source_page_ids=range(128),
        target_page_ids=range(128),
        token_start=0,
        token_count=2048,
    )

    assert sum(len(batch.sizes) for batch in batches) == 4096
    assert len(batches) == 4
    assert all(len(batch.sizes) <= 1024 for batch in batches)


def test_illegal_tp_head_ratio_fails_before_registration() -> None:
    with pytest.raises(KVReshardCompatibilityError, match="exact shard/replica"):
        _runtime(
            "decode",
            total_layers=1,
            pp_rank=0,
            pp_size=1,
            tp_rank=0,
            tp_size=3,
            total_heads=4,
        )


def test_request_wire_roundtrip() -> None:
    header = {
        "protocol": KV_RESHARD_PROTOCOL,
        "room": 17,
        "endpoint": "10.0.0.2",
        "dst_port": 19001,
        "mooncake_session_id": "decode:19000",
        "required_dst_info_num": 2,
        "decode_prefix_len": 3,
    }
    msg = [
        f"{KV_RESHARD_PROTOCOL}_REQUEST".encode("ascii"),
        encode_wire_json(header),
        np.asarray([3, 7], dtype=np.int32).tobytes(),
        b"4",
    ]

    info = TransferInfo.from_kv_reshard_zmq(msg)

    assert info.room == 17
    assert info.decode_prefix_len == 3
    assert info.required_dst_info_num == 2
    assert info.dst_kv_indices.tolist() == [3, 7]
    assert info.dst_aux_index == 4


def test_exact_writer_completion_is_idempotent_and_rejects_unknown() -> None:
    expected = {"prefill:dp0:pp0:tp0", "prefill:dp0:pp1:tp0"}
    arrived = set()

    assert record_writer_completion(expected, arrived, "unexpected") == (
        False,
        False,
    )
    assert record_writer_completion(expected, arrived, "prefill:dp0:pp0:tp0") == (
        True,
        False,
    )
    assert record_writer_completion(expected, arrived, "prefill:dp0:pp0:tp0") == (
        True,
        False,
    )
    assert record_writer_completion(expected, arrived, "prefill:dp0:pp1:tp0") == (
        True,
        True,
    )


@patch("sglang.srt.disaggregation.common.conn.requests.get")
def test_manifest_capability_mismatch_fails_before_transfer(mock_get) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "attn_tp_size": 1,
        "attn_cp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "page_size": 16,
        "kv_cache_dtype": "float16",
        "follow_bootstrap_room": False,
        "kv_reshard_capability": False,
    }
    mock_get.return_value = response
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.prefill_info_table = {}
    manager.kv_args = SimpleNamespace(page_size=16)
    manager.kv_cache_dtype_str = "float16"
    manager.dcp_size = 1
    manager.is_mla_backend = False
    manager.is_hybrid_mla_backend = False
    manager.enable_kv_reshard = True

    with pytest.raises(RuntimeError, match="KV_RESHARD"):
        manager.try_ensure_parallel_info("127.0.0.1:8999")


def test_legacy_route_payload_does_not_advertise_reshard() -> None:
    server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
    server._is_ready = lambda: True
    server.attn_tp_size = 2
    server.attn_cp_size = 1
    server.dp_size = 1
    server.pp_size = 2
    server.page_size = 16
    server.kv_cache_dtype = "float16"
    server.follow_bootstrap_room = True
    server.enable_dsa_cache_layer_split = False
    server.prefill_http_port = 31020
    server.kv_reshard_capability = False
    request = SimpleNamespace(
        query={
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
    )

    response = asyncio.run(server._handle_route_get(request))
    payload = json.loads(response.body)

    assert response.status == 200
    assert "kv_reshard_capability" not in payload


def test_bootstrap_aggregates_complete_source_placement() -> None:
    placement, _ = _source_placement(total_layers=5, pp_size=2, tp_size=2)
    server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
    server._is_ready = lambda: True
    server.kv_reshard_capability = True
    server.dp_size = 1
    server.pp_size = 2
    server.attn_tp_size = 2
    server.kv_reshard_parts = {
        0: {
            part.participant_id: kv_cache_part_to_json(part)
            for part in placement.parts
        }
    }
    server.kv_reshard_routes = {
        0: {
            part.participant_id: {
                "rank_ip": "127.0.0.1",
                "rank_port": 19000 + index,
            }
            for index, part in enumerate(placement.parts)
        }
    }
    server.lock = asyncio.Lock()
    request = SimpleNamespace(query={"prefill_dp_rank": "0"})

    response = asyncio.run(server._handle_kv_reshard_placement(request))
    cached_response = asyncio.run(server._handle_kv_reshard_placement(request))
    payload = json.loads(response.body)

    assert response.status == 200
    assert payload["protocol"] == KV_RESHARD_PROTOCOL
    assert (
        kv_cache_placement_from_json(payload["placement_json"]).digest
        == placement.digest
    )
    assert set(payload["routes"]) == {part.participant_id for part in placement.parts}
    assert cached_response.body == response.body


@patch.object(MooncakeKVReceiver, "_register_kv_args", return_value=True)
@patch("sglang.srt.disaggregation.mooncake.conn.requests.get")
def test_concurrent_connection_cache_miss_is_single_flight(
    mock_get, mock_register
) -> None:
    source, routes = _source_placement(total_layers=2, pp_size=2, tp_size=1)
    target = _runtime(
        "decode",
        total_layers=2,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        capacity_tokens=256,
    )
    target.plan_target_routes = MagicMock(wraps=target.plan_target_routes)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "protocol": KV_RESHARD_PROTOCOL,
        "placement_json": kv_cache_placement_to_json(source),
        "routes": routes,
    }

    def delayed_get(*args, **kwargs):
        time.sleep(0.05)
        return response

    mock_get.side_effect = delayed_get
    manager = SimpleNamespace(
        enable_kv_reshard=True,
        kv_reshard=target,
        kv_reshard_connection_pool={},
        connection_lock=threading.Lock(),
        record_failure=MagicMock(),
        update_status=MagicMock(),
    )
    receivers = []
    for room in (1, 2):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_addr = "127.0.0.1:8999"
        receiver.prefill_dp_rank = 0
        receiver.bootstrap_room = room
        receiver.conclude_state = None
        receivers.append(receiver)

    errors = []

    def setup(receiver):
        try:
            receiver._setup_bootstrap_infos()
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=setup, args=(receiver,)) for receiver in receivers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mock_get.call_count == 1
    assert target.plan_target_routes.call_count == 1
    assert mock_register.call_count == 1


@patch.object(CommonKVReceiver, "disconnect_endpoint")
def test_prefill_failure_invalidates_namespaced_reshard_cache_and_endpoint(
    mock_disconnect,
) -> None:
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.enable_kv_reshard = True
    manager.connection_lock = threading.Lock()
    manager.connection_pool = {}
    manager.kv_reshard_connection_pool = {
        ("dead:8999", 0): [{"rank_ip": "10.0.0.1", "rank_port": 19000}],
        ("live:8999", 0): [],
    }
    manager.prefill_info_table = {"dead:8999": object(), "live:8999": object()}
    manager.addr_to_rooms_tracker = defaultdict(set)
    manager.request_status = {}

    manager._handle_node_failure("dead:8999")

    assert set(manager.kv_reshard_connection_pool) == {("live:8999", 0)}
    mock_disconnect.assert_called_once_with("tcp://10.0.0.1:19000")


@pytest.mark.parametrize(
    "mode,backend,staging,message",
    [
        ("null", "mooncake", False, "requires PD prefill/decode mode"),
        ("prefill", "nixl", False, "requires the mooncake transfer backend"),
        ("decode", "mooncake", True, "mutually exclusive"),
    ],
)
def test_reshard_argument_validation_uses_pd_hook(
    mode: str, backend: str, staging: bool, message: str
) -> None:
    from sglang.srt.arg_groups import pd_disaggregation_hook

    args = SimpleNamespace(
        enable_prefix_mm_cache=False,
        encoder_only=False,
        language_only=False,
        enable_mooncake_kv_reshard=True,
        disaggregation_mode=mode,
        disaggregation_transfer_backend=backend,
    )
    with (
        patch.object(pd_disaggregation_hook, "resolving_view", return_value=args),
        patch("sglang.srt.arg_groups.model_hook.handle_language_model_only"),
        patch.object(
            pd_disaggregation_hook.envs.SGLANG_DISAGG_STAGING_BUFFER,
            "get",
            return_value=staging,
        ),
        pytest.raises(ValueError, match=message),
    ):
        pd_disaggregation_hook.handle_encoder_disaggregation(args)
