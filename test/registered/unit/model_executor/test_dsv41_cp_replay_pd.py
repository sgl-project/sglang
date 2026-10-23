"""CPU contracts for CP owner fan-in and the upstream paged PD ABI."""

import dataclasses
import datetime
import enum
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_dsv41_cp_replay_layout import definitions

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def pd_contract():
    return definitions("srt/disaggregation/utils.py", ["dsv41_cp_replay_enabled"])


@pytest.mark.parametrize("end,window_size", [(129, 32), (257, 128)])
def test_window_handoff_checks_history_and_excludes_page_padding(end, window_size):
    cls = definitions("srt/mem_cache/dsv41_request_window.py", ["RequestWindow"])[
        "RequestWindow"
    ]
    window = object.__new__(cls)
    window.capacity, window.num_slots, window.page_size = 256, 8, 256
    window.tags = torch.full((3, 8 * 256 + 256), -1, dtype=torch.int64)
    start = max(0, end - window_size) // 256 * 256
    positions = torch.arange(max(start, end - window_size), end)
    rows = window.row_loc(2, positions)
    window.tags[:, rows] = positions
    assert window.validate_transfer(2, start, end, window_size=window_size)
    window.tags[-1, rows[-1]] = -1
    assert not window.validate_transfer(2, start, end, window_size=window_size)


@pytest.mark.parametrize("cp", [1, 8])
@pytest.mark.parametrize("replay", [False, True])
def test_decode_derives_owner_fan_in_from_remote_cp(cp, replay):
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["_resolve_rank_mapping"],
        pd_contract(),
        klass="CommonKVManager",
    )
    manager = NS(
        attn_tp_size=8,
        attn_cp_size=1,
        attn_cp_rank=0,
        pp_size=1,
        pp_rank=0,
        is_mla_backend=True,
        is_hybrid_mla_backend=False,
        kv_args=NS(engine_rank=3),
        enable_all_cp_ranks_for_transfer=False,
    )
    info = NS(
        attn_tp_size=8 // cp,
        attn_cp_size=cp,
        pp_size=1,
        enable_kv_cache_sharding=False,
        enable_dsa_cache_layer_split=False,
        dsv41_spec_layout={"target": "v4", "draft": "v4"}
        | ({"cp_encoder_replay_version": 1} if replay and cp > 1 else {}),
    )
    functions["_resolve_rank_mapping"](manager, info)
    owners = cp if replay else 1
    assert info.target_cp_ranks == list(range(owners))
    assert info.required_prefill_response_num == owners
    assert info.required_dst_info_num == cp


def test_replicated_state_uses_one_sender():
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["_should_skip_cp_replicated_state_transfer"],
        klass="CommonKVManager",
        env={
            "get_parallel": lambda: NS(
                enable_dsa_cache_layer_split=False, enable_cp_tp_group_sharing=False
            )
        },
    )
    selected = []
    for rank in range(8):
        manager = NS(
            attn_cp_size=8, attn_cp_rank=rank, enable_all_cp_ranks_for_transfer=True
        )
        if not functions["_should_skip_cp_replicated_state_transfer"](manager):
            selected.append(rank)
    assert selected == [0]


def handoff_fixture(cp=8, end=257, prefix=0, *, replay=True, reduce=None):
    state_type = definitions(
        "srt/disaggregation/base/conn.py", ["StateType"], {"enum": enum}
    )["StateType"]
    state_types = [
        state_type.SWA,
        state_type.DSA,
        state_type.DSV4_REQUEST_STATE,
        state_type.SWA,
        state_type.REQUEST_ROW,
    ]
    window_cls = definitions(
        "srt/mem_cache/dsv41_request_window.py", ["RequestWindow"]
    )["RequestWindow"]
    window = object.__new__(window_cls)
    window.capacity, window.num_slots, window.page_size = 256, 8, 256
    window.tags = torch.full((3, 8 * 256 + 256), -1, dtype=torch.int64)
    positions = torch.arange(max(prefix, end - 128), end)
    window.tags[:, window.row_loc(3, positions)] = positions
    window.validate_transfer = Mock(wraps=window.validate_transfer)
    pending_pair = np.array([51], dtype=np.int32)
    kvcache = NS(
        request_window=window,
        sliding_window=128,
        request_state_transfer_indices=Mock(return_value=pending_pair),
    )
    table = torch.arange(8 * 8192).reshape(8, 8192)
    sender = NS(
        get_max_transfer_tokens=lambda: None,
        should_send_kv_chunk=lambda *a: True,
        send=Mock(),
    )
    req = NS(
        start_send_idx=prefix,
        origin_input_ids=[0] * end,
        extend_end=end,
        disagg_decode_prefix_len=prefix,
        kv=NS(req_pool_idx=3),
        disagg_kv_sender=sender,
        rid="request",
        bootstrap_room=17,
    )
    metadata = Mock()
    manager = NS(
        kv_args=NS(state_types=state_types),
        dsv41_spec_layout={"cp_encoder_replay_version": 1} if cp > 1 and replay else {},
    )
    manager.conclude_failure = Mock()
    scheduler = NS(
        token_to_kv_pool_allocator=NS(
            page_size=256,
            get_kvcache=lambda: kvcache,
            translate_kv_indices_for_transfer=lambda x: x,
        ),
        req_to_token_pool=NS(req_to_token=table),
        enable_staging=False,
        disagg_metadata_buffers=NS(set_buf=metadata),
        disagg_prefill_bootstrap_queue=NS(kv_manager=manager),
        disagg_prefill_pending_chunk_rids={req.rid},
        attn_cp_cpu_group=object(),
    )
    scheduler.clear_pending_chunk_send = lambda r: (
        scheduler.disagg_prefill_pending_chunk_rids.discard(r.rid)
    )
    suffix = Mock(return_value=np.array([91], dtype=np.int32))
    collective = Mock(side_effect=reduce)
    cpu_torch = NS(
        tensor=torch.tensor,
        int32=torch.int32,
        distributed=NS(all_reduce=collective, ReduceOp=dist.ReduceOp),
    )
    send = definitions(
        "srt/disaggregation/prefill.py",
        ["_send_kv_chunk"],
        {
            **pd_contract(),
            "CommonKVManager": NS,
            "torch": cpu_torch,
            "StateType": state_type,
            "get_parallel": lambda: NS(attn_cp_size=cp),
            "_is_npu": False,
            "np": np,
            "kv_to_page_indices": lambda x, size: (x[::size] // size).numpy(),
            "suffix_page_indices": suffix,
        },
        klass="SchedulerDisaggregationPrefillMixin",
    )["_send_kv_chunk"]
    return NS(
        send=lambda: send(scheduler, req, last_chunk=True),
        window=window,
        scheduler=scheduler,
        req=req,
        sender=sender,
        metadata=metadata,
        manager=manager,
        collective=collective,
        table=table,
        pending_pair=pending_pair,
        suffix=suffix,
        kvcache=kvcache,
        state_types=state_types,
    )


@pytest.mark.parametrize(
    "cp,end,prefix",
    [(1, 257, 0), (8, 257, 0), (8, 4097, 0), (8, 4097, 3840), (8, 4096, 4096)],
)
def test_final_pd_payload_keeps_both_windows_pending_pair_and_valid_from(
    cp, end, prefix
):
    f = handoff_fixture(cp, end, prefix)
    f.send()
    payload = f.sender.send.call_args.args[1]
    assert len(payload) == len(f.state_types)
    expected = f.window.page_indices(3, max(prefix, end - 128) // 256 * 256, end)
    np.testing.assert_array_equal(payload[0], expected)
    np.testing.assert_array_equal(payload[3], expected)
    np.testing.assert_array_equal(payload[1], np.array([91], dtype=np.int32))
    assert payload[2] is f.pending_pair
    np.testing.assert_array_equal(payload[4], np.array([3], dtype=np.int32))
    f.suffix.assert_called_once_with(f.table, 3, prefix, end, 256)
    f.kvcache.request_state_transfer_indices.assert_called_once_with(3, end)
    f.metadata.assert_called_once_with(f.req)
    if cp > 1:
        f.window.validate_transfer.assert_called_once_with(
            3, prefix, end, window_size=128
        )
        f.collective.assert_called_once()
    else:
        f.window.validate_transfer.assert_not_called()
        f.collective.assert_not_called()
    f.manager.conclude_failure.assert_not_called()
    assert f.req.start_send_idx == end
    assert not f.scheduler.disagg_prefill_pending_chunk_rids


@pytest.mark.parametrize("failure", ["local", "peer", "missing_window", "slot"])
def test_bad_window_concludes_failure_before_metadata_or_send(failure):
    def reduce(value, **kwargs):
        if failure == "peer":
            value.zero_()

    f = handoff_fixture(reduce=reduce)
    if failure == "local":
        f.window.tags.fill_(-1)
    elif failure == "missing_window":
        f.kvcache.request_window = None
    elif failure == "slot":
        f.req.kv.req_pool_idx = -1
    f.send()
    f.manager.conclude_failure.assert_called_once_with(
        bootstrap_room=17,
        failure_reason="CP replay PD handoff has missing SWA history",
    )
    f.metadata.assert_not_called()
    f.sender.send.assert_not_called()
    f.collective.assert_called_once()
    assert f.req.start_send_idx == 0
    assert not f.scheduler.disagg_prefill_pending_chunk_rids


def test_encoder_off_does_not_validate_or_reduce_window():
    f = handoff_fixture(replay=False)
    f.window.tags.fill_(-1)
    f.send()
    f.window.validate_transfer.assert_not_called()
    f.collective.assert_not_called()
    f.sender.send.assert_called_once()


@pytest.mark.parametrize(
    "mode,cp,replay,version",
    [
        ("prefill", 8, True, 1),
        ("prefill", 8, False, None),
        ("prefill", 1, True, None),
        ("decode", 1, False, None),
        ("null", 8, True, None),
    ],
)
def test_wire_extension_only_advertises_cp_encoder_replay(mode, cp, replay, version):
    state_type = definitions(
        "srt/disaggregation/base/conn.py", ["StateType"], {"enum": enum}
    )["StateType"]
    layout = definitions(
        "srt/disaggregation/utils.py",
        ["get_dsv41_spec_layout"],
        {
            "StateType": state_type,
            "get_spec": lambda: NS(
                speculative_algorithm="DSPARK", speculative_num_draft_tokens=4
            ),
            "get_disagg": lambda: NS(disaggregation_mode=mode),
            "get_parallel": lambda: NS(attn_cp_size=cp),
            "get_exec": lambda: NS(
                features=NS(enable_encoder_swa_bounded_replay=replay)
            ),
        },
        strip_imports=True,
    )["get_dsv41_spec_layout"](
        NS(
            mla_compression_ratios=[1, 2],
            state_types=[state_type.SWA, state_type.SWA],
            kv_layer_ids=[0],
            kv_item_lens=[528],
            state_item_lens=[[528], [528]],
        )
    )
    assert layout.get("cp_encoder_replay_version") == version
    assert ("cp_encoder_replay_version" in layout) == (version is not None)


@pytest.mark.parametrize("version", [None, 1, 2])
def test_bootstrap_validates_version_before_mapping_or_registration(version):
    base_layout = {"num_draft_tokens": 4, "kv_item_lens": [528]}
    remote = base_layout | (
        {} if version is None else {"cp_encoder_replay_version": version}
    )
    info_cls = definitions(
        "srt/disaggregation/common/conn.py",
        ["PrefillServerInfo"],
        {"dataclasses": dataclasses},
    )["PrefillServerInfo"]
    payload = dict(
        attn_tp_size=1,
        attn_cp_size=8,
        dp_size=1,
        pp_size=1,
        page_size=256,
        kv_cache_dtype="bf16",
        follow_bootstrap_room=False,
        dsv41_spec_layout=remote,
    )
    manager = NS(
        prefill_info_table={},
        kv_args=NS(page_size=256),
        kv_cache_dtype_str="bf16",
        kv_layout=None,
        dsv41_spec_layout=base_layout,
        attn_cp_size=1,
        attn_tp_size=8,
        is_mla_backend=True,
        is_hybrid_mla_backend=False,
        dcp_size=1,
        _resolve_rank_mapping=Mock(),
    )
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["try_ensure_parallel_info"],
        {
            "PrefillServerInfo": info_cls,
            "requests": NS(
                get=lambda *args, **kwargs: NS(status_code=200, json=lambda: payload)
            ),
            "logger": Mock(),
        },
        klass="CommonKVManager",
    )
    if version is None or type(version) is int and version == 1:
        assert functions["try_ensure_parallel_info"](manager, "peer:8998")
        manager._resolve_rank_mapping.assert_called_once()
        assert manager.prefill_info_table["peer:8998"].dsv41_spec_layout == remote
    else:
        with pytest.raises(RuntimeError, match="Unsupported.*CP encoder replay PD"):
            functions["try_ensure_parallel_info"](manager, "peer:8998")
        manager._resolve_rank_mapping.assert_not_called()
        assert not manager.prefill_info_table
    assert payload["dsv41_spec_layout"] == remote


def _handoff_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        for healthy in (False, True):
            f = handoff_fixture(reduce=dist.all_reduce)
            f.scheduler.attn_cp_cpu_group = dist.group.WORLD
            if rank == 1 and not healthy:
                f.window.tags.fill_(-1)
            f.send()
            assert f.collective.call_count == 1
            assert f.sender.send.call_count == int(healthy)
            assert f.metadata.call_count == int(healthy)
            assert f.manager.conclude_failure.call_count == int(not healthy)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_one_bad_cp_window_aborts_all_peers_and_next_request_can_send():
    with tempfile.TemporaryDirectory(prefix="v41-pd-handoff-gloo-") as directory:
        mp.spawn(
            _handoff_worker,
            args=(str(Path(directory) / "rendezvous"),),
            nprocs=2,
            join=True,
        )


@pytest.mark.parametrize("pages", [0, 7, 17])
def test_cp_ranks_without_pages_still_send_the_final_state_and_completion(pages):
    filters = definitions(
        "srt/disaggregation/utils.py",
        ["_get_cp_rank_page_bounds", "filter_kv_indices_for_cp_rank"],
    )
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["_prepare_send_indices"],
        {**filters, "get_parallel": lambda: NS(enable_dsa_cache_layer_split=False)},
        klass="CommonKVSender",
    )
    owners = []
    for rank in range(8):
        sender = NS(
            curr_idx=0,
            num_kv_indices=pages,
            kv_mgr=NS(
                kv_shard_size=1,
                enable_all_cp_ranks_for_transfer=True,
                attn_cp_rank=rank,
                attn_cp_size=8,
            ),
        )
        selected, index, final, skip = functions["_prepare_send_indices"](
            sender, np.arange(pages, dtype=np.int32)
        )
        assert final and not skip
        owners.extend(selected.tolist())
    assert sorted(owners) == list(range(pages))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
