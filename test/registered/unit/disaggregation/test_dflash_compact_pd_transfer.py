import concurrent.futures
import inspect
import logging
import os
import struct
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import sglang.srt.disaggregation.decode as decode_mod
import sglang.srt.disaggregation.utils as disagg_utils
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import (
    DraftKVTransferSpec,
    TransferBackend,
    draft_swa_suffix_start,
    get_dflash_draft_kv_transfer_locs,
    get_dflash_draft_kv_transfer_spec,
    get_legacy_draft_kv_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import spec_prepare_for_decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


SUFFIX_ENV = "SGLANG_DFLASH_PD_DRAFT_SWA_SUFFIX"
DEFERRED_ENV = "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE"


def _scheduler(*, compact=True):
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                sliding_window_size=2047,
                attn=SimpleNamespace(layer_id=i),
            )
        )
        for i in range(5)
    ]
    return SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        draft_worker=SimpleNamespace(
            use_compact_draft_cache=compact,
            draft_window_size=2048,
            draft_model=SimpleNamespace(layers=layers),
            compact_capability=(
                SimpleNamespace(
                    eligible=True,
                    num_layers=5,
                    checkpoint_window_tokens=2048,
                    attention_window_left=2047,
                )
                if compact
                else None
            ),
        ),
    )


def _runtime_bags():
    return SimpleNamespace(
        parallel=SimpleNamespace(pp_size=1, dp_size=1, attn_dcp_size=1, attn_cp_size=1),
        memory=SimpleNamespace(
            disable_radix_cache=True,
            enable_hierarchical_cache=False,
            hicache_storage_backend=None,
            enable_unified_memory=False,
        ),
        disagg=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False,
            disaggregation_decode_extra_slots=0,
        ),
        schedule=SimpleNamespace(page_size=1),
    )


def _bag_patches(bags):
    return patch.multiple(
        disagg_utils,
        get_parallel=Mock(return_value=bags.parallel),
        get_memory=Mock(return_value=bags.memory),
        get_disagg=Mock(return_value=bags.disagg),
        get_schedule=Mock(return_value=bags.schedule),
    )


def _registration(**overrides):
    values = dict(
        room="None",
        endpoint="127.0.0.1",
        dst_port=1,
        mooncake_session_id="session",
        dst_kv_ptrs=[21, 22],
        dst_kv_data_lens=[64, 64],
        dst_aux_ptrs=[],
        dst_state_data_ptrs=[[201, 202]],
        dst_state_data_lens=[[32, 32]],
        dst_tp_rank=0,
        dst_attn_tp_size=2,
        dst_kv_item_len=8,
        dst_state_item_lens=[[4, 4]],
        dst_state_dim_per_tensor=[[]],
        dst_kv_layer_ids=[],
        dst_state_layer_ids=[[0, 0]],
        draft_swa_suffix_enabled=True,
        draft_swa_window_size=2048,
        draft_swa_page_size=1,
    )
    values.update(overrides)
    return KVArgsRegisterInfo(**values)


def _capability_manager():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.attn_tp_size = 2
    mgr.dcp_size = 1
    mgr.kv_args = SimpleNamespace(
        draft_swa_suffix_enabled=True,
        draft_swa_window_size=2048,
        page_size=1,
        state_types=[StateType.DRAFT_SWA],
        state_data_ptrs=[[101, 102]],
        state_data_lens=[[32, 32]],
        state_item_lens=[[4, 4]],
        state_layer_ids=[[0, 0]],
        kv_data_ptrs=[11, 12],
        kv_data_lens=[64, 64],
        kv_item_lens=[8, 8],
    )
    return mgr


def test_registration_wire_defaults_legacy_and_reads_suffix_extension():
    legacy = [
        b"room",
        b"127.0.0.1",
        b"1234",
        b"session",
        struct.pack("Q", 0x1000),
        b"",
        b"",
        b"0",
        b"2",
        b"8",
        b"",
        b"",
        b"",
        b"",
        b"",
        b"0",
        b"1",
        b"0",
    ]
    legacy_info = KVArgsRegisterInfo.from_zmq(legacy)
    assert legacy_info.draft_swa_suffix_enabled is False
    assert legacy_info.draft_swa_window_size == 0
    assert legacy_info.draft_swa_page_size == 0
    assert legacy_info.dst_kv_data_lens == []
    assert legacy_info.dst_state_data_lens == []

    # Frame 18 is the upstream staging slot metadata. Compact suffix
    # capabilities are appended after it so older peers can ignore them.
    suffix_info = KVArgsRegisterInfo.from_zmq(legacy + [b"", b"1", b"2048", b"1"])
    assert suffix_info.draft_swa_suffix_enabled is True
    assert suffix_info.draft_swa_window_size == 2048
    assert suffix_info.draft_swa_page_size == 1
    assert suffix_info.dst_kv_data_lens == []
    assert suffix_info.dst_state_data_lens == []

    ranged_info = KVArgsRegisterInfo.from_zmq(
        legacy
        + [
            b"",
            b"1",
            b"2048",
            b"1",
            struct.pack("2Q", 64, 96),
            pack_int_lists([[32, 48]], "Q"),
        ]
    )
    assert ranged_info.dst_kv_data_lens == [64, 96]
    assert ranged_info.dst_state_data_lens == [[32, 48]]

    empty_extension = KVArgsRegisterInfo.from_zmq(legacy + [b"", b"", b"", b""])
    assert empty_extension.draft_swa_suffix_enabled is False
    assert empty_extension.draft_swa_window_size == 0
    assert empty_extension.draft_swa_page_size == 0


def test_suffix_start_and_role_local_compact_locs_are_exact():
    spec = DraftKVTransferSpec(2048, (0, 0), True)

    class Worker:
        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def compact_physical_transfer_locs(self, owner, start, end):
            self.calls.append((owner, start, end))
            return np.arange(start, end, dtype=np.int64) + self.offset

    source = Worker(10_000)
    destination = Worker(20_000)
    source_locs = get_dflash_draft_kv_transfer_locs(source, 7, 3000, spec, 1)
    destination_locs = get_dflash_draft_kv_transfer_locs(destination, 9, 3000, spec, 1)

    assert draft_swa_suffix_start(3000, 2048, 1) == 952
    assert source.calls == [(7, 952, 3000)]
    assert destination.calls == [(9, 952, 3000)]
    assert source_locs[0] == 10_952
    assert destination_locs[0] == 20_952
    assert len(source_locs) == len(destination_locs) == 2048


def test_role_local_legacy_physical_locs_are_exact_and_independent():
    class DraftTableMustNotBeRead:
        @property
        def req_to_token(self):
            raise AssertionError("legacy suffix locs must read only the target table")

    def worker_with_table(rows):
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._compact_physical_layout = None
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor(rows, dtype=torch.int64)
            )
        )
        worker.draft_model_runner = SimpleNamespace(
            req_to_token_pool=DraftTableMustNotBeRead()
        )
        return worker

    source = worker_with_table(
        [
            [0, 0, 0, 0, 0, 0],
            [301, 307, 311, 313, 317, 331],
            [101, 103, 107, 109, 113, 127],
        ]
    )
    destination = worker_with_table(
        [
            [0, 0, 0, 0, 0, 0],
            [211, 223, 227, 229, 233, 239],
            [401, 409, 419, 421, 431, 433],
        ]
    )
    spec = DraftKVTransferSpec(4, (0, 0), True, False)

    source_locs = get_dflash_draft_kv_transfer_locs(source, 2, 6, spec, 1)
    destination_locs = get_dflash_draft_kv_transfer_locs(destination, 1, 6, spec, 1)

    assert source_locs.tolist() == [107, 109, 113, 127]
    assert destination_locs.tolist() == [227, 229, 233, 239]
    assert set(source_locs.tolist()).isdisjoint(destination_locs.tolist())


def test_legacy_physical_locs_reject_padding_sentinel():
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker._compact_physical_layout = None
    worker.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[0, 0], [11, 0]], dtype=torch.int64)
        )
    )
    spec = DraftKVTransferSpec(2, (0, 0), True, False)
    with pytest.raises(RuntimeError, match="unallocated/padding"):
        get_dflash_draft_kv_transfer_locs(worker, 1, 2, spec, 1)


@patch.dict(os.environ, {SUFFIX_ENV: "1", DEFERRED_ENV: "1"})
def test_static_compact_gate_uses_runtime_config_bags():
    bags = _runtime_bags()
    pool = SimpleNamespace(layer_num=5, page_size=1, use_hnd=False)
    with _bag_patches(bags):
        spec = get_dflash_draft_kv_transfer_spec(
            _scheduler(), pool, TransferBackend.MOONCAKE
        )
    assert spec == DraftKVTransferSpec(2048, tuple(range(5)) * 2, True)


@patch.dict(os.environ, {SUFFIX_ENV: "1", DEFERRED_ENV: "1"})
def test_static_legacy_physical_gate_uses_same_suffix_capability():
    bags = _runtime_bags()
    pool = SimpleNamespace(layer_num=5, page_size=1, use_hnd=False)
    with _bag_patches(bags):
        spec = get_dflash_draft_kv_transfer_spec(
            _scheduler(compact=False), pool, TransferBackend.MOONCAKE
        )

    assert spec == DraftKVTransferSpec(2048, tuple(range(5)) * 2, True, False)


@pytest.mark.parametrize(
    ("bag", "field", "value", "message"),
    [
        ("parallel", "pp_size", 2, "pipeline parallelism"),
        ("parallel", "dp_size", 2, "data parallelism"),
        ("parallel", "attn_dcp_size", 2, "DCP"),
        ("parallel", "attn_cp_size", 2, "context parallelism"),
        ("disagg", "disaggregation_decode_extra_slots", 1, "extra slots"),
        ("memory", "disable_radix_cache", False, "radix cache"),
        (
            "disagg",
            "disaggregation_decode_enable_radix_cache",
            True,
            "decode radix cache",
        ),
        ("memory", "enable_hierarchical_cache", True, "HiCache"),
    ],
)
@patch.dict(os.environ, {SUFFIX_ENV: "1", DEFERRED_ENV: "1"})
def test_static_compact_gate_rejects_unsupported_bags(bag, field, value, message):
    bags = _runtime_bags()
    setattr(getattr(bags, bag), field, value)
    pool = SimpleNamespace(layer_num=5, page_size=1, use_hnd=False)
    with _bag_patches(bags), pytest.raises(RuntimeError, match=message):
        get_dflash_draft_kv_transfer_spec(_scheduler(), pool, TransferBackend.MOONCAKE)


def test_static_compact_gate_requires_suffix_and_deferred_release(monkeypatch):
    bags = _runtime_bags()
    pool = SimpleNamespace(layer_num=5, page_size=1, use_hnd=False)
    monkeypatch.delenv(SUFFIX_ENV, raising=False)
    monkeypatch.setenv(DEFERRED_ENV, "1")
    with _bag_patches(bags), pytest.raises(RuntimeError, match="suffix protocol"):
        get_dflash_draft_kv_transfer_spec(_scheduler(), pool, TransferBackend.MOONCAKE)

    monkeypatch.setenv(SUFFIX_ENV, "1")
    monkeypatch.delenv(DEFERRED_ENV, raising=False)
    with _bag_patches(bags), pytest.raises(RuntimeError, match="deferred decode"):
        get_dflash_draft_kv_transfer_spec(_scheduler(), pool, TransferBackend.MOONCAKE)

    # With suffix mode disabled, non-compact DFlash retains the historical
    # full-pool route for backward compatibility.
    monkeypatch.delenv(SUFFIX_ENV, raising=False)
    noncompact = _scheduler(compact=False)
    assert (
        get_dflash_draft_kv_transfer_spec(noncompact, pool, TransferBackend.MOONCAKE)
        is None
    )

    # Once suffix mode is selected, legacy physical slots need the same drain
    # ownership guarantee as compact owner-local slots.
    monkeypatch.setenv(SUFFIX_ENV, "1")
    with _bag_patches(bags), pytest.raises(RuntimeError, match="deferred decode"):
        get_dflash_draft_kv_transfer_spec(noncompact, pool, TransferBackend.MOONCAKE)


@patch.dict(os.environ, {SUFFIX_ENV: "1", DEFERRED_ENV: "1"})
def test_static_compact_gate_rejects_backend_and_page_size():
    bags = _runtime_bags()
    pool = SimpleNamespace(layer_num=5, page_size=1, use_hnd=False)
    with _bag_patches(bags), pytest.raises(RuntimeError, match="not Mooncake"):
        get_dflash_draft_kv_transfer_spec(_scheduler(), pool, TransferBackend.NIXL)

    bags.schedule.page_size = 16
    pool.page_size = 16
    with _bag_patches(bags), pytest.raises(RuntimeError, match="page_size"):
        get_dflash_draft_kv_transfer_spec(_scheduler(), pool, TransferBackend.MOONCAKE)


def test_draft_transfer_spec_is_keyword_only_and_state_is_independent():
    parameter = inspect.signature(setup_state_kv_args).parameters[
        "draft_kv_transfer_spec"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    args = SimpleNamespace()
    draft_pool = SimpleNamespace(
        get_contiguous_buf_infos=lambda: ([101, 102], [1000, 1000], [4, 4])
    )
    setup_state_kv_args(
        args,
        object(),
        draft_pool,
        draft_kv_transfer_spec=DraftKVTransferSpec(2048, (3, 3), True),
    )
    assert args.state_types == [StateType.DRAFT_SWA]
    assert args.state_data_ptrs == [[101, 102]]
    assert args.state_item_lens == [[4, 4]]
    assert args.state_layer_ids == [[3, 3]]


def test_suffix_main_kv_excludes_both_physical_draft_layouts():
    infos = ([101, 102], [1000, 1000], [4, 4])
    draft_pool = SimpleNamespace(get_contiguous_buf_infos=Mock(return_value=infos))
    compact_spec = DraftKVTransferSpec(2048, (3, 3), True)
    legacy_physical_spec = DraftKVTransferSpec(2048, (3, 3), True, False)

    assert get_legacy_draft_kv_buf_infos(draft_pool, compact_spec) is None
    assert get_legacy_draft_kv_buf_infos(draft_pool, legacy_physical_spec) is None
    draft_pool.get_contiguous_buf_infos.assert_not_called()
    assert get_legacy_draft_kv_buf_infos(draft_pool, None) == infos
    draft_pool.get_contiguous_buf_infos.assert_called_once_with()


def test_dflash_prebuilt_builds_first_decode_spec_info():
    batch = SimpleNamespace(
        seq_lens=torch.tensor([9, 17], dtype=torch.int64),
        enable_overlap=False,
    )
    bonus_tokens = torch.tensor([41, 42], dtype=torch.int64)
    future_map = Mock()

    spec_info = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
        batch, bonus_tokens, future_map
    )

    assert isinstance(spec_info, DFlashDraftInputV2)
    assert spec_info.bonus_tokens.tolist() == [41, 42]
    assert spec_info.new_seq_lens.tolist() == [9, 17]
    assert spec_info.future_indices is None
    future_map.publish.assert_not_called()
    future_map.stash.assert_not_called()


def test_process_prebuilt_dispatches_dflash_prepare_for_decode():
    req = SimpleNamespace(
        output_ids=[41],
        grammar=None,
        skip_radix_cache_insert=False,
    )
    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=SimpleNamespace(cache_unfinished_req=Mock()),
        device=torch.device("cpu"),
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        req_pool_indices=torch.tensor([2], dtype=torch.int64),
        seq_lens=torch.tensor([9], dtype=torch.int64),
        enable_overlap=False,
    )
    future_map = Mock()

    ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, future_map)

    assert isinstance(batch.spec_info, DFlashDraftInputV2)
    with patch.object(
        DFlashDraftInputV2, "prepare_for_decode", autospec=True
    ) as prepare, patch(
        "sglang.srt.speculative.spec_utils.mamba_extra_buffer_lazy_enabled",
        return_value=False,
    ):
        spec_prepare_for_decode(batch)
    prepare.assert_called_once_with(batch.spec_info, batch)


def test_peer_gate_is_per_room_equal_tp_and_state_only():
    mgr = _capability_manager()
    assert mgr.can_use_draft_swa_suffix(_registration())
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_attn_tp_size=1))
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_dcp_size=2))
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_state_layer_ids=[[1, 1]]))
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_state_item_lens=[[8, 8]]))
    # Distinct base pointers are insufficient: byte ranges must not overlap.
    assert not mgr.can_use_draft_swa_suffix(
        _registration(dst_kv_ptrs=[21, 190], dst_kv_data_lens=[64, 32])
    )
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_kv_data_lens=[]))
    assert not mgr.can_use_draft_swa_suffix(_registration(dst_state_data_lens=[]))
    assert not mgr.can_use_draft_swa_suffix(
        _registration(draft_swa_suffix_enabled=False)
    )

    mgr.transfer_infos = {
        17: {
            "session-a": SimpleNamespace(
                is_dummy=False, mooncake_session_id="session-a"
            ),
            "dummy": SimpleNamespace(is_dummy=True, mooncake_session_id="dummy"),
        }
    }
    mgr.decode_kv_args_table = {
        "session-a": _registration(mooncake_session_id="session-a")
    }
    assert mgr.can_use_draft_swa_suffix_for_room(17)
    mgr.decode_kv_args_table["session-a"] = _registration(dst_attn_tp_size=1)
    assert not mgr.can_use_draft_swa_suffix_for_room(17)


def test_compact_sender_rejects_dynamic_negotiation_failure():
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.bootstrap_room = 17
    sender._draft_swa_suffix_active = None
    sender.kv_mgr = SimpleNamespace(
        kv_args=SimpleNamespace(
            draft_swa_suffix_enabled=True,
            draft_swa_window_size=2048,
            page_size=1,
        ),
        can_use_draft_swa_suffix_for_room=Mock(return_value=False),
        room_has_draft_swa_suffix_peer=Mock(return_value=False),
    )
    with pytest.raises(RuntimeError, match="refusing legacy full-pool transfer"):
        sender.can_send_draft_swa_suffix()


def test_prefill_without_suffix_rejects_decode_suffix_wire_layout():
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.bootstrap_room = 18
    sender._draft_swa_suffix_active = None
    sender.kv_mgr = SimpleNamespace(
        kv_args=SimpleNamespace(draft_swa_suffix_enabled=False),
        can_use_draft_swa_suffix_for_room=Mock(return_value=False),
        room_has_draft_swa_suffix_peer=Mock(return_value=True),
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="decode peer requires"):
            sender.can_send_draft_swa_suffix()


def test_draft_state_send_is_exact_flat_and_layer_paired():
    mgr = _capability_manager()
    mgr.is_mla_backend = False
    mgr.pp_size = 1
    mgr.kv_args.state_dim_per_tensor = [[]]
    mgr.kv_args.state_conv_shard_groups = [[]]
    mgr.kv_args.state_slice_outer_counts = [[]]
    mgr._send_kvcache_generic = Mock(return_value=0)
    req = SimpleNamespace(mooncake_session_id="session", dst_state_indices=[[7, 8]])
    registration = _registration()

    assert mgr.maybe_send_extra(req, [[3, 4]], None, registration) == 0
    kwargs = mgr._send_kvcache_generic.call_args.kwargs
    assert kwargs["src_data_ptrs"] == [101, 102]
    assert kwargs["dst_data_ptrs"] == [201, 202]
    assert kwargs["force_flat"] is True
    assert kwargs["src_layer_ids"] == [0, 0]
    assert kwargs["dst_layer_ids"] == [0, 0]
    np.testing.assert_array_equal(kwargs["prefill_data_indices"], [3, 4])
    np.testing.assert_array_equal(kwargs["dst_data_indices"], [7, 8])

    with pytest.raises(RuntimeError, match="index length mismatch"):
        mgr.maybe_send_extra(req, [[3]], None, registration)


def test_transfer_metric_counts_each_state_component_layout_once():
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender._transfer_num_kv_indices = 4
    sender._transfer_num_state_indices = [2, 5]
    sender._transfer_metric = SimpleNamespace(transfer_total_bytes=0)
    sender.kv_mgr = SimpleNamespace(
        kv_item_lens_sum=16,
        state_item_lens_sums=[8, 3],
        get_kv_replica_factor=lambda: 1,
    )
    assert sender.get_transfer_metric().transfer_total_bytes == 95


@pytest.mark.parametrize("latched", [False, True])
def test_sender_failed_status_waits_for_source_drain(latched):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.bootstrap_room = 23
    sender.conclude_state = KVPoll.Failed if latched else None
    sender.trace_ctx = Mock()
    sender._draft_swa_suffix_active = False
    sender._draft_swa_suffix_completion_logged = False
    sender.kv_mgr = SimpleNamespace(
        _staging_outstanding={23: 1},
        check_status=Mock(return_value=KVPoll.Failed),
    )
    assert sender.poll() == KVPoll.Transferring
    sender.kv_mgr._staging_outstanding.clear()
    assert sender.poll() == KVPoll.Failed


def test_sender_records_compact_completion(caplog):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.bootstrap_room = 24
    sender.conclude_state = None
    sender.trace_ctx = Mock()
    sender._draft_swa_suffix_active = True
    sender._draft_swa_suffix_completion_logged = False
    sender._draft_swa_suffix_state_indices_sent = 2048
    sender._full_draft_indices_omitted = 3000
    sender._transfer_num_kv_indices = 3000
    sender._transfer_num_state_indices = [2048]
    sender._transfer_metric = SimpleNamespace(transfer_total_bytes=0)
    sender.kv_mgr = SimpleNamespace(
        _staging_outstanding={},
        check_status=Mock(return_value=KVPoll.Success),
        kv_item_lens_sum=16,
        state_item_lens_sums=[8],
        get_kv_replica_factor=lambda: 1,
    )
    with caplog.at_level(logging.INFO):
        assert sender.poll() == KVPoll.Success
    assert sender._draft_swa_suffix_completion_logged
    assert '"event":"completed"' in caplog.text
    assert '"draft_state_indices":2048' in caplog.text


@pytest.mark.parametrize(
    ("require_drain_ack", "expect_release"), [(True, False), (False, True)]
)
def test_deferred_release_timeout_holds_suffix_but_releases_legacy_full_route(
    require_drain_ack, expect_release, caplog
):
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.deferred_kv_release_timeout = 30.0
    queue.require_drain_ack_for_deferred_release = require_drain_ack
    queue._do_release = Mock()
    kv_mgr = SimpleNamespace(is_abort_release_safe=Mock(return_value=False))
    decode_req = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=31),
        kv_receiver=SimpleNamespace(kv_mgr=kv_mgr),
    )
    queue._deferred_releases = [(decode_req, 50.0, 7, 2)]

    with patch.object(
        decode_mod.time, "monotonic", return_value=100.0
    ), caplog.at_level(logging.ERROR):
        queue.resolve_deferred_releases()

    assert queue._do_release.called is expect_release
    if require_drain_ack:
        assert len(queue._deferred_releases) == 1
        assert queue._deferred_releases[0][1] == 130.0
        assert "continuing to hold" in caplog.text
    else:
        assert queue._deferred_releases == []


@patch.dict(os.environ, {SUFFIX_ENV: "1", DEFERRED_ENV: "1"})
def test_legacy_physical_suffix_requires_permanent_drain_ack_hold():
    queue = DecodeTransferQueue(
        gloo_group=None,
        req_to_metadata_buffer_idx_allocator=None,
        tp_rank=0,
        metadata_buffers=None,
        scheduler=_scheduler(compact=False),
        tree_cache=None,
    )
    assert queue.require_drain_ack_for_deferred_release is True


def test_outstanding_chunk_terminal_cleanup_is_idempotent():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr._staging_outstanding = {41: 1}
    mgr.enable_deferred_decode_kv_release = True
    mgr._maybe_ack_drained_abort = Mock()
    chunk = SimpleNamespace(room=41, staging_counted=True)

    mgr._finish_outstanding_chunk(chunk)
    mgr._finish_outstanding_chunk(chunk)

    assert mgr._staging_outstanding == {}
    assert chunk.staging_counted is False
    mgr._maybe_ack_drained_abort.assert_called_once_with(41)


def test_future_exception_drains_running_rdma_before_outstanding_ack():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.enable_deferred_decode_kv_release = True
    mgr._staging_outstanding = {42: 1}
    mgr._maybe_ack_drained_abort = Mock()
    chunk = SimpleNamespace(room=42, staging_counted=True)
    blocked_started = threading.Event()
    release_blocked = threading.Event()
    controller_done = threading.Event()

    def blocked_transfer():
        blocked_started.set()
        release_blocked.wait(timeout=5)
        return 0

    def failed_transfer():
        raise RuntimeError("rdma failed")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        blocked = executor.submit(blocked_transfer)
        assert blocked_started.wait(timeout=1)
        failed = executor.submit(failed_transfer)

        def await_then_finish():
            try:
                mgr._await_transfer_futures([failed, blocked])
            except RuntimeError:
                pass
            finally:
                mgr._finish_outstanding_chunk(chunk)
                controller_done.set()

        controller = threading.Thread(target=await_then_finish)
        controller.start()
        assert not controller_done.wait(timeout=0.1)
        assert mgr._staging_outstanding == {42: 1}
        mgr._maybe_ack_drained_abort.assert_not_called()

        release_blocked.set()
        assert controller_done.wait(timeout=2)
        controller.join(timeout=1)

    assert mgr._staging_outstanding == {}
    mgr._maybe_ack_drained_abort.assert_called_once_with(42)


def _abort_test_manager(room):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.enable_deferred_decode_kv_release = True
    mgr._deferred_abort_ack_tracker = {}
    mgr._receiver_armed_abort_rooms = set()
    mgr.request_status = {room: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = 9000
    return mgr


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _receiver_with_synchronous_ack(room, mgr):
    receiver = _ConcreteReceiver.__new__(_ConcreteReceiver)
    receiver.bootstrap_room = room
    receiver.kv_mgr = mgr
    receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 8000}]
    receiver.abort_notified = False
    receiver.conclude_state = None
    receiver.init_time = 0.0
    receiver.invalidate_cached_bootstrap_infos = Mock()

    class Socket:
        def send_multipart(self, _frames):
            mgr.note_abort_ack(room, 0)

    receiver._connect_to_bootstrap_server = Mock(
        return_value=(Socket(), threading.Lock())
    )
    return receiver


def test_receiver_abort_arms_before_synchronous_ack():
    room = 51
    mgr = _abort_test_manager(room)
    receiver = _receiver_with_synchronous_ack(room, mgr)

    receiver.abort()

    assert receiver.abort_notified
    assert mgr.is_abort_release_safe(room, required_acks=1)
    # The scheduler's historical post-send registration is now idempotent.
    mgr.register_deferred_abort_room(room)
    assert mgr.is_abort_release_safe(room, required_acks=1)


def test_waiting_timeout_arms_ack_and_allows_final_deferred_release():
    room = 52
    mgr = _abort_test_manager(room)
    mgr.waiting_timeout = 1.0
    receiver = _receiver_with_synchronous_ack(room, mgr)
    with patch("sglang.srt.disaggregation.common.conn.time.time", return_value=10.0):
        assert receiver._check_waiting_timeout() == KVPoll.Failed
    assert mgr.is_abort_release_safe(room, required_acks=1)

    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.deferred_kv_release_timeout = 30.0
    queue.require_drain_ack_for_deferred_release = True
    queue._do_release = Mock()
    decode_req = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room),
        kv_receiver=SimpleNamespace(kv_mgr=mgr),
    )
    queue._deferred_releases = [(decode_req, 100.0, 7, 1)]
    with patch.object(decode_mod.time, "monotonic", return_value=20.0):
        queue.resolve_deferred_releases()
    queue._do_release.assert_called_once_with(decode_req, 7)
    assert queue._deferred_releases == []


def test_success_room_with_outstanding_write_retains_abort_ack_target():
    room = 53
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.request_status = {room: KVPoll.Success}
    mgr._staging_outstanding = {room: 1}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_send_failures = {}
    mgr._send_abort_ack = Mock(return_value=True)

    assert not mgr.handle_deferred_abort_notification(room, "127.0.0.1", 9000)
    assert mgr.request_status[room] == KVPoll.Success
    assert mgr._deferred_ack_targets[room] == ("127.0.0.1", 9000)
    mgr._send_abort_ack.assert_not_called()

    mgr._staging_outstanding.clear()
    mgr._maybe_ack_drained_abort(room)
    mgr._send_abort_ack.assert_called_once_with("127.0.0.1", 9000, room)
    assert room not in mgr._deferred_ack_targets


def test_abort_during_running_transfer_keeps_failed_sticky_until_drain_ack():
    room = 55
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.request_status = {room: KVPoll.Transferring}
    mgr._staging_outstanding = {room: 1}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_send_failures = {}
    mgr._send_abort_ack = Mock(return_value=True)

    assert mgr.handle_deferred_abort_notification(room, "127.0.0.1", 9000)
    assert mgr.request_status[room] == KVPoll.Failed
    # Simulate the already-running last-chunk worker reaching its old Success.
    mgr.update_status(room, KVPoll.Success)
    assert mgr.request_status[room] == KVPoll.Failed
    mgr._send_abort_ack.assert_not_called()

    mgr._staging_outstanding.clear()
    mgr._maybe_ack_drained_abort(room)
    mgr._send_abort_ack.assert_called_once_with("127.0.0.1", 9000, room)


def test_abort_ack_send_failure_is_retained_and_retried():
    room = 54
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.attn_tp_rank = 0
    mgr.pp_size = 1
    mgr.attn_cp_size = 1
    mgr.pp_rank = 0
    mgr.attn_cp_rank = 0
    mgr._staging_outstanding = {}
    mgr._deferred_ack_targets = {room: ("127.0.0.1", 9000)}
    mgr._deferred_ack_send_failures = {}
    mgr._send_multipart_locked = Mock(
        side_effect=[RuntimeError("temporary send failure"), None]
    )

    mgr._maybe_ack_drained_abort(room)
    assert room in mgr._deferred_ack_targets
    assert mgr._deferred_ack_send_failures[room] == 1

    mgr.retry_deferred_abort_acks()
    assert room not in mgr._deferred_ack_targets
    assert room not in mgr._deferred_ack_send_failures
    assert mgr._send_multipart_locked.call_count == 2
