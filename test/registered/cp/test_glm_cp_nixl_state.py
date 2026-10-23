"""CPU regression checks for GLM CP recurrent-state transfer contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.nixl import conn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def manager():
    m = conn.NixlKVManager.__new__(conn.NixlKVManager)
    m._state_layout_rejections = {}
    m._prepared_state_plans = {}
    m.kv_args = SimpleNamespace(
        state_types=[StateType.MAMBA],
        state_data_ptrs=[[100]],
        state_item_lens=[[128]],
        state_dim_per_tensor=[[1]],
        state_layer_ids=[[3]],
    )
    m.is_hybrid_mla_backend = True
    m.state_tp_size = 4
    m.attn_cp_rank = 2
    m.pp_size = 1
    m._send_mamba_state = MagicMock(return_value="state-handle")
    return m


def transfer(m, **updates):
    kwargs = dict(
        peer_name="peer",
        prefill_state_indices=[[0]],
        dst_state_data_ptrs=[[200, 300]],
        dst_state_indices=[[1]],
        dst_gpu_id=6,
        notif="",
        decode_tp_size=4,
        decode_tp_rank=2,
        dst_state_item_lens=[[128, 256]],
        dst_state_dim_per_tensor=[[1, 1]],
        dst_state_layer_ids=[[3, 45]],
    )
    kwargs.update(updates)
    with patch.object(
        conn,
        "get_parallel",
        return_value=SimpleNamespace(enable_cp_tp_group_sharing=True),
    ):
        return list(m.maybe_send_extra(**kwargs))


def test_matching_shard_ignores_extra_decode_mtp_layer():
    m = manager()
    assert transfer(m) == ["state-handle"]
    m._send_mamba_state.assert_called_once()


@pytest.mark.parametrize(
    "updates",
    [
        {"decode_tp_size": 2},
        {"decode_tp_rank": 1},
        {"dst_state_item_lens": [[256, 256]]},
        {"dst_state_dim_per_tensor": [[2, 1]]},
    ],
)
@pytest.mark.parametrize("prepared", [False, True])
def test_incompatible_state_fails_before_transfer(updates, prepared):
    m = manager()
    m._send_prepared_states = MagicMock(return_value=iter(()))
    if prepared:
        m._prepared_state_plans["peer"] = SimpleNamespace(components={0: object()})
    with pytest.raises(RuntimeError):
        transfer(m, **updates)
    m._send_mamba_state.assert_not_called()
    m._send_prepared_states.assert_not_called()


def test_matching_prepared_shard_is_transferred():
    m = manager()
    m._prepared_state_plans["peer"] = SimpleNamespace(components={0: object()})
    m._send_prepared_states = MagicMock(return_value=iter(["prepared-handle"]))
    assert transfer(m) == ["prepared-handle"]
    m._send_prepared_states.assert_called_once()
    m._send_mamba_state.assert_not_called()


@pytest.mark.parametrize("last", [False, True])
def test_empty_chunk_keeps_contiguous_ids_and_final_state(last):
    sender = conn.NixlKVSender.__new__(conn.NixlKVSender)
    sender._send_failed = False
    sender._transfer_start_time = None
    sender._prepare_send_indices = lambda *a: (
        np.array([], dtype=np.int32),
        slice(0, 0),
        last,
        False,
    )
    sender.kv_mgr = SimpleNamespace(add_transfer_request=MagicMock())
    sender.bootstrap_room = 1
    sender.chunk_id = 7
    sender.aux_index = 0
    sender.has_sent = False
    sender._record_transfer_indices = MagicMock()
    sender.send(np.array([], dtype=np.int32))
    assert sender.chunk_id == 7 + int(last)
    assert sender.has_sent == last
    assert sender.kv_mgr.add_transfer_request.call_count == int(last)
