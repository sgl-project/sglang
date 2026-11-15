"""--enable-kda-tp PD decode: late-bound staging slots (SGLANG_K3_KDA_TP_LATE_STAGE).

CPU:
- The switch (SGLANG_K3_KDA_TP_LATE_STAGE) defaults off and applies only under KDA TP.
- PdStateStaging.bind: binds a free stage, returns the bound one again, returns None (no raise) when every stage is
  taken, and binds again once an import drains or the request leaves.
- Decode (DecodeTransferQueue._serve_kda_stage_reqs): KDA_STAGE_REQs bind in arrival order; a room whose request
  left the queue is dropped; while stages are full, later rooms wait but a second prefill rank of an already-bound
  room is still answered; every prefill rank of the room gets the KDA_STAGE_RSP.
- Prefill (NixlKVManager._kda_stage_pending / _bind_kda_stage): only a last chunk with state and an unbound
  destination waits; it asks each decode peer once; the KDA_STAGE_RSP rewrites the destination and the chunk
  then proceeds; a response for a cleared room is ignored; the bind rewrites only the MAMBA state components,
  so other state components (e.g. sparse-attention index pages) keep their page indices.
"""

import collections
import contextlib
import os
import time
import types
import unittest
import unittest.mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEADS, HEAD_DIM, KERNEL, LAYERS = 8, 4, 4, 3


def _par(world, rank=0):
    return types.SimpleNamespace(
        attn_tp_size=1,
        attn_dp_size=world,
        tp_size=world,
        kda_weight_shard_size=None,
        enable_kda_tp=True,
        attn_dp_rank=rank,
    )


def _staging(world, slots):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
    )
    from sglang.srt.layers.communication import k3_kda_tp

    shape = KimiLinearStateShape.create(
        tp_world_size=world, num_heads=HEADS, head_dim=HEAD_DIM, conv_kernel_size=KERNEL
    )
    params = KimiLinearCacheParams(shape=shape, layers=list(range(LAYERS)))
    pool = types.SimpleNamespace(
        mamba_layer_ids=list(range(LAYERS)),
        state_allocation_region=contextlib.nullcontext,
    )
    with unittest.mock.patch.object(k3_kda_tp, "get_parallel", lambda: _par(world)):
        return k3_kda_tp.PdStateStaging(pool, params, slots, "cpu")


def _late_env(on):
    return unittest.mock.patch.dict(
        os.environ, {"SGLANG_K3_KDA_TP_LATE_STAGE": "1" if on else "0"}
    )


class SwitchTest(unittest.TestCase):
    def test_default_off_and_kda_tp_only(self):
        from sglang.srt.layers.communication import k3_kda_tp

        os.environ.pop("SGLANG_K3_KDA_TP_LATE_STAGE", None)
        with unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 8):
            self.assertFalse(k3_kda_tp.late_stage())
            with _late_env(True):
                self.assertTrue(k3_kda_tp.late_stage())
        with (
            unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 1),
            _late_env(True),
        ):
            self.assertFalse(k3_kda_tp.late_stage())


class BindTest(unittest.TestCase):
    def test_bind_lifecycle(self):
        st = _staging(4, 2)
        a, b = st.bind(-1), st.bind(-2)
        self.assertEqual({a, b}, {0, 1})
        self.assertEqual(st.bind(-1), a)  # second prefill rank: same stage
        self.assertIsNone(st.bind(-3))  # full: wait, no raise
        self.assertEqual(st.num_bound(), 2)
        st.release_slot(-2)  # request left
        self.assertEqual(st.bind(-3), b)
        st.rekey(-1, 7)
        st.queue_import(7)
        pairs = st.take_pending()
        st.finish_import(pairs, None)  # import drained (no event on CPU)
        self.assertEqual(st.bind(-4), a)
        self.assertIsNone(st.stage_of(-1))


class _Staging:
    def __init__(self, n):
        self.num_slots, self.free, self.of = n, list(range(n)), {}

    def stage_of(self, key):
        return self.of.get(key)

    def bind(self, key):
        if key in self.of:
            return self.of[key]
        if not self.free:
            return None
        self.of[key] = self.free.pop(0)
        return self.of[key]

    def num_bound(self):
        return len(self.of)


class _Receiver:
    def __init__(self, sent, ranks=2):
        self.kv_mgr = types.SimpleNamespace(kda_stage_reqs=collections.deque())
        self.bootstrap_infos = [{"rank": r} for r in range(ranks)]
        self.sent = sent

    def _connect_to_bootstrap_server(self, info):
        sent = self.sent

        class Sock:
            def send_multipart(self, parts):
                sent.append((info["rank"], [p.decode() for p in parts]))

        return Sock(), unittest.mock.MagicMock()


class _Items:
    def __init__(self, items):
        self._items = items

    def __bool__(self):
        return bool(self._items)

    def items(self):
        return list(self._items)


def _queue(rooms, stages):
    from sglang.srt.disaggregation.decode import DecodeTransferQueue

    q = object.__new__(DecodeTransferQueue)
    sent = []
    recv = _Receiver(sent)
    items = [
        types.SimpleNamespace(
            req=types.SimpleNamespace(bootstrap_room=r, kda_tp_staged_slot=-(i + 1)),
            kv_receiver=recv,
        )
        for i, r in enumerate(rooms)
    ]
    # The transfer queue's request list: truthiness and items() are all it reads.
    q.queue = _Items(items)
    staging = _Staging(stages)
    q.scheduler = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(
            mamba_pool=types.SimpleNamespace(pd_staging=staging)
        )
    )
    q._kda_stage_waiting, q._kda_stage_stats, q._kda_stage_bound_max = {}, [], 0
    q._kda_stage_logged = time.monotonic()
    return q, recv.kv_mgr.kda_stage_reqs, sent, staging


class DecodeServeTest(unittest.TestCase):
    def test_fifo_full_and_answers(self):
        q, incoming, sent, staging = _queue([10, 11, 12], stages=1)
        incoming.extend([(10, "s"), (11, "s"), (12, "s")])
        q._serve_kda_stage_reqs()
        self.assertEqual(staging.of, {-1: 0})
        self.assertEqual(
            sorted(r for r, m in sent), [0, 1]
        )  # both prefill ranks of room 10
        self.assertEqual(sent[0][1], ["KDA_STAGE_RSP", "10", "s", "0"])
        self.assertEqual(list(q._kda_stage_waiting), [11, 12])
        # a second ask for bound room 10 is answered though the stages are full
        sent.clear()
        incoming.append((10, "s2"))
        q._serve_kda_stage_reqs()
        self.assertEqual([m[2] for _, m in sent], ["s2", "s2"])
        self.assertEqual(list(q._kda_stage_waiting), [11, 12])
        # room 10 imported (stage free again); 11 binds before 12
        del staging.of[-1]
        staging.free.append(0)
        sent.clear()
        q._serve_kda_stage_reqs()
        self.assertEqual({m[1] for _, m in sent}, {"11"})
        self.assertEqual(list(q._kda_stage_waiting), [12])

    def test_left_room_dropped(self):
        q, incoming, sent, staging = _queue([10], stages=1)
        incoming.append((99, "s"))  # not in the queue (aborted)
        q._serve_kda_stage_reqs()
        self.assertEqual((q._kda_stage_waiting, sent, staging.of), ({}, [], {}))


class PrefillTest(unittest.TestCase):
    def _mgr(self, state_types=None):
        import threading

        from sglang.srt.disaggregation.base.conn import StateType
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        m = object.__new__(NixlKVManager)
        m.kv_args = types.SimpleNamespace(state_types=state_types or [StateType.MAMBA])
        m._kda_stage_asked, m._kda_stage_socks, m._kda_stage_lock = (
            {},
            {},
            threading.Lock(),
        )
        m.transfer_infos = {}
        return m

    def _tinfo(self, idx):
        return types.SimpleNamespace(
            room=5,
            endpoint="127.0.0.1",
            dst_port=1,
            is_dummy=False,
            dst_state_indices=[[idx]],
        )

    def test_wait_ask_once_then_bind(self):
        m = self._mgr()
        sent = []
        sock = types.SimpleNamespace(
            send_multipart=lambda p: sent.append([x.decode() for x in p])
        )
        ctx = types.SimpleNamespace(
            socket=lambda kind: types.SimpleNamespace(
                setsockopt=lambda *a: None,
                connect=lambda ep: None,
                send_multipart=sock.send_multipart,
            )
        )
        tinfo = self._tinfo(-1)
        m.transfer_infos[5] = {"agent": tinfo}
        last = types.SimpleNamespace(is_last_chunk=True, state_indices=[3])
        with unittest.mock.patch("zmq.Context.instance", lambda: ctx):
            self.assertTrue(m._kda_stage_pending(last, m.transfer_infos[5]))
            self.assertTrue(m._kda_stage_pending(last, m.transfer_infos[5]))
        self.assertEqual(sent, [["KDA_STAGE_REQ", "5", "agent"]])  # asked once
        m._bind_kda_stage([b"KDA_STAGE_RSP", b"5", b"agent", b"2"])
        self.assertEqual(tinfo.dst_state_indices, [[2]])
        self.assertFalse(m._kda_stage_pending(last, m.transfer_infos[5]))
        self.assertEqual(m._kda_stage_asked, {})

    def test_only_unbound_last_chunk_waits(self):
        m = self._mgr()
        bound = {"agent": self._tinfo(1)}
        unbound = {"agent": self._tinfo(-1)}
        mid = types.SimpleNamespace(is_last_chunk=False, state_indices=[3])
        nostate = types.SimpleNamespace(is_last_chunk=True, state_indices=None)
        last = types.SimpleNamespace(is_last_chunk=True, state_indices=[3])
        self.assertFalse(m._kda_stage_pending(mid, unbound))
        self.assertFalse(m._kda_stage_pending(nostate, unbound))
        self.assertFalse(m._kda_stage_pending(last, bound))

    def test_bind_keeps_other_state_components(self):
        from sglang.srt.disaggregation.base.conn import StateType

        m = self._mgr([StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL])
        tinfo = self._tinfo(-1)
        tinfo.dst_state_indices = [[-1], [2, 3, 4, 5], [7]]
        m.transfer_infos[5] = {"agent": tinfo}
        m._bind_kda_stage([b"KDA_STAGE_RSP", b"5", b"agent", b"1"])
        self.assertEqual(tinfo.dst_state_indices, [[1], [2, 3, 4, 5], [7]])

    def test_response_for_cleared_room_ignored(self):
        m = self._mgr()
        m._bind_kda_stage(
            [b"KDA_STAGE_RSP", b"9", b"agent", b"0"]
        )  # no transfer info: no raise


# A request that finishes in its prebuilt batch (a 1-token health check on an idle decode
# rank: no forward between admission and finish) returns its late-bound stage.


def _req(row, lazy=True, staged=None, slot=None):
    return types.SimpleNamespace(
        kv=_Kv(row, slot), kda_tp_lazy_mamba=lazy, kda_tp_staged_slot=staged
    )


class _Kv:
    def __init__(self, row, slot):
        self.req_pool_idx = row
        self.mamba_pool_idx = None if slot is None else torch.tensor(slot)
        self.mamba_needs_clear = False

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


def _pool(staging, first_slot=1, n_slots=2, rows=8):
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    pool = types.SimpleNamespace(
        mamba_allocator=MambaSlotAllocator(
            size=n_slots, device="cpu", first_slot=first_slot
        ),
        req_index_to_mamba_index_mapping=torch.zeros(rows, dtype=torch.int32),
        mamba_pool=types.SimpleNamespace(pd_staging=staging),
        free_slots=[],
    )
    from sglang.srt.mem_cache.kv.state_storage import StateStorage

    pool.state = StateStorage.empty("mamba")
    pool.state.pool = pool.mamba_pool
    pool.state.slots = pool.mamba_allocator
    return pool


STAGES = 4


class _LateStage:
    def __enter__(self):
        from sglang.srt.layers.communication import k3_kda_tp

        self._patches = [
            unittest.mock.patch.dict(os.environ, {"SGLANG_K3_KDA_TP_LATE_STAGE": "1"}),
            unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 4),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


def _admit(pool, req):
    """_get_new_prebuilt_batch for one request: bind, give a state slot, queue its import."""
    from sglang.srt.disaggregation.decode import (
        _admit_lazy_mamba_slot,
        _bind_late_stage,
    )

    if not _bind_late_stage(pool, req) or not _admit_lazy_mamba_slot(pool, req):
        return False
    pool.mamba_pool.pd_staging.queue_import(req.kda_tp_staged_slot)
    req.kda_tp_staged_slot = None
    return True


def _health_check(row):
    from sglang.srt.layers.communication import k3_kda_tp

    # Prealloc under late staging: keyed by the request row, no stage bound (bootstrap_room 0
    # sends no KDA_STAGE_REQ).
    return _req(row=row, staged=k3_kda_tp.lazy_stage_key(row))


def _free(pool, req):
    pool.state.release(pool, req)


class PrebuiltFinishTest(unittest.TestCase):
    def setUp(self):
        self.st = _staging(4, STAGES)
        self.pool = _pool(self.st)
        self.pool.enable_mamba_extra_buffer = False

    def test_health_checks_beyond_stage_count_never_starve(self):
        with _LateStage():
            for i in range(STAGES + 2):
                req = _health_check(row=1 + i % 3)
                self.assertTrue(
                    _admit(self.pool, req),
                    f"health check {i + 1} could not bind a stage",
                )
                _free(self.pool, req)  # finished in its prebuilt batch, no forward
                self.assertEqual(
                    sorted(self.st._free),
                    list(range(STAGES)),
                    f"after health check {i + 1}",
                )
                self.assertEqual(self.st._stage_of, {})
                self.assertEqual(
                    self.st.take_pending(), []
                )  # nothing left to import into a freed slot
        self.assertEqual(self.pool.mamba_allocator.available_size(), 2)

    def test_imported_request_and_other_pending_import(self):
        with _LateStage():
            ran, waiting = _health_check(row=1), _health_check(row=2)
            self.assertTrue(_admit(self.pool, ran))
            self.st.finish_import(
                self.st.take_pending(), None
            )  # its first forward imported it
            self.assertTrue(_admit(self.pool, waiting))
            _free(self.pool, ran)  # leaves after decoding: its stage is already back
            self.assertEqual(sorted(self.st._free), [0, 1, 2])
            pending = self.st.take_pending()
            self.assertEqual(
                [slot for _, slot in pending], [int(waiting.kv.mamba_pool_idx)]
            )
            self.st.finish_import(pending, None)
            _free(self.pool, waiting)
            self.st._reclaim()  # its import drained (no event on CPU)
            self.assertEqual(sorted(self.st._free), list(range(STAGES)))


if __name__ == "__main__":
    unittest.main()
