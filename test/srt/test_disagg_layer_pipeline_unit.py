import threading
import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVSender,
    TransferKVChunk,
    split_layer_groups,
)


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeKVManager:
    """Lightweight stand-in for MooncakeKVManager used by the sender unit tests.

    Only emulates the surface area MooncakeKVSender touches:
    - request lifecycle / status bookkeeping
    - layer-pipeline gating (`_room_supports_layer_pipeline`)
    - chunk dispatch (`add_transfer_request` -> transfer_queues[shard].put)
    """

    def __init__(
        self,
        *,
        num_layers=10,
        enabled=True,
        group_size=4,
        min_prefill_len=0,
        decode_enabled=True,
        decode_group_size=None,
        decode_kv_dtype="fp8_e4m3",
        prefill_kv_dtype="fp8_e4m3",
        enable_staging=False,
        page_size=1,
        num_queues=1,
    ):
        self.layer_pipeline_enabled = enabled
        self.layer_group_size = group_size
        self.layer_pipeline_min_prefill_len = min_prefill_len
        self.enable_staging = enable_staging
        self.server_args = SimpleNamespace(
            kv_cache_dtype=prefill_kv_dtype,
            dp_size=1,
            load_balance_method="follow_bootstrap_room",
        )
        self.is_dummy_cp_rank = False
        self.enable_all_cp_ranks_for_transfer = False
        # CP layer-shard mode (Phase 3 CP shard). Defaults match
        # production unset: "page" shard semantics.
        self.cp_transfer_shard_mode = "page"
        self.attn_cp_rank = 0
        self.attn_cp_size = 1
        # request_status uses room as key
        self.request_status = {}
        self.session_id = "127.0.0.1:1234"
        # Single dummy req entry per room; suffices for queue dispatch
        self.transfer_infos = {
            7: {
                self.session_id: SimpleNamespace(
                    is_dummy=False, mooncake_session_id=self.session_id
                )
            }
        }
        self.decode_kv_args_table = {
            self.session_id: SimpleNamespace(
                layer_pipeline_enabled=decode_enabled,
                layer_group_size=(
                    decode_group_size if decode_group_size is not None else group_size
                ),
                kv_dtype=decode_kv_dtype,
            )
        }
        self.transfer_queues = [_FakeQueue() for _ in range(num_queues)]
        self.kv_args = SimpleNamespace(page_size=page_size)
        self._num_layers = num_layers
        # #10 fix: sender's _layer_groups_for_send reads `is_mla_backend`
        # to decide whether the staging fallback applies. Default False
        # preserves pre-#10 behavior (staging_enabled ⇒ fallback) for
        # any existing test that didn't set this explicitly.
        self.is_mla_backend = False
    # --- methods MooncakeKVSender / MooncakeKVManager use ---

    def update_status(self, bootstrap_room, status):
        self.request_status[bootstrap_room] = status

    def check_status(self, bootstrap_room):
        return self.request_status.get(bootstrap_room, KVPoll.Bootstrapping)

    def local_num_kv_layers(self):
        return self._num_layers

    def use_layer_cp_shard_for_transfer(self):
        return (
            self.layer_pipeline_enabled
            and self.enable_all_cp_ranks_for_transfer
            and self.attn_cp_size > 1
            and self.cp_transfer_shard_mode == "layer"
        )

    def _room_supports_layer_pipeline(self, room):
        if not self.layer_pipeline_enabled:
            return False
        for req in self.transfer_infos.get(room, {}).values():
            if req.is_dummy:
                continue
            target_info = self.decode_kv_args_table.get(req.mooncake_session_id)
            if target_info is None:
                return False
            if not target_info.layer_pipeline_enabled:
                return False
            if (
                target_info.layer_group_size > 0
                and target_info.layer_group_size != self.layer_group_size
            ):
                return False
            if target_info.kv_dtype != self.server_args.kv_cache_dtype:
                return False
        return True

    def add_transfer_request(
        self,
        bootstrap_room,
        kv_indices,
        index_slice,
        is_last_chunk,
        aux_index=None,
        state_indices=None,
        layer_group_id=0,
        layer_range=None,
        total_layer_groups=1,
        total_chunks_in_request=None,
        transfer_event=None,
        skip_aux_rdma=False,
        trace_ctx=None,
    ):
        # Failed rooms are dropped (mirrors the real implementation).
        if self.check_status(bootstrap_room) == KVPoll.Failed:
            return
        if bootstrap_room not in self.transfer_infos:
            return
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(s.rsplit(":", 1)[1]) for s in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                layer_group_id=layer_group_id,
                layer_range=layer_range,
                total_layer_groups=total_layer_groups,
                kv_dtype=self.server_args.kv_cache_dtype,
                total_chunks_in_request=total_chunks_in_request,
                transfer_event=transfer_event,
                skip_aux_rdma=skip_aux_rdma,
            )
        )


def _make_sender(mgr, room=7):
    sender = MooncakeKVSender(
        mgr,
        bootstrap_addr="127.0.0.1:8998",
        bootstrap_room=room,
        dest_tp_ranks=[0],
        pp_rank=0,
    )
    return sender


def _all_chunks(mgr):
    chunks = []
    for q in mgr.transfer_queues:
        chunks.extend(q.items)
    return chunks


class TestKVArgsRegisterInfo(unittest.TestCase):
    def test_kv_args_register_info_parses_num_main_kv_layers_slot(self):
        """Pin msg[17]: decode-side `num_main_kv_layers` ships after the two staging slots (15, 16)
        so the staging-always-at-final invariant in `_register_kv_args` holds. A reorder of slot
        indices would silently drop the draft-tail fail-loud signal and re-introduce dst-mismain corruption."""
        msg = [
            b"None",                    # 0
            b"127.0.0.1",               # 1
            b"2345",                    # 2
            b"127.0.0.1:1234",          # 3
            b"",                         # 4 packed kv ptrs
            b"",                         # 5 packed aux ptrs
            b"",                         # 6 packed state ptrs
            b"0",                        # 7 dst_tp_rank
            b"1",                        # 8 dst_attn_tp_size
            b"64",                       # 9 dst_kv_item_len
            b"",                         # 10 state item lens
            b"",                         # 11 state dim per tensor
            b"1",                        # 12 layer_pipeline_enabled
            b"4",                        # 13 layer_group_size
            b"fp8_e4m3",                 # 14 kv_dtype
            b"",                         # 15 staging base ptr (absent)
            b"",                         # 16 staging total size (absent)
            b"64",                       # 17 num_main_kv_layers
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_num_main_kv_layers, 64)
        # Staging is still parsed at offset 15 (slot 17 lives after it).
        self.assertIsNone(info.staging)
        # LP fields stay intact alongside the new slot.
        self.assertTrue(info.layer_pipeline_enabled)
        self.assertEqual(info.layer_group_size, 4)


class _SliceCaptureMgr:
    """Minimal stand-in exposing only the surface `_send_kvcache_layer_group` touches. Invokes the
    real layer-group method as an unbound function so the slicing logic is exercised without a real
    MooncakeKVManager / RDMA engine."""

    def __init__(
        self,
        *,
        is_mla,
        layers_per_pp_stage,
        src_k_ptrs,
        src_v_ptrs=None,
        dst_k_ptrs=None,
        dst_v_ptrs=None,
        item_lens=None,
    ):
        self.is_mla_backend = is_mla
        self._layers = layers_per_pp_stage
        self._src_k = src_k_ptrs
        self._src_v = src_v_ptrs
        self._dst_k = dst_k_ptrs
        self._dst_v = dst_v_ptrs
        self._item_lens = item_lens
        self.captured = None
        # Production reads `kv_args.prefill_num_main_kv_layers` for the draft tail; this mock doesn't model draft, so an empty namespace is enough (getattr falls back to None).
        self.kv_args = SimpleNamespace()

    def get_mla_kv_ptrs_with_pp(self, src, dst, **kwargs):
        return self._src_k, self._dst_k, self._layers

    def get_mha_kv_ptrs_with_pp(self, src, dst, **kwargs):
        return self._src_k, self._src_v, self._dst_k, self._dst_v, self._layers

    def _send_kvcache_generic(self, **kwargs):
        self.captured = kwargs
        return 0


def _invoke_layer_group(mgr, **kwargs):
    """Call the real `_send_kvcache_layer_group` on the capture mgr."""
    defaults = dict(
        mooncake_session_id="sess-1",
        prefill_data_indices=np.array([0, 1, 2], dtype=np.int32),
        dst_data_indices=np.array([10, 11, 12], dtype=np.int32),
        executor=None,
    )
    defaults.update(kwargs)
    # src/dst_data_ptrs / item_lens are required by the signature but ignored by _SliceCaptureMgr, so placeholders are safe.
    defaults.setdefault("src_data_ptrs", [])
    defaults.setdefault("dst_data_ptrs", [])
    defaults.setdefault("item_lens", [])
    return MooncakeKVManager._send_kvcache_layer_group(mgr, **defaults)


class TestSendKVCacheLayerGroupSlicingMHA(unittest.TestCase):
    def _make_mgr(self, layers_per_pp_stage=12):
        src_k = [100 + i for i in range(layers_per_pp_stage)]
        src_v = [200 + i for i in range(layers_per_pp_stage)]
        dst_k = [300 + i for i in range(layers_per_pp_stage)]
        dst_v = [400 + i for i in range(layers_per_pp_stage)]
        return src_k, src_v, dst_k, dst_v, _SliceCaptureMgr(
            is_mla=False,
            layers_per_pp_stage=layers_per_pp_stage,
            src_k_ptrs=src_k,
            src_v_ptrs=src_v,
            dst_k_ptrs=dst_k,
            dst_v_ptrs=dst_v,
        )

    def test_mha_item_lens_layout_violation_triggers_assert(self):
        # Issue #8: if item_lens length != 2 * layers_per_pp_stage, K-then-V slicing breaks; the assert must fire instead of corrupting slices silently.
        _, _, _, _, mgr = self._make_mgr(12)
        bogus_item_lens = [1] * 12  # missing the V half
        with self.assertRaises(AssertionError):
            _invoke_layer_group(
                mgr, item_lens=bogus_item_lens, layer_range=(0, 4)
            )


class _StopWorkerSignal(Exception):
    """Sentinel raised by the test queue to break `transfer_worker` out of its `while True`."""


class _OneShotQueue:
    """Yields a fixed list of chunks then raises `_StopWorkerSignal`; the worker's try/except
    converts it into a RuntimeError the test can catch."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0

    def get(self):
        if self._idx >= len(self._chunks):
            raise _StopWorkerSignal("test stop signal")
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class _WorkerEarlySkipMgr:
    """Minimal stand-in for `MooncakeKVManager` exposing only the surface `transfer_worker` reads
    before it would call back into mooncake engine code. Records side effects so the test can assert
    nothing expensive happened on the early-skip path (issue #13 regression)."""

    def __init__(self, room, initial_status):
        self.request_status = {room: initial_status}
        self.transfer_infos = {
            room: {
                "sess-1": SimpleNamespace(
                    is_dummy=False, mooncake_session_id="sess-1"
                )
            }
        }
        self.session_lock = threading.RLock()
        # Mirrors `MooncakeKVManager.__init__`: the failure path does `self.session_failures[sid] += 1`
        # and would KeyError on a plain dict, getting rewrapped into RuntimeError before
        # record_failure / sync_status_to_decode_endpoint run. MUST stay a defaultdict.
        from collections import defaultdict
        self.session_failures = defaultdict(int)
        self.failed_sessions = set()
        self.enable_staging = False
        # Required for the prefill_unique_rank calc; the early-skip path never reaches it but transfer_worker reads them unconditionally on the legacy path.
        self.attn_tp_rank = 0
        self.pp_size = 1
        self.attn_cp_size = 1
        self.pp_rank = 0
        self.attn_cp_rank = 0
        self.is_mla_backend = False
        # Read by the worker's catch-all exception handler when wrapping `_StopWorkerSignal`.
        self.bootstrap_port = 0
        # Watermark tests model LP-on chunks by default; tests asserting LP-off legacy behavior flip this to False.
        self.layer_pipeline_enabled = True
        # #18 Phase 3: transfer_worker calls `_record_layer_group_metric` after `_record_chunk_done`.
        # Non-LP chunks have enqueue_ns=0 so the method silently skips, but the manager needs the
        # state + method bound for the call site to resolve.
        self._lp_metrics_lock = threading.Lock()
        self._lp_chunks_total = 0
        self._lp_chunks_periodic = 0
        self._lp_chunk_ms_samples = []
        self._LP_SAMPLE_BUFFER_CAP = 4096
        # Captured side-effects.
        self.record_failure_calls = []
        self.update_status_calls = []
        self.sync_status_calls = []

    def check_status(self, room):
        return self.request_status.get(room, KVPoll.Bootstrapping)

    def record_failure(self, room, msg):
        self.record_failure_calls.append((room, msg))

    def update_status(self, room, status):
        self.update_status_calls.append((room, status))
        self.request_status[room] = status

    def sync_status_to_decode_endpoint(self, *args, **kwargs):
        self.sync_status_calls.append((args, kwargs))

    def _try_create_staging_strategy(self, *_a, **_kw):
        return None


class _WatermarkProgressMgr(_WorkerEarlySkipMgr):
    """Extends `_WorkerEarlySkipMgr` to drive the success/failure path of `transfer_worker`
    end-to-end, exercising the real `_record_chunk_done` / `_record_aux_sent` /
    `_maybe_sync_success_locked` watermark helpers (issue #6). Per-test controls:
      - send_kvcache_returns / send_aux_returns: per-(chunk x dst-rank) return codes
      - transfer_infos[room]: 1 or N dst entries to drive single/multi-rank
    """

    def __init__(
        self,
        room,
        *,
        num_dst_ranks=1,
        send_kvcache_returns=None,
        send_aux_returns=None,
    ):
        super().__init__(room=room, initial_status=KVPoll.WaitingForInput)
        # Real watermark state the helpers read/write.
        self.layer_pipeline_progress = {}
        self.layer_pipeline_progress_lock = threading.Lock()
        # Build N dst-rank reqs so worker iterates the per-req loop N times per chunk.
        # required_dst_info_num must equal the dict size, mirroring the production fan-out.
        sessions = {}
        for i in range(num_dst_ranks):
            sid = f"sess-{i}"
            sessions[sid] = SimpleNamespace(
                is_dummy=False,
                mooncake_session_id=sid,
                endpoint=f"10.0.0.{i + 1}",
                dst_port=20000 + i,
                room=room,
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=num_dst_ranks,
            )
        self.transfer_infos = {room: sessions}
        # target_rank_registration_info for the success path; is_mla path needs is_mla_backend=True
        # so we hit the plain send_kvcache branch being mocked.
        self.is_mla_backend = True
        self.attn_tp_size = 1
        self.decode_kv_args_table = {
            sid: SimpleNamespace(
                dst_kv_ptrs=[],
                dst_aux_ptrs=[],
                dst_state_data_ptrs=[],
                dst_attn_tp_size=1,
                dst_num_main_kv_layers=None,
                staging=None,
            )
            for sid in sessions.keys()
        }
        # Per-call return code lists; pop(0) so tests assert order too.
        self._send_kvcache_returns = list(send_kvcache_returns or [0] * 64)
        self._send_aux_returns = list(send_aux_returns or [0] * 64)
        self.send_kvcache_calls = 0
        self.send_aux_calls = 0
        self.maybe_send_extra_calls = 0
        # Bind the real watermark helpers to this mock so transfer_worker finds them via attribute
        # lookup and exercises the production state machine (helpers are deliberately NOT mocked).
        import types as _types
        for _name in (
            "_record_chunk_done",
            "_record_aux_sent",
            "_maybe_sync_success_locked",
            "_dispatch_sync_outside_lock",
            "_clear_layer_pipeline_progress",
            "_record_layer_group_metric",
            "pop_layer_pipeline_metrics",
        ):
            setattr(
                self,
                _name,
                _types.MethodType(getattr(MooncakeKVManager, _name), self),
            )

    def send_kvcache(self, *_args, **_kwargs):
        self.send_kvcache_calls += 1
        return self._send_kvcache_returns.pop(0) if self._send_kvcache_returns else 0

    def send_aux(self, *_args, **_kwargs):
        self.send_aux_calls += 1
        return self._send_aux_returns.pop(0) if self._send_aux_returns else 0

    def maybe_send_extra(self, *_args, **_kwargs):
        self.maybe_send_extra_calls += 1


def _wm_chunk(room, group_id, *, is_last, total_groups, total_chunks=None):
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([], dtype=np.int32),
        index_slice=slice(0, 0),
        is_last_chunk=is_last,
        prefill_aux_index=42 if is_last else None,
        state_indices=None,
        layer_group_id=group_id,
        layer_range=(group_id * 4, (group_id + 1) * 4),
        total_layer_groups=total_groups,
        kv_dtype="auto",
        total_chunks_in_request=total_chunks if is_last else None,
    )


def _legacy_chunk(room, *, is_last=True):
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([5], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=is_last,
        prefill_aux_index=42 if is_last else None,
        state_indices=None,
        layer_group_id=0,
        layer_range=None,
        total_layer_groups=1,
        kv_dtype="auto",
        total_chunks_in_request=1 if is_last else None,
    )


class TestPerGroupWatermarkSync(unittest.TestCase):
    """Issue #6: receiver-side success notification must wait for every enqueued chunk to actually
    complete, regardless of completion order, so Phase 2's parallel/out-of-order RDMA cannot trigger
    premature `KVPoll.Success` and silent KV corruption."""

    @staticmethod
    def _run_worker(mgr, chunks):
        queue = _OneShotQueue(chunks)
        with unittest.TestCase().assertRaises(RuntimeError):
            MooncakeKVManager.transfer_worker(
                mgr, queue, executor=None, staging_buffer=None
            )

    def test_partial_arrival_no_premature_sync(self):
        """If a middle group never arrives, sync must never fire."""
        room = 102
        mgr = _WatermarkProgressMgr(room=room, num_dst_ranks=1)
        # Drop g2. g3 carries total_chunks=4 but only 3 chunks complete. Watermark waits forever.
        chunks = [
            _wm_chunk(room, 0, is_last=False, total_groups=4),
            _wm_chunk(room, 1, is_last=False, total_groups=4),
            _wm_chunk(room, 3, is_last=True, total_groups=4, total_chunks=4),
        ]
        self._run_worker(mgr, chunks)

        self.assertEqual(mgr.send_kvcache_calls, 3)
        self.assertEqual(mgr.send_aux_calls, 1)
        self.assertEqual(
            mgr.sync_status_calls,
            [],
            "sync_status MUST NOT fire when chunks are still missing — this "
            "is the silent-data-corruption case Phase 2 watermark prevents",
        )
        # request_status should remain non-Success.
        self.assertNotEqual(mgr.request_status.get(room), KVPoll.Success)

    def test_multi_dst_rank_each_must_complete(self):
        """`required_dst_info_num=2`: every dst rank must independently accumulate (chunks_done >= total) AND aux_sent before sync."""
        room = 104
        mgr = _WatermarkProgressMgr(room=room, num_dst_ranks=2)
        # 2 groups × 2 dst ranks = 4 send_kvcache + 2 send_aux total.
        chunks = [
            _wm_chunk(room, 0, is_last=False, total_groups=2),
            _wm_chunk(room, 1, is_last=True, total_groups=2, total_chunks=2),
        ]
        self._run_worker(mgr, chunks)

        # 2 chunks x 2 dst ranks = 4 send_kvcache; 1 last_chunk x 2 = 2 aux.
        self.assertEqual(mgr.send_kvcache_calls, 4)
        self.assertEqual(mgr.send_aux_calls, 2)
        # Sync fires once per dst rank -> 2 total, both Success.
        self.assertEqual(len(mgr.sync_status_calls), 2)
        for call in mgr.sync_status_calls:
            self.assertEqual(call[0][3], KVPoll.Success)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)

    def test_sync_dispatch_releases_lock_before_zmq_notify(self):
        """Issue #30 regression: `_maybe_sync_success_locked` snapshots endpoints under
        `layer_pipeline_progress_lock`, then the caller dispatches the actual
        `sync_status_to_decode_endpoint` (ZMQ send) OUTSIDE that lock so a blocked decode
        endpoint cannot stall every other room. Verified by acquiring the lock from inside the
        ZMQ notify — non-reentrant `threading.Lock` returns False if anyone still holds it."""
        room = 105
        mgr = _WatermarkProgressMgr(room=room, num_dst_ranks=2)
        lock_was_held_during_zmq = []

        original_sync = mgr.sync_status_to_decode_endpoint

        def _spy_sync(*args, **kwargs):
            acquired = mgr.layer_pipeline_progress_lock.acquire(blocking=False)
            # If we got the lock, nobody held it — that's the contract.
            lock_was_held_during_zmq.append(not acquired)
            if acquired:
                mgr.layer_pipeline_progress_lock.release()
            return original_sync(*args, **kwargs)

        mgr.sync_status_to_decode_endpoint = _spy_sync
        chunks = [
            _wm_chunk(room, 0, is_last=False, total_groups=2),
            _wm_chunk(room, 1, is_last=True, total_groups=2, total_chunks=2),
        ]
        self._run_worker(mgr, chunks)

        # Sanity: the original sync path still emitted the Success notify.
        self.assertEqual(len(mgr.sync_status_calls), 2)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)
        # Spy must have triggered; otherwise the test is a no-op.
        self.assertTrue(
            lock_was_held_during_zmq,
            "spy never observed a sync_status_to_decode_endpoint call — "
            "test setup is broken",
        )
        # The contract: NONE of those calls happened with the watermark lock still held (else #30 regression).
        self.assertFalse(
            any(lock_was_held_during_zmq),
            "layer_pipeline_progress_lock was still held during "
            "sync_status_to_decode_endpoint — issue #30 regression "
            "(ZMQ notify must happen outside the lock)",
        )


class _MultiRoomWatermarkMgr(_WorkerEarlySkipMgr):
    """Multi-room variant of `_WatermarkProgressMgr` for issue #5 stress tests. Drives the real
    watermark helpers across many rooms / dst ranks / arbitrary chunk orderings to expose races the
    single-room scaffolding can't reach. The send_kvcache_fn / send_aux_fn callbacks inject
    per-(room, dst-rank, chunk-index) return codes."""

    def __init__(
        self,
        room_to_dst_ranks: dict,
        *,
        send_kvcache_fn=None,
        send_aux_fn=None,
    ):
        # Pick any room for the base class's single-room init; request_status / transfer_infos are rebuilt next.
        any_room = next(iter(room_to_dst_ranks.keys()))
        super().__init__(room=any_room, initial_status=KVPoll.WaitingForInput)

        # Reset for full multi-room registration.
        self.request_status = {
            room: KVPoll.WaitingForInput for room in room_to_dst_ranks
        }
        self.layer_pipeline_progress = {}
        self.layer_pipeline_progress_lock = threading.Lock()
        self.transfer_infos = {}
        sessions_by_sid = {}
        for room, num_dst in room_to_dst_ranks.items():
            sessions = {}
            for i in range(num_dst):
                sid = f"sess-r{room}-d{i}"
                sess = SimpleNamespace(
                    is_dummy=False,
                    mooncake_session_id=sid,
                    endpoint=f"10.0.{room % 256}.{i + 1}",
                    dst_port=20000 + i,
                    room=room,
                    dst_kv_indices=np.array([], dtype=np.int32),
                    dst_aux_index=0,
                    dst_state_indices=[],
                    required_dst_info_num=num_dst,
                )
                sessions[sid] = sess
                sessions_by_sid[sid] = sess
            self.transfer_infos[room] = sessions

        self.is_mla_backend = True
        self.attn_tp_size = 1
        self.decode_kv_args_table = {
            sid: SimpleNamespace(
                dst_kv_ptrs=[],
                dst_aux_ptrs=[],
                dst_state_data_ptrs=[],
                dst_attn_tp_size=1,
                dst_num_main_kv_layers=None,
                staging=None,
            )
            for sid in sessions_by_sid.keys()
        }

        # Per-call counters (incremented inside callbacks); tests use these for total-call assertions.
        self.send_kvcache_calls = 0
        self.send_aux_calls = 0
        self.maybe_send_extra_calls = 0
        # Per-room call ledgers — easier for stress tests to assert "room R got exactly K send_kvcache calls".
        self.per_room_kv_calls = {room: 0 for room in room_to_dst_ranks}
        self.per_room_aux_calls = {room: 0 for room in room_to_dst_ranks}

        # Injectable return-code policy. Default = always succeed.
        self._send_kvcache_fn = send_kvcache_fn or (lambda room, dst, group: 0)
        self._send_aux_fn = send_aux_fn or (lambda room, dst: 0)
        # Track which (room, dst) the next send_aux reports against; transfer_worker iterates dst ranks
        # in a per-req loop, mirrored here via a counter per room.
        self._kvcache_call_log = []  # (room, dst, group) tuples in call order
        self._aux_call_log = []  # (room, dst) tuples in call order

        # Bind the real watermark helpers (same as `_WatermarkProgressMgr`).
        import types as _types
        for _name in (
            "_record_chunk_done",
            "_record_aux_sent",
            "_maybe_sync_success_locked",
            "_dispatch_sync_outside_lock",
            "_clear_layer_pipeline_progress",
            "_record_layer_group_metric",
            "pop_layer_pipeline_metrics",
        ):
            setattr(
                self,
                _name,
                _types.MethodType(getattr(MooncakeKVManager, _name), self),
            )

    # transfer_worker invokes `send_kvcache(req.mooncake_session_id, kv_indices, dst_kv_ptrs,
    # dst_kv_indices, executor, layer_range=...)`. Session id format is "sess-r{room}-d{dst}",
    # so parse room/dst back out for the per-(room, dst) return-code policy.
    def send_kvcache(self, mooncake_session_id, *_args, **_kwargs):
        import re
        m = re.match(r"sess-r(\d+)-d(\d+)", str(mooncake_session_id))
        room = int(m.group(1)) if m else 0
        dst = int(m.group(2)) if m else 0
        group = self.per_room_kv_calls[room] // max(
            len(self.transfer_infos[room]), 1
        )
        ret = self._send_kvcache_fn(room, dst, group)
        self.send_kvcache_calls += 1
        self.per_room_kv_calls[room] += 1
        self._kvcache_call_log.append((room, dst, group))
        return ret

    # transfer_worker invokes `send_aux(req, prefill_aux_index, dst_aux_ptrs)`; req is a
    # SimpleNamespace from `transfer_infos[room].values()` so `req.room` is set in __init__.
    def send_aux(self, req, *_args, **_kwargs):
        room = req.room
        dst = self.per_room_aux_calls[room] % len(self.transfer_infos[room])
        ret = self._send_aux_fn(room, dst)
        self.send_aux_calls += 1
        self.per_room_aux_calls[room] += 1
        self._aux_call_log.append((room, dst))
        return ret

    def maybe_send_extra(self, *_args, **_kwargs):
        self.maybe_send_extra_calls += 1


def _wm_chunk_for(room, group_id, *, is_last, total_groups, total_chunks=None):
    """Same as `_wm_chunk` but with explicit aux index per room so failures are easier to attribute."""
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([], dtype=np.int32),
        index_slice=slice(0, 0),
        is_last_chunk=is_last,
        prefill_aux_index=(room * 1000 + 42) if is_last else None,
        state_indices=None,
        layer_group_id=group_id,
        layer_range=(group_id * 4, (group_id + 1) * 4),
        total_layer_groups=total_groups,
        kv_dtype="auto",
        total_chunks_in_request=total_chunks if is_last else None,
    )


class TestWatermarkStress(unittest.TestCase):
    """Issue #5: stress the per-(room, dst-rank) watermark state machine under combinatorial chunk
    orderings, many concurrent rooms, and per-room failure injection — beyond the 5 baseline
    scenarios in `TestPerGroupWatermarkSync`. Covers the Phase 2 worker hot path under realistic
    burst traffic (1P2D + 1000s reqs/min)."""

    @staticmethod
    def _run_worker(mgr, chunks):
        queue = _OneShotQueue(chunks)
        with unittest.TestCase().assertRaises(RuntimeError):
            MooncakeKVManager.transfer_worker(
                mgr, queue, executor=None, staging_buffer=None
            )

    def test_32_groups_shuffled_completion_syncs_exactly_once(self):
        """32 layer groups in one room arrive in deterministic-random order with last_chunk landing
        mid-stream. Watermark must never sync until ALL 32 complete (regardless of when is_last lands),
        fire EXACTLY one Success sync when the final straggler completes, and clean up its progress dict."""
        import random
        room = 5000
        total_groups = 32
        mgr = _MultiRoomWatermarkMgr({room: 1})

        # last_chunk is group 31. Shuffle the other 31 around it so `is_last` lands mid-stream.
        all_chunks = [
            _wm_chunk_for(room, gid, is_last=(gid == total_groups - 1),
                          total_groups=total_groups,
                          total_chunks=total_groups if gid == total_groups - 1 else None)
            for gid in range(total_groups)
        ]
        rng = random.Random(0xC0FFEE)
        rng.shuffle(all_chunks)
        # Sanity: ensure the shuffle moved last_chunk away from the end (otherwise degenerate).
        last_pos = next(i for i, c in enumerate(all_chunks) if c.is_last_chunk)
        self.assertLess(
            last_pos, total_groups - 1,
            "shuffle did not relocate the last_chunk — test is degenerate",
        )

        self._run_worker(mgr, all_chunks)

        self.assertEqual(mgr.send_kvcache_calls, total_groups)
        self.assertEqual(mgr.send_aux_calls, 1)
        self.assertEqual(
            len(mgr.sync_status_calls), 1,
            f"expected exactly 1 Success sync after all {total_groups} "
            f"chunks complete; got {len(mgr.sync_status_calls)}",
        )
        self.assertEqual(mgr.sync_status_calls[0][0][3], KVPoll.Success)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)
        self.assertNotIn(
            room, mgr.layer_pipeline_progress,
            "watermark progress dict must be cleared on Success",
        )

    def test_one_room_failure_does_not_taint_other_rooms(self):
        """5 rooms x 4 groups; room R_FAIL's first chunk returns -1. The failing room emits Failed
        once and cleans up; the other 4 rooms must still succeed independently. Pins the per-room
        isolation contract production 1P2D depends on — one decode-side OOM / network blip cannot
        stall or contaminate every other in-flight prefill."""
        rooms = [7000, 7001, 7002, 7003, 7004]
        room_fail = 7002
        groups_per_room = 4

        def kv_policy(room, dst, group):
            # Fail only on room_fail's very first chunk.
            if room == room_fail and group == 0:
                return -1
            return 0

        mgr = _MultiRoomWatermarkMgr(
            {r: 1 for r in rooms},
            send_kvcache_fn=kv_policy,
        )

        # Build chunks interleaved across rooms so the failure happens mid-stream, not at queue start.
        all_chunks = []
        for gid in range(groups_per_room):
            for r in rooms:
                all_chunks.append(_wm_chunk_for(
                    r, gid,
                    is_last=(gid == groups_per_room - 1),
                    total_groups=groups_per_room,
                    total_chunks=groups_per_room
                    if gid == groups_per_room - 1 else None,
                ))

        self._run_worker(mgr, all_chunks)

        # Failing room: 1 Failed sync, no aux call (failure short-circuits before is_last_chunk), watermark cleared.
        self.assertEqual(mgr.request_status[room_fail], KVPoll.Failed)
        self.assertNotIn(room_fail, mgr.layer_pipeline_progress)

        # Surviving rooms: each Success.
        for r in rooms:
            if r == room_fail:
                continue
            self.assertEqual(
                mgr.request_status[r], KVPoll.Success,
                f"room {r} should not be tainted by room {room_fail}'s failure",
            )
            self.assertNotIn(r, mgr.layer_pipeline_progress)

        # Sync calls: 4 Success + 1 Failed (the failure branch emits sync_status_to_decode_endpoint with Failed around mooncake/conn.py:2305+).
        failed_syncs = [
            c for c in mgr.sync_status_calls if c[0][3] == KVPoll.Failed
        ]
        success_syncs = [
            c for c in mgr.sync_status_calls if c[0][3] == KVPoll.Success
        ]
        self.assertEqual(len(failed_syncs), 1,
                         "exactly one Failed sync expected for the bad room")
        self.assertEqual(len(success_syncs), len(rooms) - 1,
                         "every other room must produce exactly one Success sync")


class _FakeCudaEvent:
    """Stand-in for `torch.cuda.Event` so tests can drive the Phase 2 transfer-event sync path without a GPU.
    Records every `synchronize()` call and exposes the count for assertions."""

    def __init__(self):
        self.record_calls = 0
        self.synchronize_calls = 0

    def record(self):
        self.record_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1


class TestTransferEventSynchronize(unittest.TestCase):
    """Phase 2 commit 2: when a chunk carries a `transfer_event` (recorded by the forward hook on the
    compute stream), `transfer_worker` MUST `event.synchronize()` before any KV send so RDMA reads
    committed bytes. Phase 1 chunks (event=None) take no extra cost."""

    def test_event_synchronize_called_before_kv_send(self):
        room = 200
        mgr = _WatermarkProgressMgr(room=room, num_dst_ranks=1)
        ev = _FakeCudaEvent()
        # Plain layer-group chunk + event attached.
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=42,
            state_indices=None,
            layer_group_id=0,
            layer_range=(0, 4),
            total_layer_groups=1,
            kv_dtype="auto",
            total_chunks_in_request=1,
            transfer_event=ev,
        )
        TestPerGroupWatermarkSync._run_worker(mgr, [chunk])

        # The KV send fires once (1 dst rank x 1 chunk), aux send fires once (is_last_chunk=True).
        # The event must have been synced AT LEAST ONCE before either side-effect — asserts the looser
        # "synced at all" because hoisting the wait out of the per-req loop is an internal optimization.
        self.assertGreaterEqual(ev.synchronize_calls, 1)
        self.assertEqual(mgr.send_kvcache_calls, 1)
        self.assertEqual(mgr.send_aux_calls, 1)


class TestMakeLayerPipelineHook(unittest.TestCase):
    """Phase 2 commit 3: `MooncakeKVManager.make_layer_pipeline_hook` returns a closure that, on every
    layer-group boundary (and on the last layer), records a single CUDA event and enqueues one chunk per
    dispatch entry. Sub-group-boundary layers must be no-ops."""

    def _make_mgr_with_hook(
        self,
        *,
        num_layers=8,
        group_size=4,
        num_dispatch=2,
        prefill_start_layer=0,
    ):
        from sglang.srt.disaggregation.mooncake.conn import (
            _LayerPipelineRequestDispatch,
            MooncakeKVManager,
        )
        # Can't easily construct a real MooncakeKVManager (needs a transfer engine), so build a minimal
        # stand-in exposing only the surface `make_layer_pipeline_hook` reads.
        mgr = SimpleNamespace(
            layer_group_size=group_size,
            local_num_kv_layers=lambda: num_layers,
            add_transfer_request_calls=[],
            # P0-A (#28): hook reads `kv_args.prefill_start_layer` to convert RadixAttention's global
            # layer_id to the local PP stage index. Tests parametrize prefill_start_layer for PP > 1 paths.
            kv_args=SimpleNamespace(prefill_start_layer=prefill_start_layer),
        )
        def _add(*args, **kwargs):
            mgr.add_transfer_request_calls.append((args, kwargs))
        mgr.add_transfer_request = _add

        # Build dispatch list with fake senders (only read `_hook_enqueued_chunks` to verify count grows).
        dispatch = []
        for i in range(num_dispatch):
            sender = SimpleNamespace(
                _hook_enqueued_chunks=0, bootstrap_room=100 + i
            )
            dispatch.append(_LayerPipelineRequestDispatch(
                room=100 + i,
                sender=sender,
                page_indices=np.array([i], dtype=np.int32),
                index_slice=slice(0, 1),
            ))
        # Bind the real factory to mgr (it doesn't depend on much else).
        import types as _types
        mgr.make_layer_pipeline_hook = _types.MethodType(
            MooncakeKVManager.make_layer_pipeline_hook, mgr
        )
        return mgr, dispatch

    def test_hook_fires_only_on_group_boundaries(self):
        # Patch torch.cuda.Event so we can run without CUDA.
        import torch as _torch
        recorded = []
        class _FakeEv:
            def record(self):
                recorded.append("rec")
        original_event = _torch.cuda.Event
        _torch.cuda.Event = _FakeEv
        try:
            mgr, dispatch = self._make_mgr_with_hook(
                num_layers=8, group_size=4, num_dispatch=2
            )
            hook = mgr.make_layer_pipeline_hook(dispatch)
            # Layers 0,1,2 → no fire. Layer 3 (group boundary) → fire
            # once for 2 dispatch entries. Layer 4,5,6 → no fire. Layer 7
            # (last layer + group boundary) → fire again, also 2 entries.
            extend_fb = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True)
            )
            for layer_id in range(8):
                hook(layer_id, extend_fb)
            self.assertEqual(
                len(mgr.add_transfer_request_calls),
                4,
                "2 group boundaries × 2 dispatch entries = 4 enqueues",
            )
            # Each dispatch entry's sender count incremented twice.
            for entry in dispatch:
                self.assertEqual(entry.sender._hook_enqueued_chunks, 2)
            # Verify the layer_range of first call is (0, 4).
            first_call_kwargs = mgr.add_transfer_request_calls[0][1]
            self.assertEqual(first_call_kwargs["layer_range"], (0, 4))
            self.assertEqual(first_call_kwargs["layer_group_id"], 0)
            self.assertIsNone(first_call_kwargs["aux_index"])
            self.assertFalse(first_call_kwargs["is_last_chunk"])
            # Last fire should be (4, 8).
            last_call_kwargs = mgr.add_transfer_request_calls[-1][1]
            self.assertEqual(last_call_kwargs["layer_range"], (4, 8))
        finally:
            _torch.cuda.Event = original_event


class TestReviewR2HookCeilDivision(unittest.TestCase):
    """R2 (review): the hook must compute `total_layer_groups` with ceil division, not floor —
    num_layers=10, group_size=4 is 3 groups, not 2. Asserts chunk metadata receivers see is correct."""

    def test_total_layer_groups_uses_ceil(self):
        from sglang.srt.disaggregation.mooncake.conn import (
            _LayerPipelineRequestDispatch,
            MooncakeKVManager,
        )
        captured = []
        mgr = SimpleNamespace(
            layer_group_size=4,
            local_num_kv_layers=lambda: 10,
            add_transfer_request=lambda *a, **kw: captured.append(kw),
            kv_args=SimpleNamespace(prefill_start_layer=0),
        )
        sender = SimpleNamespace(_hook_enqueued_chunks=0, bootstrap_room=400)
        dispatch = [_LayerPipelineRequestDispatch(
            room=400,
            sender=sender,
            page_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
        )]
        import torch as _torch
        class _FakeEv:
            def record(self): pass
        original = _torch.cuda.Event
        _torch.cuda.Event = _FakeEv
        try:
            import types as _types
            hook = _types.MethodType(
                MooncakeKVManager.make_layer_pipeline_hook, mgr
            )(dispatch)
            fake_fb = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True)
            )
            for layer_id in range(10):
                hook(layer_id, fake_fb)
        finally:
            _torch.cuda.Event = original

        # 3 fires (layers 3, 7, 9) x 1 dispatch entry = 3 captures.
        self.assertEqual(len(captured), 3)
        # Every chunk must report total_layer_groups=3, not 10//4=2.
        for kw in captured:
            self.assertEqual(
                kw["total_layer_groups"],
                3,
                "ceil(10/4) = 3, not floor(10/4) = 2",
            )


class TestReviewR3EventSyncFailureIsolation(unittest.TestCase):
    """R3 (review): a CUDA error in one room's `event.synchronize()` must not bring down the whole transfer
    worker thread. Mark just that room failed and continue."""

    def test_event_sync_exception_marks_room_failed_and_continues(self):
        room_bad = 500
        room_good = 501
        mgr = _WatermarkProgressMgr(room=room_bad, num_dst_ranks=1)
        # Add a second room so we can verify the worker keeps running after the first one raises.
        mgr.transfer_infos[room_good] = {
            "sess-good": SimpleNamespace(
                is_dummy=False,
                mooncake_session_id="sess-good",
                endpoint="10.0.0.2",
                dst_port=20002,
                room=room_good,
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=1,
            )
        }
        mgr.request_status[room_good] = KVPoll.WaitingForInput
        # Mock the table entry for sess-good so the success path can resolve the registration info.
        mgr.decode_kv_args_table["sess-good"] = SimpleNamespace(
            dst_kv_ptrs=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            dst_attn_tp_size=1,
            dst_num_main_kv_layers=None,
            staging=None,
        )

        class _ExplodingEvent:
            def synchronize(self):
                raise RuntimeError("simulated CUDA context error")

        bad_chunk = TransferKVChunk(
            room=room_bad,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            layer_group_id=0,
            layer_range=(0, 4),
            total_layer_groups=1,
            kv_dtype="auto",
            transfer_event=_ExplodingEvent(),
        )
        good_chunk = TransferKVChunk(
            room=room_good,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=42,
            state_indices=None,
            layer_group_id=0,
            layer_range=(0, 4),
            total_layer_groups=1,
            kv_dtype="auto",
            total_chunks_in_request=1,
        )
        TestPerGroupWatermarkSync._run_worker(mgr, [bad_chunk, good_chunk])

        # Bad room: marked Failed via update_status + record_failure.
        self.assertEqual(mgr.request_status[room_bad], KVPoll.Failed)
        self.assertTrue(
            any(call[0] == room_bad for call in mgr.record_failure_calls),
            "bad room must have a failure recorded",
        )
        # Good room: worker kept going and finished it normally.
        self.assertEqual(mgr.send_kvcache_calls, 1)
        self.assertEqual(mgr.send_aux_calls, 1)
        # At least one Success sync (for the good room) must have fired.
        good_syncs = [c for c in mgr.sync_status_calls
                      if c[0][2] == room_good and c[0][3] == KVPoll.Success]
        self.assertEqual(
            len(good_syncs),
            1,
            "good room must complete successfully even after bad room's "
            "event.synchronize raised — bad room's failure must NOT take "
            "down the whole worker",
        )


def _make_hook_mgr_with_offset(
    *, num_layers, group_size, prefill_start_layer, is_dummy_cp_rank=False,
    num_main_kv_layers=None,
    use_layer_cp_shard=False, attn_cp_size=1, attn_cp_rank=0,
):
    """Build a `make_layer_pipeline_hook`-callable mock manager with a user-supplied PP layer offset.
    Used by P0-A coverage (PP offset), #32 coverage (CP dummy rank guard), and the post-2026-05-25 fix
    for "draft model appends KV ptrs -> hook needs num_main_kv_layers".

    `num_main_kv_layers` defaults to `num_layers` (no draft, all slots are main) so existing call sites
    keep semantics. Pass `num_main_kv_layers < num_layers` to simulate the GLM-MoE-DSA + EAGLE+MTP layout
    where the trailing N slots in `kv_data_ptrs` are draft layers the main forward never visits.

    `use_layer_cp_shard` toggles CP layer-shard mode. The hook closure captures (attn_cp_size, attn_cp_rank)
    via `use_layer_cp_shard_for_transfer()` returning True; non-owner ranks of a given group_id skip
    enqueue + counter bump.
    """
    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVManager,
        _LayerPipelineRequestDispatch,
    )
    import types as _types

    captured = []
    mgr = SimpleNamespace(
        layer_group_size=group_size,
        local_num_kv_layers=lambda: num_layers,
        add_transfer_request=lambda *a, **kw: captured.append(kw),
        kv_args=SimpleNamespace(
            prefill_start_layer=prefill_start_layer,
            prefill_num_main_kv_layers=num_main_kv_layers,
        ),
        is_dummy_cp_rank=is_dummy_cp_rank,
        attn_cp_size=attn_cp_size,
        attn_cp_rank=attn_cp_rank,
        use_layer_cp_shard_for_transfer=lambda: use_layer_cp_shard,
    )
    mgr.make_layer_pipeline_hook = _types.MethodType(
        MooncakeKVManager.make_layer_pipeline_hook, mgr
    )
    sender = SimpleNamespace(_hook_enqueued_chunks=0, bootstrap_room=900)
    dispatch = [
        _LayerPipelineRequestDispatch(
            room=900,
            sender=sender,
            page_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
        )
    ]
    return mgr, dispatch, captured, sender


class TestMakeLayerPipelineHookPpOffset(unittest.TestCase):
    """P0-A (#28): the hook receives the GLOBAL `RadixAttention.layer_id` but `local_num_kv_layers()` /
    `kv_data_ptrs` are sliced to the local PP stage. Hook must subtract `kv_args.prefill_start_layer` and drop
    fires outside `[0, num_layers)`. PP=1 (offset=0) keeps Phase 1 + 35 existing cases bit-equivalent; PP > 1
    with offset > 0 must enqueue chunks with **local** layer_range and ignore foreign layer ids (GPT review #2 latent HIGH)."""

    def _fire(self, mgr, dispatch, layer_ids):
        import torch as _torch

        class _FakeEv:
            def record(self):
                pass

        original = _torch.cuda.Event
        _torch.cuda.Event = _FakeEv
        try:
            hook = mgr.make_layer_pipeline_hook(dispatch)
            extend_fb = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True)
            )
            for lid in layer_ids:
                hook(lid, extend_fb)
        finally:
            _torch.cuda.Event = original

    def test_pp_rank1_local_layer_range_when_owning_layers(self):
        """PP rank 1 with prefill_start_layer=32: layer_id 32..63 are ours. After P0-A's `local_id = layer_id - 32`,
        fires should be at local ids 3,7,11,...,31 with local layer_range (0,4), (4,8), ..., (28,32). Critically
        NOT at global ranges like (32,36) which would index out of bounds in `_send_kvcache_layer_group`."""
        mgr, dispatch, captured, sender = _make_hook_mgr_with_offset(
            num_layers=32, group_size=4, prefill_start_layer=32
        )
        self._fire(mgr, dispatch, range(32, 64))
        self.assertEqual(len(captured), 8, "8 layer groups in 32 local layers")
        expected_ranges = [(g * 4, g * 4 + 4) for g in range(8)]
        for kw, expected in zip(captured, expected_ranges):
            self.assertEqual(kw["layer_range"], expected)
        # layer_group_id is also LOCAL (so receiver-side metric labels don't collide between PP ranks).
        for i, kw in enumerate(captured):
            self.assertEqual(kw["layer_group_id"], i)
        self.assertEqual(sender._hook_enqueued_chunks, 8)


class TestMakeLayerPipelineHookDraftLayerSlots(unittest.TestCase):
    """When a draft (MTP/EAGLE NEXTN) model exists on prefill, its KV pool ptrs get APPENDED to the main
    pool's ptrs in `disaggregation/prefill.py:147-155` so mooncake registers a single contiguous RDMA buffer
    covering both. `len(kv_data_ptrs)` then becomes `num_main + num_draft` (e.g. GLM-MoE-DSA: 78 main + 1
    nextn = 79), but the main forward only iterates `num_main` layers — the draft model has a separate forward
    intentionally never hooked (per #21).

    Original bug (2026-05-25): hook set `last_layer_idx = num_layers - 1 = 78` (the trailing draft slot).
    Main forward never reaches local_id=78, so `is_last` never matched -> final main partial group never
    fired -> main layers `(last_full_boundary, num_main)` silently dropped from LP transfer (production:
    GLM main layers 76 and 77 = 2 layers of KV bytes never reach decode under hook-mode LP).

    First fix attempt (#35, reverted in B2): used `num_main - 1` for `is_last` AND extended `layer_end =
    num_layers` on the final main fire to ship draft KV alongside main 76/77 in one RDMA submit. The H200
    H200 smoke test (2026-05-26) proved this wrong: the LP hook fires from inside
    `forward_target_extend`, BEFORE `forward_draft_extend` runs (both inside the same
    `forward_batch_generation` per eagle_worker.py:299-323). At hook fire time the prefill draft pool still
    holds stale / zero bytes; RDMA snapshotted those and decode L78 ended up all zeros.

    Final design (B2): hook keeps `num_main - 1` for `is_last` (fixes the original "main 76/77 dropped" bug)
    but does NOT extend layer_end into draft. Draft KV is shipped by a SEPARATE `MooncakeKVSender.send_draft_kv`
    call from `prefill.py send_kv_chunk`, which fires AFTER `forward_batch_generation` returns (and thus after
    `forward_draft_extend` has written draft KV) — see `TestMooncakeKVSenderSendDraftKV` below.
    """

    def _fire(self, mgr, dispatch, layer_ids):
        import torch as _torch

        class _FakeEv:
            def record(self):
                pass

        original = _torch.cuda.Event
        _torch.cuda.Event = _FakeEv
        try:
            hook = mgr.make_layer_pipeline_hook(dispatch)
            extend_fb = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True)
            )
            for lid in layer_ids:
                hook(lid, extend_fb)
        finally:
            _torch.cuda.Event = original

    def test_glm_like_78_main_plus_1_draft_main_only_fires(self):
        """GLM-MoE-DSA production layout: 78 main + 1 nextn draft = 79 total slots, group_size=4. Main forward
        fires hook for layer_ids 0..77 (never 78). Expected (B2 design):
          - 19 boundary fires at local_id 3, 7, ..., 75 -> (0,4)..(72,76)
          - 1 partial fire at local_id 77 covering (76, 78) — main only, NOT extended into draft (#35 reverted)
          - Total 20 fires covering main 78 layers ONLY
          - Draft layer 78 is shipped by send_draft_kv (separately tested)
        """
        mgr, dispatch, captured, sender = _make_hook_mgr_with_offset(
            num_layers=79, group_size=4, prefill_start_layer=0,
            num_main_kv_layers=78,
        )
        # Fire all 78 main layers (0..77). Layer 78 is draft (never iterated by main forward).
        self._fire(mgr, dispatch, range(78))

        self.assertEqual(
            len(captured), 20,
            f"expected 20 fires (19 full boundaries + 1 partial); "
            f"original 2026-05-25 bug gave 19, missing layers 76/77",
        )
        ranges = [kw["layer_range"] for kw in captured]
        # First 19 boundaries cover (0,4)..(72,76) — pure main.
        self.assertEqual(
            ranges[:19],
            [(g * 4, g * 4 + 4) for g in range(19)],
        )
        # The 20th (partial) fire covers (76, 78) — main 76,77 ONLY. NOT extended into draft 78 (send_draft_kv's job).
        self.assertEqual(
            ranges[19], (76, 78),
            "final partial fire's layer_range MUST be (76, 78) — main only. "
            "Extending to (76, 79) was the #35 mistake that shipped stale "
            "draft KV before forward_draft_extend ran.",
        )
        # Sender counter for the 20 main LP chunks; send_draft_kv would add a 21st later (tested separately).
        self.assertEqual(sender._hook_enqueued_chunks, 20)


class TestSenderHookMixedPathDetection(unittest.TestCase):
    """P0-B (#28) fail-loud: scheduler sets `_hook_handled_in_current_send` True but no hook fire bumped
    `_hook_enqueued_chunks` in between (eg. the dispatched req's forward path bypassed `RadixAttention.forward`,
    or eligibility flipped mid-batch). `send()` MUST raise rather than emit an aux finalizer with
    `total_chunks_in_request = 1` that contradicts the receiver-side per-(room,dst-rank) watermark count."""

    def _make_sender(self):
        mgr = _FakeKVManager(num_layers=8, group_size=4)
        sender = MooncakeKVSender(
            mgr=mgr,
            bootstrap_addr="bootstrap-host:30002",
            bootstrap_room=7,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        sender.num_kv_indices = 3
        sender.aux_index = 7
        return mgr, sender

    def test_flag_set_but_no_hook_fire_raises(self):
        mgr, sender = self._make_sender()
        # Scheduler dispatched hook mode...
        sender._hook_handled_in_current_send = True
        # ...but no hook fire happened. Counter stays at the previous baseline (0 here for first send).
        self.assertEqual(sender._hook_enqueued_chunks, 0)
        self.assertEqual(sender._hook_chunks_at_last_send, 0)

        with self.assertRaises(RuntimeError) as cm:
            sender.send(np.array([0, 1, 2], dtype=np.int32))
        msg = str(cm.exception)
        self.assertIn("Layer-pipeline contract violated", msg)
        self.assertIn(str(sender.bootstrap_room), msg)
        # Flag was consumed even on error so a follow-up send() with the flag NOT set re-enters Phase 1 cleanly.
        self.assertFalse(sender._hook_handled_in_current_send)


def _make_metrics_mgr():
    """Build a minimal `MooncakeKVManager`-method-callable mock with only the LP metrics state pre-wired.
    Bypasses the real __init__ (which needs RDMA + ZMQ) by binding the two public methods to a SimpleNamespace."""
    import threading
    import types as _types

    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    mgr = SimpleNamespace(
        _lp_metrics_lock=threading.Lock(),
        _lp_chunks_total=0,
        _lp_chunks_periodic=0,
        _lp_chunk_ms_samples=[],
        _LP_SAMPLE_BUFFER_CAP=4096,
    )
    mgr._record_layer_group_metric = _types.MethodType(
        MooncakeKVManager._record_layer_group_metric, mgr
    )
    mgr.pop_layer_pipeline_metrics = _types.MethodType(
        MooncakeKVManager.pop_layer_pipeline_metrics, mgr
    )
    return mgr


class TestLayerPipelineMetrics(unittest.TestCase):
    """#18 Phase 3: per-LP-chunk metrics — Counter + Histogram fed through `MooncakeKVManager._record_layer_group_metric`
    and snapshot via `pop_layer_pipeline_metrics`. Also covers the failure-path regression: failed chunks must NOT
    increment any metric (the `transfer_worker` failure branches return/break before reaching the record call)."""

    def test_reporter_clears_lp_metric_delta_after_log(self):
        """LP metric snapshots are one-shot deltas. Leaving them on the reporter's persistent SchedulerStats would
        make idle logging replay the same counter increment and histogram samples every 30 seconds."""
        import types as _types

        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )

        stats = SimpleNamespace(
            kv_transfer_layer_group_chunks_delta=3,
            kv_transfer_layer_group_ms_samples=[1.0, 2.0],
        )
        seen = []

        def _log_stats(stats_arg):
            seen.append(
                (
                    stats_arg.kv_transfer_layer_group_chunks_delta,
                    list(stats_arg.kv_transfer_layer_group_ms_samples),
                )
            )

        reporter = SimpleNamespace(
            stats=stats,
            metrics_collector=SimpleNamespace(log_stats=_log_stats),
        )
        reporter._clear_transient_metric_deltas = _types.MethodType(
            SchedulerMetricsReporter._clear_transient_metric_deltas, reporter
        )
        reporter._log_stats_and_clear_transient_deltas = _types.MethodType(
            SchedulerMetricsReporter._log_stats_and_clear_transient_deltas,
            reporter,
        )

        reporter._log_stats_and_clear_transient_deltas()
        self.assertEqual(seen, [(3, [1.0, 2.0])])
        self.assertEqual(stats.kv_transfer_layer_group_chunks_delta, 0)
        self.assertEqual(stats.kv_transfer_layer_group_ms_samples, [])

        reporter._log_stats_and_clear_transient_deltas()
        self.assertEqual(
            seen[-1],
            (0, []),
            "second log must not replay the previous LP metric snapshot",
        )


class TestMooncakeKVSenderSendDraftKV(unittest.TestCase):
    """B2: draft KV transfer via separate `send_draft_kv` call from `prefill.py send_kv_chunk`, AFTER
    `forward_batch_generation` returns so the draft model forward has actually written draft KV bytes.

    Pre-B2 (#35 fix): main LP hook's final fire extended layer_range to include draft slots — but that fires
    BEFORE `forward_draft_extend` (eagle_worker.py:299-323 runs target_fwd then draft_fwd sequentially inside
    the same forward_batch_generation), so RDMA snapshotted stale draft pool bytes and decode L78 was all zeros.

    B2 design: main hook covers MAIN ONLY (revert #35's layer_end extension). Draft KV is shipped by
    `send_draft_kv` called from scheduler BEFORE `sender.send` — by then draft fwd has completed. Draft chunk
    counts toward `_hook_enqueued_chunks` so the aux finalizer's `total_chunks_in_request = _hook_enqueued_chunks + 1`
    accounting stays consistent without extra plumbing."""

    def _make_sender(
        self,
        *,
        num_kv_indices=3,
        num_main=8,
        num_draft=1,
        group_size=4,
        is_dummy=False,
    ):
        # `local_num_kv_layers()` in production returns the LOGICAL total (main + draft) — for MHA that's
        # `len(kv_data_ptrs) // 2`. The fake stores `_num_layers` directly; pass `num_main + num_draft` so
        # `_draft_layer_range` sees a unit-consistent total.
        mgr = _FakeKVManager(
            num_layers=num_main + num_draft, group_size=group_size,
        )
        # Augment kv_args with draft-aware fields the new send_draft_kv reads. _FakeKVManager.kv_args defaults to just page_size; extend here without touching the base fixture.
        mgr.kv_args.kv_data_ptrs = [0] * (num_main + num_draft)
        mgr.kv_args.prefill_num_main_kv_layers = num_main
        mgr.kv_args.engine_rank = 0
        mgr.is_dummy_cp_rank = is_dummy
        sender = MooncakeKVSender(
            mgr=mgr,
            bootstrap_addr="bootstrap-host:30002",
            bootstrap_room=7,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        sender.num_kv_indices = num_kv_indices
        sender.aux_index = 7
        return mgr, sender

    def test_aux_finalizer_total_chunks_includes_draft(self):
        """End-to-end accounting: hook fires K times (simulated), then send_draft_kv runs (adds +1), then
        sender.send last_chunk=True emits aux finalizer with total_chunks_in_request = K + 1 (draft) + 1 (aux).
        Receiver-side watermark closes exactly when all K+2 chunks land."""
        mgr, sender = self._make_sender(
            num_main=78, num_draft=1, group_size=4, num_kv_indices=3,
        )
        # Simulate main LP hook fired K=20 chunks across this token chunk
        sender._hook_handled_in_current_send = True
        sender._hook_enqueued_chunks = 20  # main LP fires
        # Now scheduler calls send_draft_kv then sender.send
        sender.send_draft_kv(np.array([3, 4, 5], dtype=np.int32))
        # _hook_enqueued_chunks now 21 (20 main + 1 draft)
        self.assertEqual(sender._hook_enqueued_chunks, 21)
        sender.send(np.array([3, 4, 5], dtype=np.int32), state_indices=None)

        chunks = mgr.transfer_queues[0].items
        self.assertEqual(len(chunks), 2, "1 draft + 1 aux finalizer")
        # First chunk: draft (enqueued by send_draft_kv before sender.send)
        self.assertEqual(chunks[0].layer_range, (78, 79))
        self.assertFalse(chunks[0].is_last_chunk)
        # Second chunk: aux finalizer (enqueued by sender.send last branch)
        self.assertTrue(chunks[1].is_last_chunk)
        self.assertIsNone(chunks[1].layer_range)
        self.assertEqual(
            chunks[1].total_chunks_in_request, 22,
            "20 main + 1 draft + 1 aux = 22; aux finalizer's "
            "total_chunks_in_request must close on all of them",
        )


class TestMergeMainAndDraftKVLayout(unittest.TestCase):
    """Regression for `merge_main_and_draft_kv_layout` MHA reorder.

    `CommonKVManager.get_mha_kv_ptrs_with_pp` slices the merged `kv_data_ptrs` as `[:half], [half:]`
    (first half all K, second all V). A naive `main + draft` concat under MHA produces
    `[K_main, V_main, K_draft, V_draft]` and the `//2` split tangles K with V — silent KV corruption.
    The helper reorders to `[K_main, K_draft, V_main, V_draft]` to preserve the invariant."""

    def _import_helper(self):
        from sglang.srt.disaggregation.utils import (
            merge_main_and_draft_kv_layout,
        )
        return merge_main_and_draft_kv_layout

    def test_mha_reorder_keeps_kv_halves_pure(self):
        """MHA happy path: 3 main layers (K=[100,101,102], V=[200,201,202]) + 1 draft layer (K=[110], V=[210]).
        After merge the first half must be all K ptrs (main K then draft K), second half all V."""
        merge = self._import_helper()
        # MHA layout: [K_0, K_1, K_2, V_0, V_1, V_2]
        main_ptrs = [100, 101, 102, 200, 201, 202]
        draft_ptrs = [110, 210]
        ptrs, _, _ = merge(
            main_ptrs, main_ptrs, main_ptrs,
            draft_ptrs, draft_ptrs, draft_ptrs,
            is_mla_backend=False,
        )
        self.assertEqual(len(ptrs), 8)
        half = len(ptrs) // 2
        # First half: K main then K draft
        self.assertEqual(ptrs[:half], [100, 101, 102, 110])
        # Second half: V main then V draft
        self.assertEqual(ptrs[half:], [200, 201, 202, 210])


class TestGetStatePtrsWithPp(unittest.TestCase):
    """Regression for the SWA/DSA state-ptr PP alignment bug in `MooncakeKVManager.maybe_send_extra`.

    Before the fix the per-LP-chunk state path applied the local `layer_range = (layer_start, layer_end)` slice
    identically to both src and dst state pointer lists. That is correct only when prefill and decode use matching
    PP layouts (both PP=1, or both PP=N). For a typical production deployment — prefill PP > 1, decode PP=1 — the
    dst state list is the FULL model while src is one local PP slice. Identical slicing then picks dst layers [0, K)
    instead of [prefill_start_layer, prefill_start_layer + K), silently corrupting decode-side state pages.

    These tests pin the helper's contract and the end-to-end behavior inside `maybe_send_extra` for the LP path with PP > 0."""

    def _make_mgr(self, *, prefill_start_layer: int):
        """Lightweight CommonKVManager stand-in: only exposes the surface `get_state_ptrs_with_pp` reads."""
        from sglang.srt.disaggregation.common.conn import CommonKVManager
        import types
        fake = SimpleNamespace(
            kv_args=SimpleNamespace(prefill_start_layer=prefill_start_layer),
        )
        fake.get_state_ptrs_with_pp = types.MethodType(
            CommonKVManager.get_state_ptrs_with_pp, fake
        )
        return fake

    def test_pp_rank_with_draft_requires_dst_num_main(self):
        """User-facing fail-loud: when src has a draft tail, decode MUST advertise ``num_main_kv_layers`` via
        registration. Silently inferring ``dst_main_count = len(dst) - local_draft_count`` reinterprets a main layer
        as a draft slot in the asymmetric case (e.g. prefill has draft + decode does not) and yields KV corruption.
        The helper raises so the deployment fails at registration replay time instead of corrupting bytes."""
        mgr = self._make_mgr(prefill_start_layer=32)
        src = [1000 + i for i in range(16)] + [9000]   # 16 main + 1 draft
        dst = [2000 + i for i in range(64)] + [8000]    # 64 main + 1 draft
        lens = [10] * 17
        with self.assertRaises(ValueError) as cm:
            mgr.get_state_ptrs_with_pp(
                src, dst, lens, num_main_local=16,
            )
        self.assertIn("num_main_kv_layers", str(cm.exception))


class TestMakeLayerPipelineHookCpLayerShard(unittest.TestCase):
    """Phase 3 CP layer-shard: `make_layer_pipeline_hook` partitions main-layer groups across CP ranks when
    `use_layer_cp_shard_for_transfer()` is True. Each rank owns groups whose `id % attn_cp_size == attn_cp_rank`;
    non-owner fires skip BOTH `add_transfer_request` AND the per-sender chunk-counter bump (so the empty-owner case
    reaches send() with `_hook_enqueued_chunks=0` and takes the empty-finalizer path)."""

    def _fire(self, mgr, dispatch, layer_ids):
        import torch as _torch

        class _FakeEv:
            def record(self):
                pass

        original = _torch.cuda.Event
        _torch.cuda.Event = _FakeEv
        try:
            hook = mgr.make_layer_pipeline_hook(dispatch)
            extend_fb = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True)
            )
            for lid in layer_ids:
                hook(lid, extend_fb)
        finally:
            _torch.cuda.Event = original

    def test_cp_partition_parametric_disjoint_and_exhaustive(self):
        """Broaden the 80-layer / cp_size=8 case to cover the modulo arithmetic's edge cases: when total groups <
        cp_size some ranks own zero groups; equal divides leave per-rank counts uniform; non-divisible cases must still
        keep the partition disjoint AND exhaustive.

        For each (num_layers, group_size, cp_size) tuple, verifies:
          * union of all ranks' owned groups == {0..total_groups-1}
          * no two ranks share a group
          * per-rank count distribution matches ceil/floor as expected
        """
        cases = [
            # (num_layers, group_size, cp_size, expected total_groups)
            (20, 1, 1, 20),   # cp_size=1 => rank 0 owns everything
            (20, 4, 2, 5),    # 5 groups across 2 ranks => 3 + 2
            (12, 4, 4, 3),    # 3 groups < 4 ranks => ranks 0-2 each own 1, rank 3 owns 0
            (28, 4, 8, 7),    # 7 groups < 8 ranks => ranks 0-6 own 1, rank 7 owns 0
            (320, 4, 8, 80),  # large case: 80 groups across 8 ranks => uniform 10/rank
        ]
        for num_layers, group_size, cp_size, expected_total_groups in cases:
            with self.subTest(
                num_layers=num_layers,
                group_size=group_size,
                cp_size=cp_size,
            ):
                # Sanity: helper math matches expectation
                self.assertEqual(
                    (num_layers + group_size - 1) // group_size,
                    expected_total_groups,
                )
                all_owned: set = set()
                per_rank_counts = []
                for rank in range(cp_size):
                    mgr, dispatch, captured, _sender = (
                        _make_hook_mgr_with_offset(
                            num_layers=num_layers,
                            group_size=group_size,
                            prefill_start_layer=0,
                            use_layer_cp_shard=True,
                            attn_cp_size=cp_size,
                            attn_cp_rank=rank,
                        )
                    )
                    self._fire(mgr, dispatch, range(num_layers))
                    owned = {c["layer_group_id"] for c in captured}
                    self.assertEqual(
                        owned & all_owned, set(),
                        f"cp_size={cp_size} rank={rank} overlaps prior",
                    )
                    all_owned |= owned
                    per_rank_counts.append(len(owned))
                # Exhaustive: union == {0..total_groups-1}
                self.assertEqual(
                    all_owned, set(range(expected_total_groups)),
                    f"union of CP-owned groups must cover all "
                    f"{expected_total_groups} groups for cp_size={cp_size}, "
                    f"num_layers={num_layers}, group_size={group_size}",
                )
                # Per-rank counts: every rank gets either ceil(total/cp) or floor(total/cp) groups (standard modulo distribution).
                ceil_per = (expected_total_groups + cp_size - 1) // cp_size
                floor_per = expected_total_groups // cp_size
                for count in per_rank_counts:
                    self.assertIn(
                        count, (ceil_per, floor_per),
                        f"per-rank count {count} not in {{ceil={ceil_per},"
                        f"floor={floor_per}}} for cp_size={cp_size}, "
                        f"total_groups={expected_total_groups}",
                    )


class TestSkipAuxRdmaWorkerBehavior(unittest.TestCase):
    """Phase 3 CP layer-shard (plan §9.1): worker honors the `skip_aux_rdma` flag on the aux-only finalizer.

    Production wiring: `MooncakeKVSender.send()` stamps the flag for non-CP0 ranks under layer-shard mode (covered by
    `TestSenderSendSkipAuxRdmaForLayerShard`). The worker reads it at the aux dispatch site:
        if kv_chunk.skip_aux_rdma:
            aux_ret = 0                  # status-only finalizer
        else:
            aux_ret = self.send_aux(...) # actual RDMA write
        self._record_aux_sent(...)       # ALWAYS — per-rank watermark close cannot drop the signal

    Drives `transfer_worker` end-to-end via `_WatermarkProgressMgr` so the real `_record_aux_sent` /
    `_maybe_sync_success_locked` helpers run, not mocks — otherwise a regression in the "still close the watermark"
    half could pass silently."""

    @staticmethod
    def _build_aux_only_chunk(room, *, skip_aux_rdma):
        # Mirrors what `MooncakeKVSender.send()`'s hook-expected branch enqueues for the trailing finalizer:
        # empty kv_indices, layer_range=None, is_last_chunk=True. `total_chunks_in_request=1` is the empty-owner
        # shape (zero hook chunks + 1 aux) — also works for non-zero hook chunks since the worker only checks
        # whether `chunks_done >= total_chunks_expected` per dst.
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=42,
            state_indices=None,
            layer_group_id=0,
            layer_range=None,
            total_layer_groups=1,
            kv_dtype="auto",
            total_chunks_in_request=1,
            transfer_event=None,
            skip_aux_rdma=skip_aux_rdma,
        )

    def test_skip_aux_rdma_true_bypasses_send_aux_keeps_watermark(self):
        # Empty-owner CP rank under layer-shard: status-only finalizer MUST NOT call send_aux, but watermark MUST still close.
        room = 700
        mgr = _WatermarkProgressMgr(room=room, num_dst_ranks=1)
        chunk = self._build_aux_only_chunk(room, skip_aux_rdma=True)
        TestPerGroupWatermarkSync._run_worker(mgr, [chunk])

        self.assertEqual(
            mgr.send_aux_calls, 0,
            "skip_aux_rdma=True ⇒ worker must bypass send_aux — non-CP0 "
            "ranks would otherwise duplicate the metadata RDMA",
        )
        # Watermark closed: _record_aux_sent fired with aux_ret=0 (any_aux_failed=False), chunks_done=1 == total_expected=1.
        self.assertEqual(
            len(mgr.sync_status_calls), 1,
            "_record_aux_sent must still fire so the per-rank watermark "
            "closes — otherwise decode hangs waiting for this rank's Success",
        )
        self.assertEqual(
            mgr.sync_status_calls[0][0][3], KVPoll.Success,
            "aux_ret=0 short-circuit must drive final status to Success "
            "(any_aux_failed=False)",
        )
        self.assertEqual(mgr.request_status[room], KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
