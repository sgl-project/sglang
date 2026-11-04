"""--disaggregation-nosync-qsa-publish: the deferred publish sends what the
synchronous publish sends, for every state set that contains a QSA type."""

import ast
import inspect
import itertools
import random
import textwrap
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation import decode as D
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

M, QP, QC = StateType.MAMBA, StateType.QSA_PENDING, StateType.QSA_COMPRESSED


def _stock_payloads_factory():
    """pop_preallocated's synchronous payload closures, cut out with ast."""
    src = textwrap.dedent(inspect.getsource(D.DecodePreallocQueue.pop_preallocated))
    closures = {
        n.name: n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("_mamba_payload", "_full_kv_pages_payload", "_qsa_pending_payload")
    }
    assert len(closures) == 3, closures.keys()
    body = "\n".join(textwrap.indent(ast.unparse(n), "    ") for n in closures.values())
    factory = (
        "def _stock_payloads(self, decode_req, seq_len):\n"
        + body
        + "\n    return {M: _mamba_payload, QP: _qsa_pending_payload,"
        " QC: _full_kv_pages_payload}\n"
    )
    ns = dict(vars(D), M=M, QP=QP, QC=QC)
    exec(compile(factory, "<stock payloads>", "exec"), ns)
    return ns["_stock_payloads"]


class _Event:
    done = True

    def __init__(self, *args, **kwargs):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def query(self):
        return self.recorded and _Event.done


class _Receiver:
    def __init__(self, log):
        self.log = log

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None, **kwargs):
        self.log.append((kv_indices, aux_index, state_indices, kwargs))


class _Slots:
    def __init__(self):
        self.n = 0

    def alloc(self):
        self.n += 1
        return self.n


def _queue(*, rows, ctx, ps_dev, id_write, gen, nosync_qsa):
    q = object.__new__(D.DecodePreallocQueue)
    req_to_token = torch.zeros((rows, ctx), dtype=torch.int32)
    for r in range(rows):
        pages = torch.randperm(rows * ctx // ps_dev + 7, generator=gen)[
            : (ctx + ps_dev - 1) // ps_dev
        ]
        t = torch.arange(ctx)
        req_to_token[r] = (pages[t // ps_dev] * ps_dev + t % ps_dev).to(torch.int32)
    mapping = torch.randperm(4 * rows, generator=gen)[:rows].to(torch.int32) + 1
    q.req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token,
        req_index_to_mamba_index_mapping=mapping,
        state_sub_pool="mamba",
        translate_mamba_indices=lambda x: x,
    )
    space = IdSpace(key=("mamba", 0), write=id_write)
    q.token_to_kv_pool_allocator = SimpleNamespace(state_ids=lambda sub: space)
    q.token_to_kv_pool = SimpleNamespace(page_size=ps_dev)
    q.req_to_metadata_buffer_idx_allocator = _Slots()
    q.transfer_queue = SimpleNamespace(enable_staging=False, staging_handler=None)
    q._deferred_sends = deque()
    q._num_published_destinations = 0
    q._nosync_qsa_publish = nosync_qsa
    return q


def _decode_req(idx, log):
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=idx),
        finished_reason=None,
        time_stats=mock.Mock(),
        bootstrap_room=1000 + idx,
    )
    return SimpleNamespace(
        req=req,
        kv_receiver=_Receiver(log),
        is_rebootstrap=False,
        metadata_buffer_index=None,
    )


def _same_payload(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(_same_payload(x, y) for x, y in zip(a, b))
        )
    return (
        isinstance(a, np.ndarray)
        and isinstance(b, np.ndarray)
        and a.dtype == b.dtype
        and a.shape == b.shape
        and np.array_equal(a, b)
    )


def _same_send(s, t):
    (ka, aa, sa, wa), (kb, ab, sb, wb) = s, t
    return (
        _same_payload(ka, kb)
        and aa == ab
        and wa == wb
        and len(sa) == len(sb)
        and all(_same_payload(x, y) for x, y in zip(sa, sb))
        and (pack_int_lists(sa, "i") if sa else b"")
        == (pack_int_lists(sb, "i") if sb else b"")
    )


@mock.patch.object(torch.cuda, "Event", _Event)
class TestNoSyncQsaPublish(CustomTestCase):
    def test_decision_table(self):
        """Off: the stock MAMBA-only deferral. On: also every subset of
        MAMBA/QSA_PENDING/QSA_COMPRESSED that holds a QSA type, nothing else."""
        newly = 0
        for enabled in (False, True):
            for k in range(len(StateType) + 1):
                for subset in itertools.combinations(list(StateType), k):
                    t = set(subset)
                    got = D._defers_qsa_publish(list(subset), enabled=enabled)
                    want = enabled and t <= {M, QP, QC} and bool(t & {QP, QC})
                    self.assertEqual(got, want, (enabled, subset))
                    newly += got
        self.assertEqual(newly, 6)

    def test_deferred_send_equals_synchronous_send(self):
        stock_payloads = _stock_payloads_factory()
        gen = torch.Generator().manual_seed(1234)
        rng = random.Random(1234)
        qsa_sets = [
            list(p)
            for k in (1, 2, 3)
            for c in itertools.combinations((M, QP, QC), k)
            if set(c) & {QP, QC}
            for p in itertools.permutations(c)
        ]
        for i in range(600):
            ps_dev = rng.choice((64, 64, 64, 16, 1))
            ps_alloc = ps_dev if rng.random() < 0.8 else rng.choice((1, 16, 64))
            rows, ctx = 6, rng.choice((256, 1024, 4096))
            perm = torch.randperm(64, generator=gen).to(torch.int32)
            # A non-identity state id space: the payload must use state_ids().ids().
            id_write = (lambda v, perm=perm: perm[v.long()]) if i % 2 else None
            state_types = [M, QP, QC] if i % 3 else rng.choice(qsa_sets)
            idx = rng.randrange(rows)
            seq_len = rng.randrange(1, ctx + 1)
            prefix = (rng.randrange(0, seq_len) // ps_alloc) * ps_alloc
            kw = dict(rows=rows, ctx=ctx, ps_dev=ps_dev, id_write=id_write)
            qa = _queue(gen=torch.Generator().manual_seed(i), nosync_qsa=True, **kw)
            qb = _queue(gen=torch.Generator().manual_seed(i), nosync_qsa=True, **kw)
            la, lb = [], []
            da, db = _decode_req(idx, la), _decode_req(idx, lb)

            payloads = stock_payloads(qa, da, seq_len)
            qa._send_kv_metadata(
                da,
                qa.req_to_token_pool.req_to_token[idx, prefix:seq_len],
                ps_alloc,
                [payloads[st]() for st in state_types],
                decode_prefix_len=prefix,
            )
            qb._defer_kv_metadata(
                db,
                qb.req_to_token_pool.req_to_token[idx, prefix:seq_len],
                ps_alloc,
                has_state=True,
                metadata_kwargs={"decode_prefix_len": prefix},
                state_sources=qb._qsa_state_sources(
                    db, state_types=state_types, seq_len=seq_len
                ),
            )
            qb.flush_deferred_sends()
            self.assertEqual((len(la), len(lb)), (1, 1))
            self.assertTrue(_same_send(la[0], lb[0]), (i, state_types, seq_len))
            self.assertEqual(qb._num_published_destinations, 1)

    def test_without_state_sources_the_stock_deferral_is_unchanged(self):
        log = []
        q = _queue(
            rows=6,
            ctx=1024,
            ps_dev=64,
            id_write=None,
            gen=torch.Generator().manual_seed(9),
            nosync_qsa=False,
        )
        dreq = _decode_req(2, log)
        q._defer_kv_metadata(
            dreq,
            q.req_to_token_pool.req_to_token[2, :300],
            64,
            has_state=True,
            metadata_kwargs={"decode_prefix_len": 0},
        )
        q.flush_deferred_sends()
        mamba = q.req_to_token_pool.req_index_to_mamba_index_mapping[2]
        self.assertEqual(len(log), 1)
        self.assertTrue(_same_payload(log[0][2], [[mamba.numpy()]]))

    def test_sends_wait_for_the_copy_and_skip_dead_requests(self):
        log = []
        q = _queue(
            rows=6,
            ctx=1024,
            ps_dev=64,
            id_write=None,
            gen=torch.Generator().manual_seed(9),
            nosync_qsa=True,
        )
        _Event.done = False
        try:
            reqs = [_decode_req(i % 6, log) for i in range(5)]
            for dreq in reqs:
                q._defer_kv_metadata(
                    dreq,
                    q.req_to_token_pool.req_to_token[dreq.req.kv.req_pool_idx, :512],
                    64,
                    has_state=True,
                    metadata_kwargs={"decode_prefix_len": 0},
                    state_sources=q._qsa_state_sources(
                        dreq, state_types=[M, QP, QC], seq_len=512
                    ),
                )
            q.flush_deferred_sends()
            self.assertEqual((log, len(q._deferred_sends)), ([], 5))
            reqs[1].req.finished_reason = D.FINISH_ABORT("x")
            reqs[3].kv_receiver = None
            _Event.done = True
            q.flush_deferred_sends()
            self.assertEqual((len(log), len(q._deferred_sends)), (3, 0))
        finally:
            _Event.done = True


class TestNoSyncQsaPublishLog(CustomTestCase):
    def test_init_line_says_engaged_only_with_the_deferred_publish(self):
        with mock.patch.object(D.l16_loopstat, "prealloc_no_sync", lambda: True):
            with self.assertLogs(D.logger, level="INFO") as logs:
                D._log_nosync_qsa_publish()
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        with mock.patch.object(D.l16_loopstat, "prealloc_no_sync", lambda: False):
            with self.assertLogs(D.logger, level="INFO") as logs:
                D._log_nosync_qsa_publish()
        self.assertEqual([r.levelname for r in logs.records], ["WARNING"])
        self.assertIn("has no effect", logs.output[0])


if __name__ == "__main__":
    unittest.main()
