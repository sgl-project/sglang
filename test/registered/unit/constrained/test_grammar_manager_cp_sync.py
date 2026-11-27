"""Real Gloo regression for grammar admission across attention DP x CP x TP."""

import datetime
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _manager(
    tp_group,
    cp_group,
    cp_size,
    dp_enabled=True,
    pp_rank=0,
    pp_size=1,
    pp_group=None,
):
    parallel = SimpleNamespace(
        attn_dp_enabled=dp_enabled,
        attn_cp_size=cp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        pp_async_batch_depth=0,
    )
    scheduler = SimpleNamespace(
        server_args=None,
        dp_tp_group=tp_group,
        dp_tp_cpu_group=tp_group.cpu_group,
        attn_cp_group=cp_group,
        pp_group=pp_group,
    )
    with (
        patch(
            "sglang.srt.constrained.grammar_manager.get_parallel", return_value=parallel
        ),
        patch(
            "sglang.srt.constrained.grammar_manager.get_serving",
            return_value=SimpleNamespace(skip_tokenizer_init=True),
        ),
    ):
        manager = GrammarManager(scheduler)
    manager.SGLANG_GRAMMAR_POLL_INTERVAL = 0.001
    manager.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 100
    manager.grammar_backend = _Backend()
    return manager


class _Backend:
    def __init__(self):
        self.cache = {}

    def set_cache(self, key, value):
        self.cache[key] = value


class _Request:
    def __init__(self, rid):
        self.rid = rid
        self.grammar = Future()
        self.grammar_key = ("json", rid)
        self.grammar_wait_ct = 0
        self.abort_reason = None

    def finished(self):
        return self.abort_reason is not None

    def set_finish_with_abort(self, reason):
        self.abort_reason = reason


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        # Both configurations use 8 ranks, with 4 replicas per DPA shard.
        for cp_size, tp_size in ((4, 1), (2, 2)):
            own_tp = own_cp = own_dp = None
            for dp in range(2):
                ranks = list(range(dp * 4, (dp + 1) * 4))
                group = dist.new_group(ranks, backend="gloo")
                if rank in ranks:
                    own_dp = group
                for cp in range(cp_size):
                    ranks = list(
                        range(dp * 4 + cp * tp_size, dp * 4 + (cp + 1) * tp_size)
                    )
                    group = dist.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        own_tp = SimpleNamespace(
                            cpu_group=group,
                            world_size=tp_size,
                            first_rank=ranks[0],
                            is_first_rank=rank == ranks[0],
                        )
                for tp in range(tp_size):
                    ranks = [dp * 4 + cp * tp_size + tp for cp in range(cp_size)]
                    group = dist.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        own_cp = SimpleNamespace(cpu_group=group, world_size=cp_size)

            manager = _manager(own_tp, own_cp, cp_size)
            requests = [_Request(f"dp{rank // 4}-req{i:02}") for i in range(40)]
            common = {1, 9, 17, 33}
            # Unequal completed Futures, including out-of-order completions.
            local_ready = common | set(range(rank % 4 + 2))
            for i in local_ready:
                requests[i].grammar.set_result(BaseGrammarObject())
            requests[25].grammar.set_result(InvalidGrammarObject("bad schema"))
            # One replica raises while peers succeed; all replicas must abort
            # in lockstep without caching a split outcome.
            if rank % 4 == 3:
                requests[27].grammar.set_exception(
                    RuntimeError("rank-local compiler failure")
                )
            else:
                requests[27].grammar.set_result(BaseGrammarObject())
            # One replica raises while peers are still compiling; all replicas
            # must cancel and abort in lockstep without caching.
            if rank % 4 == 2:
                requests[28].grammar.set_exception(
                    RuntimeError("early rank-local failure")
                )
            requests[29].grammar.set_exception(ValueError("compiler error"))
            requests[35].abort_reason = "cancelled"
            requests[35].grammar = None
            # Only one replica times out. All replicas must reject request32.
            if rank % 4 == 3:
                requests[32].grammar_wait_ct = 99
            manager.grammar_queue = requests[:]
            first = manager.get_ready_grammar_requests()
            expected = sorted(common | {0, 25, 35}) + [27, 28, 29, 32]
            # Every local_ready includes 0 and 1. Other extras differ by rank.
            assert [r.rid for r in first] == [requests[i].rid for i in expected]
            assert all(requests[i].finished() for i in (25, 27, 28, 29, 32, 35))
            assert all(
                isinstance(requests[i].grammar, InvalidGrammarObject)
                for i in (25, 27, 28, 29)
            )
            assert "rank-local compiler failure" in requests[27].abort_reason
            assert "early rank-local failure" in requests[28].abort_reason
            assert "timed out" in requests[32].abort_reason
            # Deterministic InvalidGrammarObject across all peers is cached;
            # rank-local exceptions and timeouts are never cached on any peer.
            assert requests[25].grammar_key in manager.grammar_backend.cache
            for idx in (27, 28, 29, 32):
                assert requests[idx].grammar_key not in manager.grammar_backend.cache

            # The actual next bootstrap operation requires matching ordered IDs
            # and same-length poll arrays. Check both without mocking Gloo.
            ids = [None] * 4
            dist.all_gather_object(ids, [r.rid for r in first], group=own_dp)
            assert ids == [ids[0]] * 4
            polls = torch.ones(len(first), dtype=torch.uint8)
            dist.all_reduce(polls, op=dist.ReduceOp.MIN, group=own_tp.cpu_group)
            dist.all_reduce(polls, op=dist.ReduceOp.MIN, group=own_cp.cpu_group)
            assert polls.tolist() == [1] * len(first)

            # GrammarManager returns ready/failed reqs for the caller to move out.
            returned_ids = {r.rid for r in first}
            remaining = [r for r in manager.grammar_queue if r.rid not in returned_ids]
            manager.grammar_queue = remaining[:]
            for request in remaining:
                if not request.grammar.done():
                    request.grammar.set_result(BaseGrammarObject())
            second = manager.get_ready_grammar_requests()
            assert [r.rid for r in second] == [r.rid for r in remaining]
            returned_ids.update(r.rid for r in second)
            assert len(returned_ids) == 40
            assert len(first) + len(second) == 40
            # DPA shards intentionally use different IDs; synchronizing them
            # together would be incorrect even though their queue lengths match.
            assert all(r.rid.startswith(f"dp{rank // 4}-") for r in first + second)
            assert not manager._resolved_futures

            # Later PP stages receive PP0's admitted indices and still resolve
            # local futures; verify TP/CP outcome agreement while preserving
            # PP0's exact admission order [0, 1, 2] (even when index 1 fails).
            pp_group = SimpleNamespace(recv_object=lambda src, tag: ({0, 1, 2}, set()))
            pp1_mgr = _manager(
                own_tp, own_cp, cp_size, pp_rank=1, pp_size=2, pp_group=pp_group
            )
            pp1_reqs = [_Request(f"dp{rank // 4}-pp1-{i}") for i in range(3)]
            pp1_reqs[0].grammar.set_result(BaseGrammarObject())
            if rank % 4 == 3:
                pp1_reqs[1].grammar.set_exception(
                    RuntimeError("pp1 rank-local failure")
                )
            else:
                pp1_reqs[1].grammar.set_result(BaseGrammarObject())
            pp1_reqs[2].grammar.set_result(InvalidGrammarObject("pp1 bad schema"))
            pp1_mgr.grammar_queue = pp1_reqs[:]
            pp1_out = pp1_mgr.get_ready_grammar_requests()
            assert [r.rid for r in pp1_out] == [
                pp1_reqs[0].rid,
                pp1_reqs[1].rid,
                pp1_reqs[2].rid,
            ]
            assert not pp1_reqs[0].finished()
            assert pp1_reqs[0].grammar_key in pp1_mgr.grammar_backend.cache
            assert pp1_reqs[1].finished()
            assert isinstance(pp1_reqs[1].grammar, InvalidGrammarObject)
            assert "pp1 rank-local failure" in pp1_reqs[1].abort_reason
            # PP0 already admitted and cached index 1, so every PP1+ peer
            # caches the agreed InvalidGrammarObject to keep retry admission
            # paths aligned across stages (avoiding a PP1 recv_object hang).
            assert isinstance(
                pp1_mgr.grammar_backend.cache.get(pp1_reqs[1].grammar_key),
                InvalidGrammarObject,
            )
            assert pp1_reqs[2].finished()
            assert pp1_reqs[2].grammar_key in pp1_mgr.grammar_backend.cache
            assert not pp1_mgr._resolved_futures
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestGrammarCpSync(unittest.TestCase):
    def test_only_add_cp_sync_when_dpa_excludes_cp(self):
        tp = SimpleNamespace(
            cpu_group=object(), world_size=1, first_rank=0, is_first_rank=True
        )
        cp = SimpleNamespace(cpu_group=object(), world_size=4)
        self.assertIs(_manager(tp, cp, 4).grammar_sync_cp_group, cp)
        self.assertIsNone(_manager(tp, cp, 1).grammar_sync_cp_group)
        self.assertIsNone(_manager(tp, cp, 4, dp_enabled=False).grammar_sync_cp_group)

    def test_ready_future_resolved_once_across_poll_rounds(self):
        """A ready peer must not repeatedly resolve the same Future while waiting."""
        tp = SimpleNamespace(
            cpu_group=object(), world_size=2, first_rank=0, is_first_rank=True
        )
        mgr = _manager(tp, None, 1, dp_enabled=False)

        class _CountingFuture(Future):
            def __init__(self):
                super().__init__()
                self.result_calls = 0

            def result(self, timeout=None):
                self.result_calls += 1
                return super().result(timeout)

        req = _Request("r0")
        req.grammar = _CountingFuture()
        req.grammar.set_result(BaseGrammarObject())
        mgr.grammar_queue = [req]

        peer_ready = False

        def fake_all_gather(output_list, local_obj, group=None):
            output_list[0] = local_obj
            output_list[1] = ({0} if peer_ready else set(), set(), {}, set())

        with patch(
            "sglang.srt.constrained.grammar_manager.torch.distributed.all_gather_object",
            side_effect=fake_all_gather,
        ):
            self.assertEqual(mgr.get_ready_grammar_requests(), [])
            self.assertEqual(mgr.get_ready_grammar_requests(), [])
            self.assertEqual(req.grammar.result_calls, 1)
            peer_ready = True
            self.assertEqual(mgr.get_ready_grammar_requests(), [req])
        self.assertNotIn(req.grammar, mgr._resolved_futures)
        self.assertEqual(len(mgr._resolved_futures), 0)

    def test_staggered_cacheable_invalid_grammar_waits_for_peer_and_caches(self):
        """A peer that returns InvalidGrammarObject early waits for peers and caches."""
        tp = SimpleNamespace(
            cpu_group=object(), world_size=2, first_rank=0, is_first_rank=True
        )
        mgr = _manager(tp, None, 1, dp_enabled=False)

        req = _Request("r_invalid")
        req.grammar.set_result(InvalidGrammarObject("deterministic invalid schema"))
        mgr.grammar_queue = [req]

        peer_done = False

        def fake_all_gather(output_list, local_obj, group=None):
            output_list[0] = local_obj
            if peer_done:
                output_list[1] = ({0}, set(), {0: "deterministic invalid schema"}, {0})
            else:
                output_list[1] = (set(), set(), {}, set())

        with patch(
            "sglang.srt.constrained.grammar_manager.torch.distributed.all_gather_object",
            side_effect=fake_all_gather,
        ):
            self.assertEqual(mgr.get_ready_grammar_requests(), [])
            self.assertFalse(req.finished())
            self.assertNotIn(req.grammar_key, mgr.grammar_backend.cache)

            peer_done = True
            self.assertEqual(mgr.get_ready_grammar_requests(), [req])
        self.assertTrue(req.finished())
        self.assertIsInstance(req.grammar, InvalidGrammarObject)
        self.assertIn(req.grammar_key, mgr.grammar_backend.cache)
        self.assertEqual(len(mgr._resolved_futures), 0)

    def test_single_rank_compile_exception_does_not_cache(self):
        """With TP=1 and CP=1, a compiler exception must not poison the cache."""
        tp = SimpleNamespace(
            cpu_group=object(), world_size=1, first_rank=0, is_first_rank=True
        )
        mgr = _manager(tp, None, 1, dp_enabled=False)

        req = _Request("r_exc")
        req.grammar.set_exception(RuntimeError("transient compiler error"))
        mgr.grammar_queue = [req]

        self.assertEqual(mgr.get_ready_grammar_requests(), [req])
        self.assertTrue(req.finished())
        self.assertIsInstance(req.grammar, InvalidGrammarObject)
        self.assertIn("transient compiler error", req.abort_reason)
        self.assertNotIn(req.grammar_key, mgr.grammar_backend.cache)

    def test_pp0_single_rank_exception_propagates_as_failed_so_both_stages_retry(self):
        """With PP>1 and TP=1/CP=1, a PP0 exception goes via failed so neither stage caches."""
        tp = SimpleNamespace(
            cpu_group=object(), world_size=1, first_rank=0, is_first_rank=True
        )
        sent = []
        pp0_group = SimpleNamespace(
            send_object=lambda data, dst, async_send, tag: (
                sent.append(data) or [SimpleNamespace(wait=lambda: None)]
            )
        )
        pp0 = _manager(
            tp, None, 1, dp_enabled=False, pp_rank=0, pp_size=2, pp_group=pp0_group
        )
        req0 = _Request("r_pp0_exc")
        req0.grammar.set_exception(RuntimeError("pp0 compiler boom"))
        pp0.grammar_queue = [req0]

        self.assertEqual(pp0.get_ready_grammar_requests(), [req0])
        self.assertTrue(req0.finished())
        self.assertNotIn(req0.grammar_key, pp0.grammar_backend.cache)
        self.assertEqual(sent, [(set(), {0})])

        pp1_group = SimpleNamespace(recv_object=lambda src, tag: sent[0])
        pp1 = _manager(
            tp, None, 1, dp_enabled=False, pp_rank=1, pp_size=2, pp_group=pp1_group
        )
        req1 = _Request("r_pp0_exc")
        req1.grammar.set_result(BaseGrammarObject())
        pp1.grammar_queue = [req1]

        self.assertEqual(pp1.get_ready_grammar_requests(), [req1])
        self.assertTrue(req1.finished())
        self.assertNotIn(req1.grammar_key, pp1.grammar_backend.cache)

    def test_pp1_single_rank_exception_caches_to_match_pp0_admission_path(self):
        """If PP0 already cached a schema and PP1 raises, PP1 also caches so retries align."""
        tp = SimpleNamespace(
            cpu_group=object(), world_size=1, first_rank=0, is_first_rank=True
        )
        pp1_group = SimpleNamespace(recv_object=lambda src, tag: ({0}, set()))
        pp1 = _manager(
            tp, None, 1, dp_enabled=False, pp_rank=1, pp_size=2, pp_group=pp1_group
        )
        req = _Request("r_pp1_exc")
        req.grammar.set_exception(RuntimeError("pp1 compiler boom"))
        pp1.grammar_queue = [req]

        self.assertEqual(pp1.get_ready_grammar_requests(), [req])
        self.assertTrue(req.finished())
        self.assertIsInstance(
            pp1.grammar_backend.cache.get(req.grammar_key), InvalidGrammarObject
        )

    def test_real_gloo_admission_timeout_and_dpa_isolation(self):
        with tempfile.TemporaryDirectory(prefix="grammar-cp-sync-") as folder:
            mp.spawn(_worker, args=(str(Path(folder) / "rdzv"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
