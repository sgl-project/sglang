"""PP PD prefill: when a stage forwards its messages (SGLANG_OPT_PP_EARLY_RELAY).

event_loop_pp_disagg_prefill relays host messages along the stages: the
iteration's requests, the transferred-rid consensus and the release sets the
last stage starts. The p2p pairs are FIFO and untagged, so a stage reads them in
the order the previous stage wrote them.

The emulation runs the real loop, one thread per stage, over FIFO channels for
the host messages, the proxy tensors and the output relay; batches, forwards and
transfers are stand-ins. It checks that every stage sees the same requests,
consensus and release sets at the same iteration with and without the early
relay.
"""

import queue
import threading
import time
import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler_pp_mixin as pp_module
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HANG_S = 20.0  # a hang guard only: a stage that waits this long is stuck


class _Stop(Exception):
    pass


class _Deadlock(Exception):
    pass


class _Net:
    """FIFO channels from stage r to stage (r + 1) % pp, one per kind."""

    def __init__(self, pp):
        self.pp = pp
        self.q = {
            (kind, r): queue.Queue()
            for kind in ("obj", "proxy", "out")
            for r in range(pp)
        }
        self.stopping = threading.Event()

    def send(self, kind, src, data):
        self.q[(kind, src)].put(data)
        return []

    def recv(self, kind, dst):
        src = (dst - 1) % self.pp
        t0 = time.monotonic()
        while True:
            try:
                return self.q[(kind, src)].get(timeout=0.01)
            except queue.Empty:
                if self.stopping.is_set():
                    raise _Stop()
                if time.monotonic() - t0 > _HANG_S:
                    raise _Deadlock(f"{kind} {src}->{dst}")


class _Stage:
    """The scheduler fields and calls the PD prefill loop touches, for one stage."""

    def __init__(self, net, rank, pp, depth, iters):
        self.net, self.rank, self.pp, self.depth = net, rank, pp, depth
        self.iters = iters
        self.it = -1
        self.rec = {}
        s = SimpleNamespace()
        self.s = s
        s.pp_group = SimpleNamespace(
            is_first_rank=rank == 0, is_last_rank=rank == pp - 1
        )
        s.pp_comm_stream_ctx = nullcontext()
        s.pp_send_done_event = None
        s._pp_capacity_digest = None
        s._pp_checked_allocator = None
        s._pp_reqs_awaiting_verdict = ()
        s.disagg_prefill_transfer_queue = []
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=None)
        s.output_streamer = None
        s.enable_staging = False
        s.pause_state = SimpleNamespace(may_launch=lambda: True)
        s.invariant_checker = SimpleNamespace(run=lambda phase, batch: None)
        s.dp_attn_adapter = SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda b: b)
        for name in (
            "event_loop_pp_disagg_prefill",
            "_pp_pd_get_prefill_transferred_ids",
            "_pp_pd_send_consensus_release_ids",
            "_pp_commit_comm_work",
            "_pp_mark_capacity_digest",
            "_pp_check_capacity_digest",
            "_pp_send_capacity_digest",
            "_pp_answer_verdict_requests",
        ):
            setattr(s, name, MethodType(getattr(SchedulerPPMixin, name), s))
        for name in (
            "init_pp_loop_state",
            "increment_scheduler_iterations",
            "_pp_ingest_requests",
            "get_rids",
            "_pp_send_reqs_to_next_stage",
            "_pp_send_pyobj_to_next_stage",
            "_pp_recv_pyobj_from_prev_stage",
            "_pp_recv_proxy_tensors",
            "_pp_send_proxy_to_next_stage",
            "_pp_commit_send_output_work_and_preprocess_output_tensors",
            "_pp_launch_batch",
            "process_disagg_prefill_transfer_queue",
            "get_new_batch_prefill",
        ):
            setattr(s, name, getattr(self, name))
        for name in (
            "process_bootstrapped_queue",
            "process_pending_chunked_abort",
            "_process_hicache_events",
            "_pp_commit_proxy_send_work",
            "_pp_apply_war_barrier",
            "on_idle",
        ):
            setattr(s, name, lambda: None)
        s.process_prefill_chunk = lambda last_batch=None, running_batch=None: None
        s._pp_process_batch_result = lambda batch, result: None

    def record(self, key, value):
        self.rec.setdefault(self.it, {})[key] = value

    # --- loop calls
    def init_pp_loop_state(self):
        s = self.s
        s.pp_loop_size = self.pp + self.depth
        s.mbs = [None] * s.pp_loop_size
        s.last_mbs = [None] * s.pp_loop_size
        s.running_mbs = [
            SimpleNamespace(batch_is_full=False) for _ in range(s.pp_loop_size)
        ]
        s.mb_metadata = [None] * s.pp_loop_size
        s.pp_outputs = None
        s.last_rank_comm_queue = []
        s.send_req_work = []
        s.launch_event = None

    def increment_scheduler_iterations(self):
        self.it += 1
        if self.it >= self.iters:
            raise _Stop()

    def _pp_ingest_requests(self):
        if self.rank == 0:
            reqs = [f"req{self.it}"] if self.it % 3 else []
        else:
            reqs = self.net.recv("obj", self.rank)
            assert isinstance(reqs, list), reqs
            assert all(str(r).startswith("req") for r in reqs), reqs
        self.record("reqs", list(reqs))
        return reqs

    def get_rids(self, queue_, poll_check, good, bad):
        # Stage r concludes request n r iterations after stage 0, and reports
        # request (it - r) failed when that is a multiple of 7.
        top = self.it - self.rank
        succeeded = [f"x{n}" for n in range(max(0, top - 8), max(0, top))]
        failed = []
        if top >= 0 and top % 7 == 0:
            failed = [f"x{top}"]
            succeeded = [r for r in succeeded if r != failed[0]]
        return succeeded, failed

    def _pp_send_reqs_to_next_stage(self, recv_reqs):
        return self.net.send("obj", self.rank, list(recv_reqs))

    def _pp_send_pyobj_to_next_stage(self, data, async_send=False):
        return self.net.send("obj", self.rank, data)

    def _pp_recv_pyobj_from_prev_stage(self):
        data = self.net.recv("obj", self.rank)
        assert isinstance(data, dict), data
        return data

    def get_new_batch_prefill(self, running_batch):
        batch = SimpleNamespace(name=f"b{self.it}") if self.it % 5 != 4 else None
        return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    def _pp_recv_proxy_tensors(self):
        if self.rank == 0:
            return None
        return self.net.recv("proxy", self.rank)

    def _pp_send_proxy_to_next_stage(self, result):
        self.net.send("proxy", self.rank, result.batch.name)

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self, next_first_rank_mb_id, next_mb_id
    ):
        # The output relay: the last stage starts it, the others pass on what
        # they got one iteration before; even stages send first.
        s, out = self.s, {}

        def send():
            if s.pp_group.is_last_rank:
                target = s.mbs[next_first_rank_mb_id]
                if target is not None:
                    self.net.send("out", self.rank, target.name)
            elif s.pp_outputs:
                self.net.send("out", self.rank, s.pp_outputs)

        def recv():
            if s.mbs[next_mb_id] is not None:
                out["v"] = self.net.recv("out", self.rank)

        if self.rank % 2 == 0:
            send()
            recv()
        else:
            recv()
            send()
        return out.get("v"), None, SimpleNamespace(synchronize=lambda: None)

    def _pp_launch_batch(self, mb_id, batch, proxy, mb_metadata, comm_queue):
        return SimpleNamespace(batch=batch), None

    def process_disagg_prefill_transfer_queue(self, release):
        assert isinstance(release, dict), release
        self.record("release", dict(release))


def _run(pp, depth, early, iters):
    net = _Net(pp)
    stages = [_Stage(net, r, pp, depth, iters) for r in range(pp)]
    errors = []

    def body(stage):
        orig = stage.s._pp_pd_get_prefill_transferred_ids

        def transferred():
            t = orig()
            stage.record("trans", dict(t))
            return t

        stage.s._pp_pd_get_prefill_transferred_ids = transferred
        try:
            stage.s.event_loop_pp_disagg_prefill()
        except _Stop:
            pass
        except BaseException as e:  # noqa: BLE001
            errors.append((stage.rank, repr(e)))
        finally:
            net.stopping.set()

    parallel = SimpleNamespace(pp_size=pp, pp_async_batch_depth=depth)
    with (
        envs.SGLANG_OPT_PP_EARLY_RELAY.override(early),
        envs.SGLANG_OPT_PP_PREFILL_EARLY_FINISH.override(False),
        patch.object(pp_module, "get_parallel", return_value=parallel),
    ):
        threads = [threading.Thread(target=body, args=(st,)) for st in stages]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2 * _HANG_S)
    return stages, errors


class TestPPDisaggPrefillEarlyRelay(CustomTestCase):
    def _complete(self, stages, upto):
        for st in stages:
            for i in range(upto):
                rec = st.rec.get(i, {})
                self.assertIn("trans", rec, (st.rank, i))
                self.assertIn("reqs", rec, (st.rank, i))
        return {st.rank: {i: st.rec[i] for i in range(upto)} for st in stages}

    def test_same_messages_at_the_same_iteration(self):
        # A middle stage both takes and relays every message.
        pp, depth = 3, 1
        loop = pp + depth
        iters = 6 * loop + 12
        upto = iters - 2 * (pp + loop) - 2
        base, err0 = _run(pp, depth, False, iters)
        early, err1 = _run(pp, depth, True, iters)
        self.assertEqual(err0, [])
        self.assertEqual(err1, [])
        a = self._complete(base, upto)
        self.assertEqual(a, self._complete(early, upto))
        # Release sets land at the same iteration on every stage.
        for i in range(upto):
            rel = [a[r][i].get("release") for r in range(pp)]
            self.assertTrue(all(x == rel[0] for x in rel), (i, rel))
        # The release ring is live and carries failures.
        outcomes = {
            failed
            for i in range(upto)
            for failed in (a[pp - 1][i].get("release") or {}).values()
        }
        self.assertEqual(outcomes, {False, True})


if __name__ == "__main__":
    unittest.main()
