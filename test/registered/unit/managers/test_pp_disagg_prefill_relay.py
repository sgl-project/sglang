"""PP PD prefill: when a stage forwards its messages (SGLANG_OPT_PP_EARLY_RELAY).

event_loop_pp_disagg_prefill relays host messages along the stages: the
iteration's requests, the transferred-rid consensus and the release sets the
last stage starts. The p2p pairs are FIFO and untagged, so a stage reads them in
the order the previous stage wrote them.

The emulation runs the real loop, one thread per stage, over FIFO channels for
the host messages, the proxy tensors and the output relay; batches, forwards and
transfers are stand-ins. It checks that every stage sees the same requests,
consensus and release sets at the same iteration with and without the early
relay, and that only the early relay lets a stage's neighbours run ahead of its
GPU wait and its launch.
"""

import os
import queue
import random
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

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Without the relay the awaited event cannot fire while the hook blocks, so the
# negative arm needs only a short window; the positive arm returns on the event.
WAIT_S = 1.0
DEADLOCK_S = 20.0


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
                if time.monotonic() - t0 > DEADLOCK_S:
                    raise _Deadlock(f"{kind} {src}->{dst}")


class _Stage:
    """The scheduler fields and calls the PD prefill loop touches, for one stage."""

    def __init__(self, net, rank, pp, depth, iters, jitter, hooks):
        self.net, self.rank, self.pp, self.depth = net, rank, pp, depth
        self.iters, self.hooks = iters, hooks
        self.rng = random.Random(1000 + rank) if jitter else None
        self.it = -1
        self.rec = {}
        self.reached = {}  # (point, iteration) -> Event
        s = SimpleNamespace()
        self.s = s
        s.pp_group = SimpleNamespace(
            is_first_rank=rank == 0, is_last_rank=rank == pp - 1
        )
        s.pp_comm_stream_ctx = nullcontext()
        s.pp_send_done_event = None
        s._pp_capacity_digest = None
        s._pp_checked_allocator = None
        s._pp_release_reqs_awaiting_verdict = ()
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
            "_pp_answer_release_requests",
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
            "_pp_process_batch_result",
            "process_disagg_prefill_transfer_queue",
            "get_new_batch_prefill",
            "process_prefill_chunk",
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

    # --- bookkeeping
    def mark(self, point):
        self.reached.setdefault((point, self.it), threading.Event()).set()

    def event(self, point, it):
        return self.reached.setdefault((point, it), threading.Event())

    def jitter(self):
        if self.rng is not None:
            time.sleep(self.rng.random() * 0.002)

    def hook(self, point):
        fn = self.hooks.get((self.rank, point))
        if fn is not None:
            fn(self.it)

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
        self.jitter()

    def _pp_ingest_requests(self):
        self.hook("ingest")
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

    def process_prefill_chunk(self, last_batch=None, running_batch=None):
        self.mark("started")

    def get_new_batch_prefill(self, running_batch):
        self.jitter()
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
        d2h = SimpleNamespace(synchronize=lambda: self.hook("d2h"))
        return out.get("v"), None, d2h

    def _pp_launch_batch(self, mb_id, batch, proxy, mb_metadata, comm_queue):
        self.hook("launch")
        self.jitter()
        return SimpleNamespace(batch=batch), None

    def _pp_process_batch_result(self, batch, result):
        pass

    def process_disagg_prefill_transfer_queue(self, release):
        assert isinstance(release, dict), release
        self.record("release", dict(release))
        self.mark("released")


def _run(pp, depth, early, iters=40, jitter=False, hooks_fn=None):
    net = _Net(pp)
    hooks = {}
    stages = [_Stage(net, r, pp, depth, iters, jitter, hooks) for r in range(pp)]
    if hooks_fn is not None:
        hooks.update(hooks_fn(stages))
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
        patch.object(pp_module, "get_parallel", return_value=parallel),
        patch.object(pp_module, "start_pp_prefill_early_finish", return_value=None),
    ):
        threads = [threading.Thread(target=body, args=(st,)) for st in stages]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
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
        for pp in (2, 3, 4):
            for depth in (0, 1, 2):
                with self.subTest(pp=pp, depth=depth):
                    loop = pp + depth
                    iters = 6 * loop + 12
                    upto = iters - 2 * (pp + loop) - 2
                    base, err0 = _run(pp, depth, False, iters, jitter=True)
                    early, err1 = _run(pp, depth, True, iters, jitter=True)
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

    def _next_stage_starts_during_d2h(self, early, depth):
        pp, loop = 4, 4 + depth
        probe = 3 * loop + 1  # an iteration that waits on a result
        seen = {}

        def hooks(stages):
            def d2h(it):
                if it == probe:
                    started = stages[1].event("started", it)
                    seen["ok"] = started.wait(DEADLOCK_S if early else WAIT_S)

            return {(0, "d2h"): d2h}

        stages, errors = _run(pp, depth, early, iters=probe + 3 * loop, hooks_fn=hooks)
        self.assertEqual(errors, [])
        return seen["ok"]

    def test_next_stage_starts_while_this_stage_waits_on_its_gpu(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                self.assertTrue(self._next_stage_starts_during_d2h(True, depth))
                self.assertFalse(self._next_stage_starts_during_d2h(False, depth))

    def _first_stage_gets_release_during_last_launch(self, early, depth):
        pp, loop = 4, 4 + depth
        probe = 3 * loop + 2
        seen = {}

        def hooks(stages):
            def launch(it):
                # Stage 0 takes the set the last stage starts at iteration i in
                # its own iteration i + pp - 1.
                if it == probe:
                    ev = stages[0].event("released", probe + pp - 1)
                    seen["ok"] = ev.wait(DEADLOCK_S if early else WAIT_S)

            return {(pp - 1, "launch"): launch}

        stages, errors = _run(pp, depth, early, iters=probe + 4 * loop, hooks_fn=hooks)
        self.assertEqual(errors, [])
        return seen["ok"]

    def test_first_stage_never_waits_for_the_last_stage_launch(self):
        # The last stage's launch may block on its own forward; stage 0's
        # release receive must not wait for it.
        for depth in (1, 2):
            with self.subTest(depth=depth):
                self.assertTrue(
                    self._first_stage_gets_release_during_last_launch(True, depth)
                )
                self.assertFalse(
                    self._first_stage_gets_release_during_last_launch(False, depth)
                )


class TestLoopEntry(CustomTestCase):
    def test_the_loop_turns_it_on_only_when_the_child_resolves_on(self):
        def stop():
            raise _Stop()

        master = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
        child = "SGLANG_OPT_PP_EARLY_RELAY"
        others = ("SGLANG_OPT_PP_PREFILL_EARLY_FINISH",)
        for env, on in (
            ({}, False),
            ({master: "1"}, True),
            ({master: "1", child: "0"}, False),
            ({master: "0", child: "1"}, True),
        ):
            with self.subTest(env=env):
                sched = SimpleNamespace(
                    init_pp_loop_state=lambda: None,
                    pp_loop_size=1,
                    increment_scheduler_iterations=stop,
                )
                environ = {
                    **{k: v for k, v in os.environ.items() if k not in (master, child)},
                    **env,
                    **dict.fromkeys(others, "0"),
                }
                with (
                    patch.dict(os.environ, environ, clear=True),
                    self.assertLogs(pp_module.logger, "INFO") as logs,
                    self.assertRaises(_Stop),
                ):
                    pp_module.logger.info("probe")
                    SchedulerPPMixin.event_loop_pp_disagg_prefill(sched)
                relay_on = [
                    line
                    for line in logs.output
                    if "SGLANG_OPT_PP_EARLY_RELAY engaged" in line
                ]
                self.assertEqual(len(relay_on), int(on), logs.output)


if __name__ == "__main__":
    unittest.main()
