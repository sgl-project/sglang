"""Pipeline stages under unified memory: every stage asks each capacity
question of every stage's geometry, so all of them decide alike."""

import random
import unittest
from contextlib import contextmanager
from typing import List

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# One byte budget, two layer mixes: the first stage holds mostly full
# attention, the second mostly recurrent state.
_STATE_STAGES = (
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=tuple(range(6)),
        state_layer_ids=(6,),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
    ),
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=(7,),
        state_layer_ids=tuple(range(8, 16)),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
        start_layer=7,
    ),
)
# With a sliding window (a float between the ends): the first stage mostly
# full attention, the second mostly window layers.
_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=tuple(range(4)),
        swa_layer_ids=(4,),
        state_layer_ids=(5,),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
    ),
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=(6,),
        swa_layer_ids=(7, 8, 9, 10),
        state_layer_ids=(11, 12),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
        start_layer=6,
    ),
)


def _stages(geometries, checks=False):
    allocators = [
        build_unified_bundle(g).token_to_kv_pool_allocator for g in geometries
    ]
    shared = [a.stage_geometry() for a in allocators]
    for stage, allocator in enumerate(allocators):
        allocator.install_stage_capacity(stage, shared, checks=checks)
    return allocators


@contextmanager
def _alone(allocator):
    """The allocator answering for its own stage only."""
    capacity, allocator.stage_capacity = allocator.stage_capacity, None
    try:
        yield allocator
    finally:
        allocator.stage_capacity = capacity


def _fits_alone(allocator, demands) -> bool:
    with _alone(allocator):
        return allocator.units_fit(demands)


def _admits_alone(allocator, demand, evictable) -> bool:
    with _alone(allocator):
        return allocator.admits(demand, evictable=evictable)


def _on_every_stage(allocators, op):
    """Run ``op`` on every stage, as every stage runs the same operations."""
    return [op(allocator) for allocator in allocators]


def _probe_demands(allocator, rng, count=12) -> List[dict]:
    owner, windows, state = allocator.sub_pool_roles()
    probes = []
    for _ in range(count):
        pages = rng.randrange(0, 80)
        demands = {owner: pages}
        for window in windows:
            demands[window] = min(pages, rng.randrange(0, 8))
        if state is not None:
            demands[state] = rng.randrange(0, 24)
        probes.append(demands)
    return probes


class TestStageCapacity(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_every_stage_decides_alike_where_the_stages_disagree(self):
        """On empty arenas of one size, a FULL-heavy demand fits only the state-heavy
        stage and a state-heavy demand only the other; both stages refuse both."""
        a0, a1 = _stages(_STATE_STAGES)
        owner, _, state = a0.sub_pool_roles()
        only = {0: [], 1: []}
        for full in range(0, 480, 8):
            for slots in range(0, 48, 4):
                demands = {owner: full, state: slots}
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                for allocator in (a0, a1):
                    self.assertEqual(
                        allocator.units_fit(demands), fit0 and fit1, demands
                    )
                if fit0 != fit1:
                    only[0 if fit0 else 1].append(demands)
        self.assertTrue(only[0] and only[1], only)
        for demands in only[0] + only[1]:
            tokens = demands[owner] * a0.page_size
            demand = KVDemand(tokens, {}, demands[state])
            self.assertFalse(a0.admits(demand), demands)
            self.assertFalse(a1.admits(demand), demands)
            self.assertNotEqual(
                _admits_alone(a0, demand, None), _admits_alone(a1, demand, None)
            )

    def test_a_cache_insert_slot_take_is_decided_alike(self):
        """A state slot a prefix-cache insert takes outside scheduling is decided on
        every stage: the stage that alone would still take it refuses it too."""
        a0, a1 = _stages(_STATE_STAGES)
        _, _, state = a0.sub_pool_roles()
        held = []
        while True:
            taken = _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(1))
            self.assertEqual(taken[0] is None, taken[1] is None)
            if taken[0] is None:
                break
            held.append(taken)
        self.assertTrue(_fits_alone(a0, {state: 1}))
        self.assertFalse(_fits_alone(a1, {state: 1}))

    def test_estimates_are_alike_on_every_stage(self):
        """Observer estimates read every stage's simulated snapshot, so every stage
        observes the same values, unlike the stages alone."""
        a0, a1 = _stages(_STATE_STAGES)
        _on_every_stage((a0, a1), lambda a: a.alloc(16 * a.page_size))
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(3))
        self.assertEqual(a0.usage(), a1.usage())
        self.assertEqual(a0.available_size(), a1.available_size())
        with _alone(a0), _alone(a1):
            self.assertNotEqual(a0.usage(), a1.usage())

    def _check_lockstep_simulation(self, geometries, seed):
        """As the stages take and free the same units, every take is decided alike and
        every answer is the conjunction of the stages' answers on their real layouts."""
        stages = _stages(geometries)
        a0, a1 = stages
        rng = random.Random(seed)
        owner, windows, state = a0.sub_pool_roles()
        tokens, slots = [], []
        disagreements = refusals = 0
        for _ in range(160):
            roll = rng.random()
            if roll < 0.4:
                n = rng.randrange(1, 24) * a0.page_size
                taken = _on_every_stage(stages, lambda a: a.alloc(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is None:
                    refusals += 1
                else:
                    tokens.append(taken)
            elif roll < 0.6 and state is not None:
                n = rng.randrange(1, 6)
                taken = _on_every_stage(stages, lambda a: a.alloc_state_slots(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is not None:
                    slots.append(taken)
            elif roll < 0.85 and tokens:
                pair = tokens.pop(rng.randrange(len(tokens)))
                for allocator, ids in zip(stages, pair):
                    allocator.free(ids)
            elif slots:
                pair = slots.pop(rng.randrange(len(slots)))
                for allocator, ids in zip(stages, pair):
                    allocator.mamba_allocator.free(ids)
            self.assertEqual(a0.stage_counts(), a1.stage_counts())
            for demands in _probe_demands(a0, rng, count=4):
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                disagreements += fit0 != fit1
                self.assertEqual(a0.units_fit(demands), fit0 and fit1, demands)
                self.assertEqual(a1.units_fit(demands), fit0 and fit1, demands)
                evictable = {
                    name: rng.randrange(0, 16) * (1 if name == state else 4)
                    for name in (owner, *windows, state)
                    if name is not None
                }
                demand = KVDemand(
                    demands[owner] * a0.page_size,
                    {w: demands[w] * a0.page_size for w in windows},
                    demands.get(state, 0),
                )
                alone = _admits_alone(a0, demand, evictable) and _admits_alone(
                    a1, demand, evictable
                )
                for allocator in stages:
                    self.assertEqual(
                        allocator.admits(demand, evictable=evictable),
                        alone,
                        (demands, evictable),
                    )
        self.assertGreater(disagreements, 0)
        self.assertGreater(refusals, 0)
        for allocator in stages:
            self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_lockstep_simulation_state_chain(self):
        self._check_lockstep_simulation(_STATE_STAGES, seed=0)

    def test_lockstep_simulation_window_chain(self):
        self._check_lockstep_simulation(_WINDOW_STAGES, seed=1)


class TestStageCapacityDigest(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_digests_agree_until_a_stage_drifts(self):
        """Stages that ask the same questions and hold the same units give the
        same digest; one that holds a unit more does not."""
        a0, a1 = _stages(_STATE_STAGES, checks=True)

        def digests():
            return [a.stage_capacity.take_digest(a.stage_counts()) for a in (a0, a1)]

        _on_every_stage((a0, a1), lambda a: a.alloc(8 * a.page_size))
        first = digests()
        self.assertIsNotNone(first[0])
        self.assertEqual(first[0], first[1])
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(2))
        with _alone(a1):
            a1.alloc_state_slots(1)
        second = digests()
        self.assertNotEqual(second[0], second[1])
        a0.install_stage_capacity(0, [a.stage_geometry() for a in (a0, a1)])
        self.assertIsNone(a0.stage_capacity.take_digest(a0.stage_counts()))


class _LoopDone(Exception):
    pass


def _loop_worker(rank, path, level, drift_round):
    """One stage of the real PP event loop over gloo, scheduling and model stood in;
    with ``drift_round`` the middle stage takes a slot of its own at that iteration."""
    import math
    import os
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    os.environ["SGLANG_INVARIANT_CHECK"] = str(level)
    from sglang.srt.managers import scheduler_pp_mixin
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.runtime_context import get_parallel

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=60 if drift_round is None else 15),
    )
    # One rank per stage: the attention TP broadcast is the identity.
    scheduler_pp_mixin.attn_cp_tp_broadcast_pyobj = lambda data: data
    geometry = _STATE_STAGES[rank % 2]
    allocator = build_unified_bundle(geometry).token_to_kv_pool_allocator
    page = allocator.page_size
    rounds = 24

    class Stage(SchedulerPPMixin):
        ingest_requests = Scheduler.ingest_requests

        def __init__(self):
            self.token_to_kv_pool_allocator = allocator
            self.tree_cache = None
            self.recv_skipper = None
            self.uocr_runtime = None
            self.device = "cpu"
            self.spec_algorithm = NS(is_none=lambda: True)
            self.pp_group = NS(
                is_first_rank=rank == 0,
                is_last_rank=rank == 2,
                cpu_group=dist.group.WORLD,
            )
            self.world_group = NS(cpu_group=dist.group.WORLD)
            self.pause_state = NS(
                holds_inputs=lambda: False,
                take_inputs=lambda reqs, stop_at_pause: list(reqs),
                may_launch=lambda: True,
            )
            self.request_receiver = NS(recv_requests=self._recv_requests)
            self.metrics_reporter = NS(record_scheduler_active=lambda start: None)
            self.invariant_checker = NS(run=lambda *args: None)
            self.new_token_ratio_tracker = NS(current=0.5)
            self.chunked_req = None
            self.forward_ct = 0
            self.waiting_queue = []
            self.held = []
            self.round = 0
            self.log = []

        def increment_scheduler_iterations(self):
            self.round += 1

        def _poll_timeout_aborts(self):
            return []

        def _recv_requests(self, local_reqs):
            if rank:
                return self._pp_recv_pyobj_from_prev_stage()
            # Requests of growing, then shrinking prompts; some fit no
            # FULL-heavy arena even empty.
            r = self.round
            return [(f"r{r}-{i}", (r * 37 + i * 101) % 360 + 8) for i in range(2)]

        def process_input_requests(self, reqs):
            for rid, tokens in reqs:
                fits = allocator.admits(KVDemand(tokens, {}, 1), empty_pool=True)
                self.log.append(("arrive", rid, tokens, fits))
                if fits:
                    self.waiting_queue.append(NS(rid=rid, tokens=tokens))

        def get_next_batch_to_run(self, running_batch, last_batch):
            if rank == 1 and self.round == drift_round:
                with _alone(allocator):
                    allocator.alloc_state_slots(1)
            if self.held and self.round % 3 == 0:
                req, ids, slots = self.held.pop(0)
                allocator.free(ids)
                allocator.mamba_allocator.free(slots)
                self.log.append(("finish", req.rid))
            while self.waiting_queue:
                req = self.waiting_queue[0]
                pages = math.ceil(req.tokens / page)
                if not allocator.admits(KVDemand(pages * page, {}, 1)):
                    self.log.append(("full", req.rid))
                    break
                ids = allocator.alloc(pages * page)
                slots = allocator.alloc_state_slots(1)
                assert ids is not None and slots is not None, req.rid
                self.waiting_queue.pop(0)
                self.held.append((req, ids, slots))
                self.log.append(("admit", req.rid))
            return NS(running_batch=running_batch, batch_to_run=None)

        def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *a):
            return None, None, None

        def _pp_commit_proxy_send_work(self):
            pass

        def on_idle(self):
            if self.round >= rounds:
                raise _LoopDone()

    stage = Stage()
    try:
        with get_parallel().override(
            pp_size=3,
            pp_rank=rank,
            tp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_rank=0,
            pp_async_batch_depth=0,
            world_group=NS(cpu_group=dist.group.WORLD),
        ):
            try:
                stage.event_loop_pp()
            except _LoopDone:
                pass
            except AssertionError as error:
                with open(f"{path}.failed{rank}", "w") as f:
                    f.write(str(error))
                raise
            stage._pp_commit_comm_work(stage.send_req_work)
            stage._pp_commit_comm_work(stage._pp_capacity_send_work)
        logs = [None] * 3
        dist.all_gather_object(logs, stage.log)
        assert logs[0] == logs[1] == logs[2], logs
        kinds = {entry[0] for entry in logs[0]}
        assert {"admit", "full", "finish"} <= kinds, logs[0]
        arrivals = [e for e in logs[0] if e[0] == "arrive"]
        assert not all(e[3] for e in arrivals), arrivals
        # The state-heavy stage alone would keep requests the FULL-heavy
        # stages cannot; it drops them as they do.
        if rank == 1:
            with _alone(allocator):
                alone = [
                    allocator.admits(KVDemand(e[2], {}, 1), empty_pool=True)
                    for e in arrivals
                ]
            assert alone != [e[3] for e in arrivals], arrivals
    finally:
        dist.destroy_process_group()


class TestStageCapacityInThePipelineLoop(CustomTestCase):
    def test_three_stages_schedule_alike_under_the_pp_loop(self):
        """Three gloo ranks in the real PP loop admit, defer and finish the same
        requests; with invariant checks on, their decision digests match."""
        import tempfile

        import torch.multiprocessing as mp

        for level in (0, 2):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as tmp:
                mp.spawn(
                    _loop_worker,
                    args=(tmp + "/rendezvous", level, None),
                    nprocs=3,
                    join=True,
                )

    def test_a_stage_that_drifts_fails_its_digest_check(self):
        """A stage holding a slot the others do not fails loudly at the next
        iteration's digest check."""
        import os
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/rendezvous"
            with self.assertRaises(Exception):
                mp.spawn(_loop_worker, args=(path, 2, 5), nprocs=3, join=True)
            self.assertTrue(os.path.exists(path + ".failed1"), os.listdir(tmp))
            with open(path + ".failed1") as f:
                self.assertIn("stage_capacity_agreement", f.read())


if __name__ == "__main__":
    unittest.main()
