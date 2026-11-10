"""Pipeline stages under unified memory across processes: the startup
share agreement, request limits, and the PP event loop over gloo ranks."""

import unittest

from test_unified_stage_capacity import (
    _STATE_STAGES,
    _alone,
    _generate_req,
    _request_stage,
)

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=150, stage="weekly", runner_config="cpu")

# FULL + window stages without state, one holding mostly full attention and
# the other mostly window layers: their empty arenas hold very unequal FULL.
_FULL_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="swa",
        full_layer_ids=(0,),
        swa_layer_ids=tuple(range(1, 17)),
        page_size=1,
        byte_budget=1 << 16,
        window=32,
    ),
    UnifiedGeometry(
        chain="swa",
        full_layer_ids=tuple(range(16)),
        swa_layer_ids=(16,),
        page_size=1,
        byte_budget=1 << 16,
        window=32,
    ),
)


def _limits_worker(rank, path):
    """One stage's startup request limits, then intake of a prompt between the
    stages' own limits, with and without auto-truncation."""
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    from sglang.srt.managers.req_types import FINISH_ABORT
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache import kv_cache_builder
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_memory, get_parallel, get_serving

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    allocator = build_unified_bundle(
        _FULL_WINDOW_STAGES[rank]
    ).token_to_kv_pool_allocator
    configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
    configurator.is_hybrid_swa = True
    runner = ModelRunner.__new__(ModelRunner)
    runner.kv_cache_configurator = configurator
    runner.is_hybrid_swa = True
    runner.max_total_num_tokens = runner.full_max_total_num_tokens = 64
    runner.swa_max_total_num_tokens = 64
    runner.max_running_requests = 4
    runner.token_to_kv_pool = NS(size=0)
    runner.token_to_kv_pool_allocator = allocator
    runner.req_to_token_pool = ReqToTokenPool.__new__(ReqToTokenPool)
    runner.req_to_token_pool.size = 4
    runner.req_to_token_pool.max_context_len = 1 << 14
    runner.req_to_token_pool._aux_cache = None
    runner.forward_stream = None
    worker = NS(
        model_runner=runner,
        model_config=NS(context_len=1 << 14),
        random_seed=0,
        device="cpu",
        dllm_algorithm=None,
    )
    own = [None] * 2
    dist.all_gather_object(own, allocator.size_full)
    assert own[0] != own[1], own
    try:
        with (
            get_parallel().override(
                pp_size=2,
                pp_rank=rank,
                pp_group=NS(cpu_group=dist.group.WORLD),
                tp_size=1,
                tp_rank=0,
            ),
            get_memory().override(enable_unified_memory=True),
        ):
            # The scheduler's startup order: the final pool, then its limits.
            kv_cache_builder.agree_on_stage_request_capacity(tp_worker=worker)
            info = TpModelWorker.get_worker_info(worker)
            max_req_len, max_req_input_len = info[4], info[5]
            outcomes = []
            for truncate in (False, True):
                with get_serving().override(allow_auto_truncate=truncate):
                    stage = _request_stage(
                        allocator,
                        max_req_len=max_req_len,
                        max_req_input_len=max_req_input_len,
                        # Short chunks, so the prompt fits both arenas.
                        chunk=1,
                    )
                    stage.handle_generate_request(_generate_req(min(own) - 5, 0))
                    (req,) = stage.queued
                    outcomes.append(
                        (
                            isinstance(req.to_finish, FINISH_ABORT),
                            len(req.origin_input_ids),
                        )
                    )
        every = [None] * 2
        dist.all_gather_object(every, outcomes)
        assert every[0] == every[1], every
    finally:
        dist.destroy_process_group()


class TestStageRequestLimits(CustomTestCase):
    def test_stages_reject_and_truncate_a_prompt_alike(self):
        """FULL + window stages whose empty arenas hold unequal FULL tokens
        reject, or auto-truncate, a prompt between their own limits alike."""
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_limits_worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)


# A FULL-only stage, as an uneven layer split can leave beside a FULL + window one.
_FULL_ONLY_STAGE = UnifiedGeometry(
    chain="full",
    full_layer_ids=(16, 17),
    page_size=1,
    byte_budget=1 << 16,
    start_layer=16,
)


def _ratio_host_sizes(allocator):
    """Each sub-pool's host-view capacity and the independent host pool that ratio
    sizing builds from it, as storage backends without a shared host arena do."""
    from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions

    sizes = {}
    for name, view in sorted(allocator.views.items()):
        host = allocator.storage.host_layout(name).build(
            HostPoolOptions(
                ratio=2.0,
                size=0,
                page_size=allocator.page_size,
                layout="layer_first",
                allocator_type="default",
                pool_label=name,
                pin_memory=False,
            )
        )
        sizes[name] = (view.host_capacity_tokens, host.size)
    return sizes


def _stage_shares_worker(rank, path, geometries, unlike):
    """One stage's startup share agreement, compared with every stage's."""
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    from sglang.srt.mem_cache import kv_cache_builder
    from sglang.srt.runtime_context import get_memory, get_parallel

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    allocator = build_unified_bundle(geometries[rank]).token_to_kv_pool_allocator
    worker = NS(model_runner=NS(token_to_kv_pool_allocator=allocator))
    try:
        with (
            get_parallel().override(
                pp_size=2, pp_rank=rank, pp_group=NS(cpu_group=dist.group.WORLD)
            ),
            get_memory().override(enable_unified_memory=True),
        ):
            before = None if unlike else _ratio_host_sizes(allocator)
            try:
                kv_cache_builder.agree_on_stage_request_capacity(tp_worker=worker)
            except ValueError as error:
                record = str(error)
            else:
                record = (
                    allocator.size_full,
                    allocator.size_swa,
                    _ratio_host_sizes(allocator),
                )
        every = [None] * 2
        dist.all_gather_object(every, (before, record))
        if unlike:
            assert every[0] == every[1], every
            assert "must plan the same sub-pools" in record, every
        else:
            # Alone, the stages' FULL host pools differ; agreed, every pool matches.
            assert every[0][0]["full"] != every[1][0]["full"], every
            assert every[0][1] == every[1][1], every
    finally:
        dist.destroy_process_group()


class TestStageShares(CustomTestCase):
    def _spawn(self, geometries, unlike):
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(
                _stage_shares_worker,
                args=(tmp + "/rendezvous", geometries, unlike),
                nprocs=2,
                join=True,
            )

    def test_stages_with_unlike_sub_pools_fail_alike_at_startup(self):
        """A FULL + window stage and a FULL-only stage both raise the same error at
        the startup agreement; neither waits on a collective the other skips."""
        self._spawn((_FULL_WINDOW_STAGES[0], _FULL_ONLY_STAGE), unlike=True)

    def test_ratio_sized_host_pools_agree_after_startup(self):
        """FULL + window stages whose arenas hold unequal FULL tokens keep one FULL
        and window share, and ratio sizing builds equal host pools from them."""
        self._spawn(_FULL_WINDOW_STAGES, unlike=False)


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
