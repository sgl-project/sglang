"""Pipeline stages under unified memory: every stage asks each capacity
question of every stage's geometry, so all of them decide alike."""

import unittest
from contextlib import contextmanager

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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

    def test_digests_do_not_depend_on_band_order(self):
        """Stages may hold the same bands in a different arena order; the same
        questions and the same units held give the same digest."""
        a0, a1 = _stages(_STATE_STAGES, checks=True)
        _on_every_stage((a0, a1), lambda a: a.alloc(8 * a.page_size))
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(2))
        counts = a1.stage_counts()
        self.assertTrue(all(len(c) > 1 for c in counts))
        reordered = tuple(dict(reversed(c.items())) for c in counts)
        self.assertEqual(
            a0.stage_capacity.take_digest(a0.stage_counts()),
            a1.stage_capacity.take_digest(reordered),
        )

    def test_digests_agree_after_a_request_is_clipped_to_memory(self):
        """A request one stage clips to memory and relays reaches the next stage
        with its limit as received, so both stages clip alike and digest alike."""
        import pickle

        from sglang.srt.managers.mm_utils import strip_mm_features_for_pp_relay

        stages = [_request_stage(a) for a in _stages(_STATE_STAGES, checks=True)]
        requested = 4096
        recv_req = _generate_req(12, requested)
        for s in stages:
            s.handle_generate_request(recv_req)
            # The relay pickles the request as this stage left it.
            (recv_req,) = pickle.loads(
                pickle.dumps(strip_mm_features_for_pp_relay([recv_req]))
            )
        clipped = [s.queued[0].sampling_params.max_new_tokens for s in stages]
        self.assertLess(clipped[0], requested)
        self.assertEqual(clipped[0], clipped[1])
        for s in stages:
            s._pp_mark_capacity_digest()
        self.assertIsNotNone(stages[0]._pp_capacity_digest)
        self.assertEqual(stages[0]._pp_capacity_digest, stages[1]._pp_capacity_digest)


def _request_stage(
    allocator, *, max_req_len=1 << 14, max_req_input_len=1 << 14, chunk=None
):
    """A scheduler that runs production request intake on ``allocator``."""
    from types import SimpleNamespace as NS

    from sglang.srt.disaggregation.mode import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.token_to_kv_pool_allocator = allocator
    if allocator.stage_capacity is not None:
        s._pp_checked_allocator = allocator
    s.page_size = allocator.page_size
    s.max_req_len, s.max_req_input_len = max_req_len, max_req_input_len
    s.max_total_num_tokens = 1 << 14
    s.kv_shard_widening = 1
    s.max_new_tokens_limit = None
    s.sliding_window_size = None
    s.chunked_prefill_size, s.truncation_align_size = chunk, None
    s.tree_cache = NS(
        supports_prefix_sharing=lambda: True,
        window_checkpoint_interval=lambda: 0,
        cache_controller=None,
    )
    s.enable_session_radix_cache = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.model_config = NS(hf_eos_token_id={1}, vocab_size=128)
    s.metrics_reporter = NS(enable_metrics=False)
    s.tokenizer = s.dllm_config = None
    s.spec_algorithm = NS(is_dflash_family=lambda: False, is_uno=lambda: False)
    s._maybe_namespace_elastic_radix_cache = lambda req: None
    s._record_kv_event_lora_name = lambda req, name: None
    s.invariant_checker = NS(expect_output=lambda req: None)
    s.grammar_manager = NS(process_req_with_grammar=lambda req: False)
    s.queued = []
    s._add_request_to_queue = s.queued.append
    return s


def _generate_req(input_len, max_new_tokens):
    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    return TokenizedGenerateReqInput(
        rid="r",
        input_text="",
        input_ids=array("q", range(input_len)),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


if __name__ == "__main__":
    unittest.main()
