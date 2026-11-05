"""Unit tests for the layer router's part routes and their load-back waits."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART
from sglang.test.test_utils import CustomTestCase

_START_LAYER = 4
_FULL_LAYERS = {7: 0, 11: 1}
_STATE_LAYERS = {4: 0, 5: 1, 6: 2, 8: 3}


class _RecordingCounter:
    def __init__(self):
        self.waits = []

    def wait_until(self, threshold: int) -> None:
        self.waits.append(threshold)


class _Pool:
    """Pipeline stage 4 has full layers 7 and 11 and state layers 4, 5, 6 and 8."""

    def __init__(self):
        self.start_layer = _START_LAYER
        self.layer_transfer_counter = None
        self.full_view = object()
        self.state_view = object()
        self.ngram_view = object()
        self.router = KVLayerRouter.from_layer_maps(
            self, [("full", self.full_view, _FULL_LAYERS)]
        )
        self.router.bind_part(
            sub_pool="mamba",
            part=STATE_PART,
            view=self.state_view,
            indices=_STATE_LAYERS,
        )
        self.router.bind_part(
            sub_pool="mamba",
            part="ngram",
            view=self.ngram_view,
            indices={min(_STATE_LAYERS): 0},
        )


class _StageWithoutStateLayers:
    """The last pipeline stage holds only full-attention layer 47; the model
    has recurrent layers, but none are assigned to this stage."""

    def __init__(self):
        self.start_layer = 47
        self.layer_transfer_counter = None
        self.router = KVLayerRouter.from_layer_maps(self, [("full", object(), {47: 0})])


class _PoolWithoutTokenKV:
    """A pool that holds only recurrent state: no token KV sub-pool, so its
    router is built from no layer maps and the state part is bound to it."""

    def __init__(self):
        self.start_layer = 0
        self.layer_transfer_counter = None
        self.router = KVLayerRouter.from_layer_maps(self, ())


class TestKVLayerRouterParts(CustomTestCase):
    def test_state_reads_wait_only_once_the_host_stack_loads_state(self):
        """State reads wait at their stage-local layer only when the host stack
        restores state; KV-only restores must not block them."""
        pool = _Pool()
        counter = _RecordingCounter()
        pool.layer_transfer_counter = counter

        pool.router.read(7)
        state = pool.router.read(6, STATE_PART)
        self.assertEqual(counter.waits, [7 - _START_LAYER])
        self.assertIs(state.view, pool.state_view)
        self.assertEqual(state.view_layer_id, 2)
        self.assertEqual(state.route.sub_pool, "mamba")

        pool.router.load_parts_layer_wise()
        pool.router.read(6, STATE_PART)
        self.assertEqual(counter.waits, [7 - _START_LAYER, 6 - _START_LAYER])

    def test_out_of_loop_readers_wait_for_their_own_layer(self):
        """N-gram reads wait for the first state layer, while whole-state copies
        wait for the last; waiting for the last early would serialize load-back."""
        pool = _Pool()
        counter = _RecordingCounter()
        pool.layer_transfer_counter = counter
        pool.router.load_parts_layer_wise()

        pool.router.wait_for_part("ngram")
        self.assertIs(pool.router.part_view("ngram"), pool.ngram_view)
        pool.router.wait_for_part(STATE_PART)
        self.assertEqual(counter.waits, [4 - _START_LAYER, 8 - _START_LAYER])

    def test_a_part_with_no_layer_on_this_stage_waits_for_nothing(self):
        """A stage without state layers must not wait for a state or N-gram load,
        even when the host stack restores state layer by layer."""
        stage = _StageWithoutStateLayers()
        state_view, ngram_view = object(), object()
        stage.router.bind_part(
            sub_pool="mamba", part=STATE_PART, view=state_view, indices={}
        )
        stage.router.bind_part(
            sub_pool="mamba", part="ngram", view=ngram_view, indices={}
        )
        stage.router.wait_for_part(STATE_PART)
        stage.router.wait_for_part("ngram")
        counter = _RecordingCounter()
        stage.layer_transfer_counter = counter
        stage.router.load_parts_layer_wise()
        stage.router.wait_for_part(STATE_PART)
        stage.router.wait_for_part("ngram")
        self.assertEqual(counter.waits, [])
        self.assertIs(stage.router.part_view(STATE_PART), state_view)
        self.assertIs(stage.router.part_view("ngram"), ngram_view)

    def test_a_stage_without_state_layers_binds_and_reuses_a_prefix(self):
        """A real request table on a stage without state layers binds empty routes
        and performs a prefix-hit state copy without waiting for nonexistent layers."""
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=64,
            n_groups=1,
            num_heads=4,
            head_dim=16,
            state_size=8,
            conv_kernel=4,
        )
        table = HybridReqToTokenPool(
            size=4,
            mamba_size=8,
            mamba_spec_state_size=4,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[]),
            mamba_layer_ids=[],
            enable_mamba_extra_buffer=False,
            ngram_context_len=3,
            ngram_eos_token_id=0,
        )
        stage = _StageWithoutStateLayers()
        stage.layer_transfer_counter = _RecordingCounter()
        table.state.bind_routes(stage.router)
        stage.router.load_parts_layer_wise()

        stage.router.wait_for_part(STATE_PART)
        table.state.copy(torch.tensor([1]), torch.tensor([2]))
        stage.router.wait_for_part(NGRAM_PART)
        self.assertEqual(stage.layer_transfer_counter.waits, [])
        self.assertIs(stage.router.part_view(STATE_PART), table.state.pool)
        self.assertIs(stage.router.part_view(NGRAM_PART), table.state.ngram_pool)

    def test_parts_stay_out_of_the_token_sub_pools(self):
        """Callers that pick "the first layer holding KV" or iterate the token
        sub-pools must not see state layers."""
        pool = _Pool()
        self.assertEqual(pool.router.layer_ids(), (7, 11))
        self.assertEqual(
            [sub_pool.name for sub_pool in pool.router.sub_pools()], ["full"]
        )
        self.assertIsNone(pool.router.sub_pool("mamba"))
        with self.assertRaises(KeyError):
            pool.router.layer(6)
        with self.assertRaises(KeyError):
            pool.router.layer(7, STATE_PART)

    def test_a_router_without_token_kv_routes_its_bound_parts(self):
        """A router built from no layer maps lists no token KV sub-pools, and a
        part bound to it routes each layer to the index it was bound at."""
        pool = _PoolWithoutTokenKV()
        state_view = object()
        pool.router.bind_part(
            sub_pool="mamba", part=STATE_PART, view=state_view, indices={2: 1, 3: 0}
        )
        self.assertEqual(pool.router.sub_pools(), ())
        self.assertEqual(pool.router.layer_ids(), ())
        self.assertIsNone(pool.router.wrapped_view())
        self.assertEqual(pool.router.part_indices(STATE_PART), {2: 1, 3: 0})
        state = pool.router.read(2, STATE_PART)
        self.assertIs(state.view, state_view)
        self.assertEqual(
            (state.route.sub_pool, state.route.index, state.view_layer_id),
            ("mamba", 1, 1),
        )
        with self.assertRaises(KeyError):
            pool.router.layer(2)


if __name__ == "__main__":
    unittest.main()
