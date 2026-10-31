"""Each MTP draft depth owns private state; its clears and copies must leave
the target unchanged. Drafts without a private depth share the target state."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.srt.mem_cache.kv_cache_configurator import runner_state
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

_STATE_LAYERS = [0, 1, 2]
_DEPTHS = 3


def _params(layers):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=1,
        num_heads=4,
        head_dim=16,
        state_size=8,
        conv_kernel=4,
    )
    return Mamba2CacheParams(shape=shape, layers=layers)


def _tensors(state):
    cache = state.pool.mamba_cache
    return [*cache.conv, cache.temporal]


def _snapshot(state):
    return [tensor.clone() for tensor in _tensors(state)]


class TestDraftDepthState(CustomTestCase):
    def setUp(self):
        self.target = HybridReqToTokenPool(
            size=4,
            mamba_size=8,
            mamba_spec_state_size=4,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
            cache_params=_params(_STATE_LAYERS),
            mamba_layer_ids=_STATE_LAYERS,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        for tensor in _tensors(self.target.state):
            tensor.fill_(1.0)
        # The configurator hands the target's state to the token allocator,
        # which every draft worker shares.
        kv = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=16, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False
        )
        self.allocator.state = self.target.state

    def _depth(self, depth):
        clone = self.target.clone_with_new_mamba(
            mamba_size=8,
            mamba_spec_state_size=4,
            cache_params=_params([depth]),
            device="cpu",
            enable_mamba_extra_buffer=False,
            draft_model_idx=depth,
        )
        state = runner_state(clone, self.allocator, table_owns_state=True)
        router = KVLayerRouter.from_layer_maps(
            SimpleNamespace(start_layer=0, layer_transfer_counter=None),
            [("full", object(), {depth: 0})],
        )
        state.bind_routes(router)
        return clone, state, router

    def test_each_depth_binds_its_own_state(self):
        states = []
        for depth in range(_DEPTHS):
            clone, state, router = self._depth(depth)
            self.assertIs(state, clone.state)
            self.assertIsNot(state, self.target.state)
            self.assertIs(router.part_view(STATE_PART), clone.state.pool)
            self.assertIsNot(router.part_view(STATE_PART), self.target.state.pool)
            states.append(state)
        self.assertEqual(len({id(state) for state in states}), _DEPTHS)

    def test_draft_writes_clears_and_copies_leave_the_target_alone(self):
        before = _snapshot(self.target.state)
        for depth in range(_DEPTHS):
            _, _, router = self._depth(depth)
            pool = router.part_view(STATE_PART)
            for tensor in [*pool.mamba_cache.conv, pool.mamba_cache.temporal]:
                tensor.fill_(float(depth + 2))
            pool.clear_slots(torch.tensor([1]))
            pool.copy_from(torch.tensor([2]), torch.tensor([3]))
        for kept, now in zip(before, _tensors(self.target.state)):
            self.assertTrue(torch.equal(kept, now))

    def test_a_draft_that_does_not_own_its_depth_shares_the_target_state(self):
        state = runner_state(self.target, self.allocator, table_owns_state=False)
        self.assertIs(state, self.target.state)


if __name__ == "__main__":
    unittest.main()
