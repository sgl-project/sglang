"""The EAGLE/NEXTN draft's DSA index cache and kpool tail of a hybrid DSA pool
(e.g. GLM-5.3-Flash) register behind the target's, in the same components."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 45


def _dsa_pool(tag, layers):
    pool = object.__new__(DSATokenToKVPool)
    pool.use_dsa = True
    pool.layer_num = layers
    pool.index_kpool = 4
    pool.kpool_use_compress = True
    pool.index = [f"{tag}I{i}" for i in range(layers)]
    pool.tails = [f"{tag}TK{i}" for i in range(layers)] + [
        f"{tag}TS{i}" for i in range(layers)
    ]
    pool.get_state_buf_infos = lambda: (
        list(pool.index),
        [4096] * layers,
        [256] * layers,
    )
    pool.get_compress_tail_buf_infos = lambda: (
        list(pool.tails),
        [640] * len(pool.tails),
        [64] * len(pool.tails),
    )
    return pool


def _hybrid_pool(full_kv_pool, use_dsa=True):
    pool = object.__new__(HybridLinearKVPool)
    pool.use_dsa = use_dsa
    pool.full_kv_pool = full_kv_pool
    pool.layer_router = SimpleNamespace(
        sub_pool=lambda kind: None, wrapped_view=lambda: full_kv_pool
    )
    pool.get_state_buf_infos = lambda: (["mamba"], [8], [9])
    pool.get_state_dim_per_tensor = lambda: [1]
    pool.get_state_conv_shard_groups = lambda: [None]
    pool.get_state_slice_outer_counts = lambda: [1]
    pool.get_state_layer_ids = lambda: [0]
    return pool


class TestHybridDsaDraftDisaggStateKvArgs(CustomTestCase):
    def setUp(self):
        self.target_dsa = _dsa_pool("t", 11)
        self.target = _hybrid_pool(self.target_dsa)

    def _run(self, draft):
        allocator = SimpleNamespace(
            storage=None, state=None, get_kvcache=lambda: self.target
        )
        storage = StaticKVStorage(
            None,
            allocator,
            draft_token_to_kv_pool=draft,
            num_hidden_layers=NUM_LAYERS,
        )
        kv_args = KVArgs()
        setup_state_kv_args(
            kv_args, storage, draft_token_to_kv_pool=draft, total_kv_layers=NUM_LAYERS
        )
        return kv_args

    def test_target_only_layout_is_unchanged(self):
        kv_args = self._run(None)
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )
        self.assertEqual(kv_args.state_data_ptrs[1], self.target_dsa.index)
        self.assertEqual(kv_args.state_data_ptrs[2], self.target_dsa.tails)

    def test_draft_entries_follow_the_targets_in_the_same_components(self):
        draft = _dsa_pool("d", 1)
        kv_args = self._run(draft)
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )
        self.assertEqual(kv_args.state_data_ptrs[1], self.target_dsa.index + ["dI0"])
        self.assertEqual(kv_args.state_item_lens[1], [256] * 12)
        self.assertEqual(
            kv_args.state_data_ptrs[2], self.target_dsa.tails + ["dTK0", "dTS0"]
        )
        self.assertEqual(kv_args.state_item_lens[2], [64] * 24)

    def test_draft_without_dsa_is_ignored(self):
        kv_args = self._run(_hybrid_pool(None, use_dsa=False))
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )
        self.assertEqual(kv_args.state_data_ptrs[1], self.target_dsa.index)


if __name__ == "__main__":
    unittest.main()
