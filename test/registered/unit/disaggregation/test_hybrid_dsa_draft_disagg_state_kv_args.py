"""PD state registration ships the EAGLE/NEXTN draft's DSA indexer cache and
kpool tail of a hybrid DSA model as components of their own.

Hybrid DSA pools (HybridLinearKVPool with use_dsa, e.g. GLM-5.3-Flash) used to
register StateType.DSA / DSA_TAIL from the target's full_kv_pool only, so the
decode-side draft ran top-k over an all-zero index for the whole prompt. The
draft's components follow the target's and keep their own key/score halves, so
the target's pairing (including the PP tail slicing) is untouched.
"""

import unittest

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 45


class _DsaPool(DSATokenToKVPool):
    def __init__(self, tag, layers, page_size=64, index_kpool=4, tail_extra_slots=6):
        self.use_dsa = True
        self.layer_num = layers
        self.page_size = page_size
        self.index_kpool = index_kpool
        self.kpool_use_compress = True
        self.tail_extra_slots = tail_extra_slots
        self._index = [f"{tag}I{i}" for i in range(layers)]
        self._tails = [f"{tag}TK{i}" for i in range(layers)] + [
            f"{tag}TS{i}" for i in range(layers)
        ]

    def get_state_buf_infos(self):
        return self._index, [4096] * self.layer_num, [256] * self.layer_num

    def get_compress_tail_buf_infos(self):
        return self._tails, [640] * len(self._tails), [64] * len(self._tails)


class _HybridPool(HybridLinearKVPool):
    def __init__(self, full_kv_pool, use_dsa=True):
        self.use_dsa = use_dsa
        self.full_kv_pool = full_kv_pool

    def get_state_buf_infos(self):
        return ["mamba"], [8], [9]

    def get_state_dim_per_tensor(self):
        return [1]

    def get_state_conv_shard_groups(self):
        return [None]

    def get_state_slice_outer_counts(self):
        return [1]

    def get_state_layer_ids(self):
        return [0]


class TestHybridDsaDraftDisaggStateKvArgs(CustomTestCase):
    def setUp(self):
        self.target_dsa = _DsaPool("t", 11)
        self.target = _HybridPool(self.target_dsa)

    def _run(self, draft):
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, self.target, draft, total_kv_layers=NUM_LAYERS)
        return kv_args

    def test_target_only_layout_is_unchanged(self):
        kv_args = self._run(None)
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )
        self.assertEqual(kv_args.state_data_ptrs[1], self.target_dsa._index)
        self.assertEqual(kv_args.state_data_ptrs[2], self.target_dsa._tails)

    def test_draft_components_follow_the_targets(self):
        draft = _DsaPool("d", 1)
        kv_args = self._run(draft)
        self.assertEqual(
            kv_args.state_types,
            [
                StateType.MAMBA,
                StateType.DSA,
                StateType.DSA_TAIL,
                StateType.DSA,
                StateType.DSA_TAIL,
            ],
        )
        # The target's components are exactly what the target-only layout registers.
        self.assertEqual(kv_args.state_data_ptrs[1], self.target_dsa._index)
        self.assertEqual(kv_args.state_data_ptrs[2], self.target_dsa._tails)
        self.assertEqual(kv_args.state_layer_ids[1], [])
        # The draft's index cache is its own component in the draft layer band.
        self.assertEqual(kv_args.state_data_ptrs[3], ["dI0"])
        self.assertEqual(kv_args.state_item_lens[3], [256])
        self.assertEqual(kv_args.state_layer_ids[3], [NUM_LAYERS])
        # The draft's tail keeps equal key/score halves of its own.
        self.assertEqual(kv_args.state_data_ptrs[4], ["dTK0", "dTS0"])

    def test_draft_with_different_geometry_is_skipped(self):
        draft = _DsaPool("d", 1, page_size=128)
        with self.assertLogs("sglang.srt.disaggregation.utils", level="WARNING"):
            kv_args = self._run(draft)
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )

    def test_draft_without_dsa_is_ignored(self):
        kv_args = self._run(_HybridPool(None, use_dsa=False))
        self.assertEqual(
            kv_args.state_types, [StateType.MAMBA, StateType.DSA, StateType.DSA_TAIL]
        )


if __name__ == "__main__":
    unittest.main()
