"""Where each Engram table lands in the shared store, given a model config.

The offsets must be identical on every rank and must not overlap even when the
tables differ in size, so this drives the real layout code against a fake config
instead of asserting on the arithmetic twice.
"""

import types
import unittest
from unittest import mock

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.layers import engram as engram_module
from sglang.srt.layers.engram_region import STORE_ALIGN, engram_region_offsets
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DIM, BLOCK = 256, 32
LAYER_IDS = [1, 14]


def _fake_model():
    """A two-table config whose tables differ in size.

    ``num_embeddings`` has to equal the sum of a table's primes, so it is read back
    from the layout rather than invented; the two tables still differ, which is what
    makes a uniform ``layer_index * region_len`` layout wrong.
    """
    hf = types.SimpleNamespace(
        engram_layer_ids=LAYER_IDS,
        engram_num_embeddings=[0, 0],
        engram_max_ngram_size=3,
        engram_n_heads=8,
        engram_vocab_size=1000,
        engram_head_dim=DIM,
    )
    layout = engram_module.EngramLayout.from_config(hf)
    hf.engram_num_embeddings = [
        layout.column_bounds(i)[-1] for i in range(len(LAYER_IDS))
    ]
    return types.SimpleNamespace(hf_text_config=hf)


class TestStoreLayout(CustomTestCase):
    def setUp(self):
        engram_module._store_layer_shard_rows.cache_clear()
        engram_module._store_offsets.cache_clear()
        self.model = _fake_model()

    def tearDown(self):
        engram_module._store_layer_shard_rows.cache_clear()
        engram_module._store_offsets.cache_clear()

    def _layout(self, replication="none", tp_size=4, tp_rank=0):
        config = EngramConfig.from_dict(
            {
                "placement": "cpu",
                "replication": replication,
                "table_path": "/dev/shm/engram",
            }
        )
        with (
            mock.patch.object(
                engram_module, "process_model_config", lambda: self.model
            ),
            mock.patch.object(
                engram_module,
                "get_parallel",
                lambda: types.SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank),
            ),
            mock.patch.object(
                engram_module,
                "get_exec",
                lambda: types.SimpleNamespace(
                    offload=types.SimpleNamespace(engram_config=config)
                ),
            ),
        ):
            return (
                engram_module._store_layer_shard_rows(),
                engram_module._store_offsets(DIM, BLOCK, False),
            )

    def test_every_table_gets_an_offset(self):
        _, offsets = self._layout()
        self.assertEqual(set(offsets), set(LAYER_IDS))

    def test_the_first_table_starts_at_zero(self):
        _, offsets = self._layout()
        self.assertEqual(offsets[min(LAYER_IDS)], 0)

    def test_offsets_are_aligned(self):
        _, offsets = self._layout()
        for offset in offsets.values():
            self.assertEqual(offset % STORE_ALIGN, 0)

    def test_regions_do_not_overlap_when_tables_differ_in_size(self):
        rows, offsets = self._layout()
        first, second = sorted(LAYER_IDS)
        region = engram_region_offsets([rows[first]], DIM, BLOCK)[0][1]
        self.assertGreaterEqual(offsets[second], region)

    def test_per_node_reserves_the_whole_table(self):
        rows, _ = self._layout("per_node")
        hf = self.model.hf_text_config
        self.assertEqual(
            rows,
            {lid: n for lid, n in zip(LAYER_IDS, hf.engram_num_embeddings)},
        )

    def test_an_uneven_split_reserves_the_largest_shard(self):
        rows, _ = self._layout("none", tp_size=3)
        hf = self.model.hf_text_config
        for lid, n in zip(LAYER_IDS, hf.engram_num_embeddings):
            # Ceiling of n / 3: one rank always gets the extra row.
            self.assertEqual(rows[lid], -(-n // 3))

    def test_the_layout_is_deterministic(self):
        first = self._layout()[1]
        engram_module._store_layer_shard_rows.cache_clear()
        engram_module._store_offsets.cache_clear()
        self.assertEqual(first, self._layout()[1])

    def test_ranks_get_distinct_offsets_with_replication_none(self):
        # Each rank owns only its own shard, so every (table, rank) needs its own
        # slice; the same offsets on two ranks would mean they overwrite each other.
        seen = []
        for rank in range(4):
            engram_module._store_layer_shard_rows.cache_clear()
            engram_module._store_offsets.cache_clear()
            seen.extend(self._layout("none", tp_size=4, tp_rank=rank)[1].values())
        self.assertEqual(len(set(seen)), len(seen))
        self.assertEqual(len(seen), len(LAYER_IDS) * 4)

    def test_every_rank_shares_the_offsets_when_replicated(self):
        per_rank = []
        for rank in range(4):
            engram_module._store_layer_shard_rows.cache_clear()
            engram_module._store_offsets.cache_clear()
            per_rank.append(self._layout("per_node", tp_size=4, tp_rank=rank)[1])
        self.assertEqual(len({tuple(sorted(d.items())) for d in per_rank}), 1)


if __name__ == "__main__":
    unittest.main()
