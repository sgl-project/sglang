"""Boot-time PP layer split from per-GPU speed: the search, its constraints and the
agreement across ranks."""

import itertools
import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import pp_partition_auto as ppa
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

UNIFORM = ppa.PPPartitionCostModel()
# A hybrid layout: one full-attention layer in every four, the rest linear attention.
HYBRID = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)]
HYBRID_COST = ppa.PPPartitionCostModel(
    layer_cost={"full_attention": 2.6, "linear_attention": 1.0},
    first_stage_cost=0.5,
    last_stage_cost=1.5,
)


def _costs(search, part, speed):
    starts = [sum(part[:s]) for s in range(len(part))]
    return [search.cost(s, starts[s], n) * speed[s] for s, n in enumerate(part)]


def _splits(num_layers, pp_size):
    for cuts in itertools.combinations(range(1, num_layers), pp_size - 1):
        bounds = (0,) + cuts + (num_layers,)
        yield tuple(b - a for a, b in zip(bounds, bounds[1:]))


class TestBestPartition(CustomTestCase):
    def test_uniform_layers_and_gpus_keep_the_default_split(self):
        """With nothing to balance, turning the search on changes no layer range."""
        for num_layers, pp_size in ((93, 8), (10, 4), (61, 4), (32, 8)):
            part, _ = ppa.best_partition(
                ["layer"] * num_layers, [1.0] * pp_size, UNIFORM
            )
            ranges = [get_pp_indices(num_layers, s, pp_size) for s in range(pp_size)]
            self.assertEqual(part, tuple(end - start for start, end in ranges))

    def test_matches_exhaustive_search(self):
        """The least slowest stage, then the least sum of squares, then the smallest
        split; checked against every split of small hybrid models."""
        rng = random.Random(0)
        for _ in range(20):
            num_layers, pp_size = rng.randint(8, 14), rng.randint(2, 4)
            kinds = [
                rng.choice(("full_attention", "linear_attention"))
                for _ in range(num_layers)
            ]
            speed = [round(rng.uniform(0.9, 1.1), 3) for _ in range(pp_size)]
            search = ppa._Search(kinds, HYBRID_COST, pp_size, None, 0)
            feasible = [
                p
                for p in _splits(num_layers, pp_size)
                if all(search.feasible(s, sum(p[:s]), n, 0.0) for s, n in enumerate(p))
            ]
            if not feasible:
                with self.assertRaises(RuntimeError):
                    ppa.best_partition(kinds, speed, HYBRID_COST)
                continue

            def key(p):
                costs = [round(c, 6) for c in _costs(search, p, speed)]
                return (max(costs), round(sum(c * c for c in costs), 6), p)

            expected = min(feasible, key=key)
            part, value = ppa.best_partition(kinds, speed, HYBRID_COST)
            self.assertEqual(part, expected)
            self.assertAlmostEqual(value, key(expected)[0], places=5)

    def test_a_slow_gpu_gets_less_work_and_every_stage_keeps_every_layer_kind(self):
        speed = [1.0, 1.0, 1.08, 1.0]
        part, value = ppa.best_partition(HYBRID, speed, HYBRID_COST)
        search = ppa._Search(HYBRID, HYBRID_COST, 4, None, 0)
        reference = ppa.even_partition(48, 4)
        self.assertLess(value, max(_costs(search, reference, speed)))
        self.assertLess(part[2], max(part[0], part[1], part[3]))
        start = 0
        for n in part:
            self.assertEqual(
                set(HYBRID[start : start + n]), {"full_attention", "linear_attention"}
            )
            start += n


class TestMemoryBounds(CustomTestCase):
    MODEL = ppa.PPPartitionCostModel(
        layer_cost={"full_attention": 2.6, "linear_attention": 1.0},
        layer_weight_bytes={"full_attention": 4e9, "linear_attention": 4e9},
        layer_state_bytes={"linear_attention": 1e9},
        kv_bytes_per_token={"full_attention": 2e4},
        last_stage_weight_bytes=6e9,
    )

    def test_every_stage_keeps_the_kv_floor(self):
        """KV token capacity is the stage minimum; the split may not lower it."""
        memory = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(20e9,) * 4)
        search = ppa._Search(HYBRID, self.MODEL, 4, memory, 0)
        # The last stage holds the LM head and runs on the fastest GPU.
        speed = [1.1, 1.0, 1.0, 0.85]
        free, free_value = ppa.best_partition(HYBRID, speed, self.MODEL)
        floor = search.min_kv_capacity(ppa.even_partition(48, 4))
        self.assertLess(search.min_kv_capacity(free), floor)

        part, value = ppa.best_partition(HYBRID, speed, self.MODEL, memory, floor)
        self.assertGreaterEqual(search.min_kv_capacity(part), floor)
        self.assertGreater(value, free_value)

    def test_activation_headroom_bounds_the_stage_that_sends_the_largest_proxy(self):
        """A stage sends activations whose size grows with its end layer here; its
        headroom outside the static pools must stay non-negative."""
        model = ppa.PPPartitionCostModel(
            proxy_bytes_per_token=[1000.0 * (1 + end // 8) for end in range(49)],
            proxy_buffers_in=2,
            proxy_buffers_out=3,
            activation_headroom_bytes=1e9,
        )
        kinds = ["layer"] * 48
        tokens = 16384
        memory = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(1.35e9,) * 4)
        search = ppa._Search(kinds, model, 4, memory, tokens)
        reference = ppa.even_partition(48, 4)
        starts = [sum(reference[:s]) for s in range(4)]
        self.assertLess(min(search.headroom(s, starts[s], 12) for s in range(4)), 0)

        part, _ = ppa.best_partition(kinds, [1.0] * 4, model, memory, 0.0, tokens)
        starts = [sum(part[:s]) for s in range(4)]
        self.assertTrue(
            all(search.headroom(s, starts[s], n) >= 0 for s, n in enumerate(part))
        )

        tight = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(1e9,) * 4)
        with self.assertRaises(RuntimeError):
            ppa.best_partition(kinds, [1.0] * 4, model, tight, 0.0, tokens)


class TestResolve(CustomTestCase):
    def _resolve(self, pp_rank, rows, chosen=None, **kwargs):
        args = dict(
            kinds=HYBRID,
            pp_rank=pp_rank,
            pp_size=4,
            all_gather=lambda row: rows,
            broadcast_from_0=lambda part: part if chosen is None else chosen,
            measure=lambda: self.fail("fixed speeds take no burn"),
            total_bytes=100e9,
            mem_fraction_static=0.8,
            m=HYBRID_COST,
            reference=ppa.even_partition(48, 4),
            proxy_tokens=8192,
            fixed_speed=[1.0, 1.0, 1.08, 1.0],
        )
        args.update(kwargs)
        return ppa.resolve(**args)

    def test_each_stage_runs_at_its_slowest_rank_and_ranks_agree(self):
        # Two ranks per stage; stage 2's slow rank decides its speed.
        rows = [(s, 1.0, 100e9) for s in range(4)] + [(2, 1.08, 100e9)]
        rows += [(s, 1.0, 100e9) for s in (0, 1, 3)]
        part, report = self._resolve(
            pp_rank=0, rows=rows, fixed_speed=(), measure=lambda: 1.0, burn_scale=1.0
        )
        expected, _ = ppa.best_partition(
            HYBRID, [1.0 + (v / 1.02 - 1.0) for v in (1.0, 1.0, 1.08, 1.0)], HYBRID_COST
        )
        self.assertEqual(part, expected)
        self.assertEqual(report["split"], list(part))
        with self.assertRaises(AssertionError):
            self._resolve(pp_rank=1, rows=rows, chosen=(12, 12, 12, 12))


class TestCostModelSource(CustomTestCase):
    def test_json_cost_model_and_layer_kinds_from_the_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cost.json")
            with open(path, "w") as f:
                json.dump(
                    {"layer_cost": {"full_attention": 2.6}, "last_stage_cost": 1.5}, f
                )
            m = ppa.PPPartitionCostModel.from_json(path)
        self.assertEqual(m.layer_cost, {"full_attention": 2.6})
        self.assertEqual(m.kv_bytes_per_token, {})

        hybrid = SimpleNamespace(full_attention_layer_ids=[3, 7])
        config = SimpleNamespace(num_hidden_layers=8, is_hybrid_swa=False)
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=hybrid
        ):
            kinds = ppa.layer_kinds(config)
        self.assertEqual(
            [i for i, kind in enumerate(kinds) if kind == "full_attention"], [3, 7]
        )
        self.assertEqual(kinds.count("linear_attention"), 6)
        swa = SimpleNamespace(
            num_hidden_layers=4, is_hybrid_swa=True, swa_attention_layer_ids=[0, 1, 2]
        )
        with patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None):
            self.assertEqual(
                ppa.layer_kinds(swa), ["sliding_attention"] * 3 + ["full_attention"]
            )


if __name__ == "__main__":
    unittest.main()
