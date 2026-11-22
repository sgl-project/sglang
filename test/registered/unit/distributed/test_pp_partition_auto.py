"""Boot-time PP layer split from per-GPU speed: the search, its constraints, the
cost model, the gathered decision and the switch."""

import itertools
import os
import random
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.distributed import pp_partition_auto as ppa
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "pp_partition_cost_model_example.json"
)
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


def _example():
    with open(EXAMPLE, "rb") as f:
        return ppa.PPPartitionCostModel.from_json(f.read(), HYBRID)


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
        reference = ppa.default_partition(48, 4)
        self.assertLess(value, max(_costs(search, reference, speed)))
        self.assertLess(part[2], max(part[0], part[1], part[3]))
        start = 0
        for n in part:
            self.assertEqual(
                set(HYBRID[start : start + n]), {"full_attention", "linear_attention"}
            )
            start += n


class TestMemoryBounds(CustomTestCase):
    """Each bound binds: without it the search picks a split that breaks it."""

    def test_every_stage_keeps_the_kv_floor(self):
        """KV token capacity is the stage minimum; the split may not lower it."""
        m = _example()
        memory = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(20e9,) * 4)
        search = ppa._Search(HYBRID, m, 4, memory, 0)
        # The last stage holds the LM head and runs on the fastest GPU.
        speed = [1.1, 1.0, 1.0, 0.85]
        free, free_value = ppa.best_partition(HYBRID, speed, m)
        floor = search.min_kv_capacity(ppa.default_partition(48, 4))
        self.assertLess(search.min_kv_capacity(free), floor)

        part, value = ppa.best_partition(HYBRID, speed, m, memory, floor)
        self.assertGreaterEqual(search.min_kv_capacity(part), floor)
        self.assertGreater(value, free_value)

    def test_activation_headroom_bounds_the_stage_that_sends_the_largest_proxy(self):
        """A stage's activations grow with its end layer here; its headroom outside
        the static pools must stay non-negative."""
        m = msgspec.structs.replace(
            _example(), kv_bytes_per_token={}, layer_cost={}, last_stage_cost=0.0
        )
        tokens = 8192
        memory = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(1.35e9,) * 4)
        search = ppa._Search(HYBRID, m, 4, memory, tokens)
        starts = [0, 12, 24, 36]
        self.assertLess(min(search.headroom(s, starts[s], 12) for s in range(4)), 0)

        part, _ = ppa.best_partition(HYBRID, [1.0] * 4, m, memory, 0.0, tokens)
        starts = [sum(part[:s]) for s in range(4)]
        self.assertTrue(
            all(search.headroom(s, starts[s], n) >= 0 for s, n in enumerate(part))
        )

        tight = ppa.StageMemory(static_bytes=(80e9,) * 4, free_bytes=(1e9,) * 4)
        with self.assertRaises(RuntimeError):
            ppa.best_partition(HYBRID, [1.0] * 4, m, tight, 0.0, tokens)


class TestCostModel(CustomTestCase):
    def test_the_example_is_valid_and_bad_input_is_rejected(self):
        self.assertEqual(_example().layer_cost["full_attention"], 2.6)
        bad = {
            "unknown field": b'{"layer_costs": {"full_attention": 2.0}}',
            "unknown layer kind": b'{"layer_cost": {"full_attn": 2.0}}',
            "negative layer cost": b'{"layer_cost": {"full_attention": -1.0}}',
            "proxy table length": b'{"proxy_bytes_per_token": [1.0, 2.0]}',
            "kv floor": b'{"kv_floor": 1.5}',
            "zero layer cost": b'{"layer_cost": {"full_attention": 0.0}}',
            "negative weight extra": b'{"first_stage_weight_bytes": -1.0}',
        }
        for name, data in bad.items():
            with (
                self.subTest(name),
                self.assertRaises((msgspec.ValidationError, ValueError)),
            ):
                ppa.PPPartitionCostModel.from_json(data, HYBRID)
        # A fitted negative intercept is accepted, but no split may give a stage a
        # non-positive cost, which the sum of squares would count as positive.
        m = ppa.PPPartitionCostModel.from_json(b'{"last_stage_cost": -3.5}', HYBRID)
        search = ppa._Search(HYBRID, m, 4, None, 0)
        self.assertFalse(search.feasible(3, 45, 3, 0.0))
        part, _ = ppa.best_partition(HYBRID, [1.0, 1.0, 1.0, 0.5], m)
        starts = [sum(part[:s]) for s in range(4)]
        self.assertTrue(
            all(search.cost(s, starts[s], n) > 0 for s, n in enumerate(part))
        )


def _row(pp_rank, speed, error=""):
    return ppa._RankRow(pp_rank, speed, 0, error)


class TestResolve(CustomTestCase):
    def _resolve(self, rows, **kwargs):
        args = dict(
            kinds=HYBRID,
            pp_rank=0,
            pp_size=4,
            all_gather=lambda row: rows,
            measure=lambda: (1.0, 0),
            pre_model_load_bytes=100e9,
            mem_fraction_static=0.8,
            m=HYBRID_COST,
            proxy_tokens=8192,
        )
        args.update(kwargs)
        return ppa.resolve(**args)

    def test_each_stage_runs_at_its_slowest_rank(self):
        # Two ranks per stage; stage 2's slow rank decides its speed.
        rows = [_row(s, 1.0) for s in range(4)] + [_row(2, 1.08)]
        rows += [_row(s, 1.0) for s in (0, 1, 3)]
        part, report = self._resolve(rows)
        expected, _ = ppa.best_partition(HYBRID, [1.0, 1.0, 1.08, 1.0], HYBRID_COST)
        self.assertEqual(part, expected)
        self.assertEqual(report["split"], list(part))
        self.assertEqual(len(report["rank_speed"]), 8)

    def test_one_rank_failing_fails_every_rank_with_its_error(self):
        """A failure before the gather (burn OOM, bad factors) must not leave the
        other ranks waiting: the failing rank still joins and all ranks raise."""

        def measure():
            raise torch.OutOfMemoryError("burn buffers")

        def gather(row):
            self.assertIn("burn buffers", row.error)
            return [row] + [_row(s, 1.0) for s in (1, 2, 3)]

        with self.assertRaisesRegex(RuntimeError, "rank 0: .*burn buffers"):
            self._resolve(None, all_gather=gather, measure=measure)
        rows = [_row(s, 1.0) for s in (1, 2, 3)] + [_row(0, 0.0, "boom")]
        with self.assertRaisesRegex(RuntimeError, "rank 3: boom"):
            self._resolve(rows)
        for speed, message in (
            (("1.0", "2.0"), "needs 4 positive factors"),
            (("1.0", "x", "1.0", "1.0"), "could not convert string to float"),
        ):
            with self.assertRaisesRegex(RuntimeError, f"rank 0: .*{message}"):
                self._resolve(None, all_gather=lambda row: [row], fixed_speed=speed)

    def test_an_outlier_is_clamped_against_the_healthy_stages(self):
        """Factors are taken against the median stage, so a disturbed GPU neither
        shifts the healthy stages nor loosens its own bound."""
        rows = [_row(0, 1.6)] + [_row(s, 1.0) for s in (1, 2, 3)]
        part, report = self._resolve(rows)
        self.assertEqual(report["clamped_stages"], [0])
        self.assertEqual(report["factors"], [ppa._FACTOR_BOUNDS[1], 1.0, 1.0, 1.0])
        expected, _ = ppa.best_partition(
            HYBRID, [ppa._FACTOR_BOUNDS[1], 1.0, 1.0, 1.0], HYBRID_COST
        )
        self.assertEqual(part, expected)


class TestMaybeApply(CustomTestCase):
    """The switch, the conflict with a fixed split, and the result reaching
    get_pp_indices, with a fake two-rank world on this process."""

    CONFIG = SimpleNamespace(num_hidden_layers=8, is_hybrid_swa=False, hidden_size=8)

    def _run(self, speed, cost_model=""):
        hybrid = SimpleNamespace(full_attention_layer_ids=[3, 7])

        def all_gather_object(out, obj, group=None):
            out[0] = obj
            if isinstance(obj, ppa._RankRow):
                out[1] = msgspec.structs.replace(obj, pp_rank=1, speed=1.3)
            else:
                out[1] = (None, "")

        with ExitStack() as stack:
            stack.enter_context(published_topology(role="scheduler", pp_size=2))
            for env, value in (
                (envs.SGLANG_ENABLE_PP_PARTITION_AUTO, True),
                (envs.SGLANG_PP_PARTITION_AUTO_SPEED, speed),
                (envs.SGLANG_PP_PARTITION_AUTO_COST_MODEL, cost_model),
            ):
                stack.enter_context(env.override(value))
            for target, value in (
                ("sglang.srt.configs.hybrid_arch.mambaish_config", hybrid),
                (
                    "sglang.srt.distributed.parallel_state.get_world_group",
                    SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None),
                ),
                ("torch.cuda.is_available", True),
            ):
                stack.enter_context(patch(target, return_value=value))
            stack.enter_context(
                patch("torch.distributed.all_gather_object", all_gather_object)
            )
            ppa.maybe_apply(
                model_config=self.CONFIG,
                is_draft_worker=False,
                mem_fraction_static=0.8,
                pre_model_load_memory=80.0,
            )
            return envs.SGLANG_PP_LAYER_PARTITION.get(), get_pp_indices(8, 0, 2)

    def test_the_split_reaches_get_pp_indices(self):
        with envs.SGLANG_PP_LAYER_PARTITION.override(None):
            # Stage 1 is 30% slower: it gives up one layer and keeps both kinds.
            self.assertEqual(self._run("1.0,1.3"), ("5,3", (0, 5)))
        self.assertNotEqual(envs.SGLANG_PP_LAYER_PARTITION.get(), "5,3")

    def test_failures_and_conflicts_raise_on_every_rank(self):
        with envs.SGLANG_PP_LAYER_PARTITION.override("4,4"):
            with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                self._run("1.0,1.3")
        with envs.SGLANG_PP_LAYER_PARTITION.override(None):
            with self.assertRaisesRegex(RuntimeError, "needs 2 positive factors"):
                self._run("1.0")
            with self.assertRaisesRegex(RuntimeError, "rank 0: .*No such file"):
                self._run("1.0,1.3", cost_model="/nonexistent/cost.json")

    def test_off_runs_no_collective(self):
        with (
            envs.SGLANG_ENABLE_PP_PARTITION_AUTO.override(False),
            envs.SGLANG_PP_LAYER_PARTITION.override(None),
            published_topology(role="scheduler", pp_size=2),
            patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                side_effect=AssertionError("no collective when off"),
            ),
        ):
            ppa.maybe_apply(
                model_config=self.CONFIG,
                is_draft_worker=False,
                mem_fraction_static=0.8,
                pre_model_load_memory=80.0,
            )
            self.assertIsNone(envs.SGLANG_PP_LAYER_PARTITION.get())


if __name__ == "__main__":
    unittest.main()
