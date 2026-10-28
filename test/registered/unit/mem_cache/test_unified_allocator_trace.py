"""Regression checks for allocator trace scenarios and configuration selection."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_trace import (
    ADAPTERS,
    HostModel,
    TraceConfig,
    apply_op,
    corner_states,
    record_run,
    replay_run,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MANUAL = Path(__file__).resolve().parents[3] / "manual" / "mem_cache"


class _SharedCapacityAdapter:
    """Finite storage boundary: tokens and state slots consume the same capacity."""

    capacity = 1024

    def __init__(self, config):
        self.page_size = config.page_size
        self.state_allocator = SimpleNamespace(size=self.capacity)
        self.tokens = set()
        self.states = set()
        self.next_token = 0

    def alloc(self, n):
        if len(self.tokens) + len(self.states) + n > self.capacity:
            return None
        ids = list(range(self.next_token, self.next_token + n))
        self.next_token += n
        self.tokens.update(ids)
        return ids

    def extend(self, tokens, n):
        return self.alloc(n)

    def free(self, ids):
        self.tokens.difference_update(ids)

    def state_alloc(self, n):
        if len(self.tokens) + len(self.states) + n > self.capacity:
            return None
        ids = [i for i in range(self.capacity) if i not in self.states][:n]
        self.states.update(ids)
        return ids

    def state_free(self, ids):
        self.states.difference_update(ids)

    def prepare_state(self, n):
        pass

    def observables(self):
        return {"tokens": sorted(self.tokens), "states": sorted(self.states)}


class TestUnifiedAllocatorTrace(CustomTestCase):
    def test_state_exhaustion_scenario_checks_refusal_and_recovery(self):
        """A fixed number of state takes can miss exhaustion as pool capacity grows."""
        scenario = json.loads(
            (_MANUAL / "unified_trace_scenarios/state_slots_exhausted.json").read_text()
        )
        config = TraceConfig(chain="mamba", page_size=16, lazy=True, device="cpu")
        with patch.dict(ADAPTERS, trace_test=_SharedCapacityAdapter):
            records = record_run(
                config, ops=scenario["ops"], adapter_name="trace_test", full=True
            )
            refusals = [
                i
                for i, rec in enumerate(records)
                if rec["op"]["op"] == "state_alloc" and rec["out"] is None
            ]
            self.assertTrue(refusals)
            first_free = next(
                i for i, rec in enumerate(records) if rec["op"]["op"] == "free"
            )
            self.assertLess(refusals[-1], first_free)
            self.assertEqual(len(records[-3]["out"]), 2)
            self.assertEqual(len(records[-1]["out"]), 1)
            self.assertNotIn("state_fill", {rec["op"]["op"] for rec in records})
            self.assertIsNone(
                replay_run(config, records, adapter_name="trace_test", full=True)
            )
            with patch.object(_SharedCapacityAdapter, "capacity", 1025):
                self.assertIsNotNone(
                    replay_run(config, records, adapter_name="trace_test", full=True)
                )

    def test_state_full_corner_exhausts_more_than_512_slots(self):
        """The state-full prefix must leave no further slot available."""
        config = TraceConfig(chain="mamba", page_size=64, lazy=False, device="cpu")
        adapter, model = _SharedCapacityAdapter(config), HostModel()
        for op in corner_states(config)["state_full"]:
            apply_op(adapter, model, op)
        self.assertIsNone(adapter.state_alloc(1))
        self.assertEqual(len(model.states), adapter.state_allocator.size)

    def test_scenarios_obey_all_configuration_selectors(self):
        """Scenario files must not silently add configurations excluded by the CLI."""
        spec = importlib.util.spec_from_file_location(
            "unified_trace_compare", _MANUAL / "unified_trace_compare.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cases = (
            ("mamba", "16", "1", {"state_slots_exhausted"}),
            ("swa", "64", "0", set()),
            (
                "mamba,tri",
                "1,16",
                "0,1",
                {
                    "state_slots_exhausted",
                    "segment_frees_state_chain",
                    "float_park_and_recentre",
                    "float_relocation_closed_gate",
                },
            ),
        )
        for chains, pages, lazy, expected_scenarios in cases:
            with self.subTest(chains=chains, pages=pages, lazy=lazy):
                args = SimpleNamespace(
                    chains=chains,
                    page_sizes=pages,
                    lazy=lazy,
                    seeds="0-1",
                    scenarios=str(_MANUAL / "unified_trace_scenarios"),
                )
                runs = list(cli._runs(args))
                seeded = [run for run in runs if run[2] is not None]
                selected = {run[1].key() for run in seeded}
                scenarios = [run for run in runs if run[2] is None]
                self.assertTrue(all(run[1].key() in selected for run in scenarios))
                self.assertEqual(
                    {run[0].rsplit("/", 1)[-1] for run in scenarios},
                    expected_scenarios,
                )
                self.assertEqual(len(seeded), 2 * len(selected))


if __name__ == "__main__":
    unittest.main()
