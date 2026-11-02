"""Focused AFD contracts; exhaustive historical regressions live on the backup branch."""

import copy
import os
import socket
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang_simulator.afd.adapters import create_adapter
from sglang_simulator.afd.adapters.sglang.loader import temporary_modules
from sglang_simulator.afd.coordinator import DeadlockError, EventCoordinator
from sglang_simulator.afd.costs import CostTable, template
from sglang_simulator.afd.native_config import NativeAFDConfig
from sglang_simulator.afd.runner import simulate
from sglang_simulator.afd.serving import AFDRequestPredictor, AFDTaskServer
from sglang_simulator.compat import (
    SGLangCompatibilityError,
    request_extend_length,
    request_prefix_length,
)


class AFDTests(unittest.TestCase):
    def test_dependency_contention_and_receive_wait(self):
        c = EventCoordinator()
        a = c.submit("a", "compute", 7, ("gpu0",))
        b = c.submit("b", "compute", 11, ("gpu1",))
        d = c.submit("d", "compute", 3, ("gpu0",), deps=(b,))
        self.assertEqual([c.finish(x) for x in (a, b, d)], [7, 11, 14])
        recv = c.recv("d", "c", "a", "b", 0, duration_ns=3, resources=("gpu",))
        work = c.submit("b", "compute", 5, ("gpu",))
        with self.assertRaises(DeadlockError):
            c.drain()
        self.assertEqual(c.events[work]["finish_ns"], 5)
        c.send("d", "c", "a", "b", 0, duration_ns=10, resources=("link",))
        self.assertEqual(c.finish(recv), 13)
        c.send("e", "c", "a", "b", 0, duration_ns=8, resources=("link",))
        self.assertEqual(c.finish(len(c.events) - 1), 18)

    def test_cycles_fail_without_fabricated_completion(self):
        c = EventCoordinator(defer_cycle_checks=True)
        a, b = c.reserve("a", "a"), c.reserve("b", "b")
        c.resolve(a, deps=(b,), duration_ns=1)
        c.resolve(b, deps=(a,), duration_ns=1)
        with self.assertRaises(DeadlockError):
            c.finish(b)
        self.assertTrue(all(e["finish_ns"] is None for e in c.export()))

    def test_cost_shapes_provenance_and_boundaries(self):
        for routing in ("f-side", "direct"):
            adapter = create_adapter("unused", "afd", routing=routing)
            cfg = adapter.template_config(
                batch_size=256, effective_kv=8448, ffn_ranks=8
            )
            mapping = adapter.cost_mapping(cfg)
            doc = template(cfg, mapping)
            provider = CostTable(doc, mapping)
            self.assertEqual(provider.quote("attention_module", layer=0, rows=16), 60)
            for value in (0, -1, True, float("nan")):
                bad = copy.deepcopy(doc)
                bad["costs"]["qkv"]["duration_ns"] = value
                with self.subTest(routing=routing, value=value), self.assertRaises(
                    ValueError
                ):
                    CostTable(bad, mapping)
            for change in ("missing", "shape", "fixture", "interpolation", "formula"):
                bad = copy.deepcopy(doc)
                if change == "missing":
                    del bad["costs"]["qkv"]
                elif change == "shape":
                    bad["costs"]["qkv"]["shape"]["m"] += 1
                else:
                    bad["fixture"] = False
                    if change != "fixture":
                        for row in bad["costs"].values():
                            row["basis"] = "exact"
                        bad["costs"]["qkv"]["basis"] = change
                with self.subTest(routing=routing, change=change), self.assertRaises(
                    ValueError
                ):
                    CostTable(bad, mapping)
            with self.assertRaises(ValueError):
                provider.quote("qkv", rows=99)
            if routing == "direct":
                shapes = mapping.expected_shapes()
                self.assertEqual(shapes["direct_A2F_message"]["bytes"], 132608)
                self.assertNotIn("all_gatherv", shapes)
                self.assertEqual(provider.quote("router", rows=16), 10)

    def test_native_configuration_and_request_api(self):
        for backend, routing in (("fixed", "f-side"), ("direct", "direct")):
            self.assertEqual(
                NativeAFDConfig.parse(dict(ffn_backend=backend, lanes=8)).routing,
                routing,
            )
        for raw in ({}, dict(ffn_backend="direct", lanes=4)):
            with self.assertRaises(ValueError):
                NativeAFDConfig.parse(raw)
        self.assertEqual(request_prefix_length(NS(prefix_indices=[1, 2])), 2)
        self.assertEqual(request_prefix_length(NS(prefix_len=0)), 0)
        self.assertEqual(request_extend_length(NS(extend_range=NS(length=4))), 4)
        self.assertEqual(request_extend_length(NS(extend_input_len=4)), 4)
        for reader in (request_prefix_length, request_extend_length):
            with self.assertRaises(SGLangCompatibilityError):
                reader(NS())

    def test_module_restoration_on_failure(self):
        name = "_afd_test_module"
        with patch.dict(sys.modules, {name: None}):
            with self.assertRaises(RuntimeError):
                with temporary_modules({name: NS()}):
                    raise RuntimeError("injected")
            self.assertIsNone(sys.modules[name])

    def test_task_rpc_preserves_request_offsets(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        native = NativeAFDConfig.parse(
            dict(ffn_backend="direct", lanes=8, rendezvous_port=port)
        )
        catalog = NS(digest="test", fixture=True)
        catalog.execute = lambda requests, phase: dict(
            duration_ns=20,
            request_finish_ns={"r0": 10, "r1": 20},
            placement={},
            method="test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            server = AFDTaskServer(native, catalog, Path(tmp) / "events")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                predictor = AFDRequestPredictor(afd_config=native.raw)
                batch = NS(
                    forward_phase="decode",
                    reqs=[
                        NS(request_id=r, extend_length=1, past_kv_length=4)
                        for r in ("r0", "r1")
                    ],
                )
                self.assertEqual(predictor.predict_infer_time(batch), 20e-9)
                self.assertEqual(
                    predictor.completion_offsets(), {"r0": 10e-9, "r1": 20e-9}
                )
                self.assertTrue((server.output / "step-000000.json").is_file())
                self.assertFalse(
                    predictor.get_metrics()["afd_service_accuracy_verified"]
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

    def check_source(self, routing, env):
        source = os.environ.get(env)
        if not source:
            self.skipTest(f"Set {env} to matching source")
        adapter = create_adapter(source, "afd", routing=routing)
        cfg = adapter.template_config(batch_size=256, effective_kv=8448, ffn_ranks=8)
        result = simulate(template(cfg, adapter.cost_mapping(cfg)), adapter=adapter)
        events = result["events"]
        counts = Counter(e["kind"] for e in events)
        key = "direct_A2F_message" if routing == "direct" else "A2F"
        self.assertEqual(counts[key], 8 * 2 * 94 * (8 if routing == "direct" else 1))
        if routing == "direct":
            self.assertEqual(counts["router"], 8 * 2 * 94)
            self.assertFalse(
                any(
                    e["kind"] == "router" and e["actor"].startswith("F") for e in events
                )
            )
        self.assertTrue(all(e["finish_ns"] is not None for e in events))
        self.assertIsNone(result["total_tgs"])
        self.assertFalse(result["service_admitted"])
        with tempfile.TemporaryDirectory() as tmp:
            for rel in result["source_files"]:
                dest = Path(tmp) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((Path(source) / rel).read_bytes())
            dest.write_bytes(dest.read_bytes() + b"\n# drift\n")
            create_adapter(tmp, "afd", routing=routing).verify()
            pipeline = Path(tmp) / "python/sglang/srt/afd/pipeline.py"
            pipeline.write_text(
                pipeline.read_text().replace("def _run_layers(", "def removed(")
            )
            with self.assertRaisesRegex(ValueError, "AFD_SOURCE_API_MISMATCH"):
                create_adapter(tmp, "afd", routing=routing).verify()

    def test_fixed_source(self):
        self.check_source("f-side", "SGLANG_SIMULATOR_SOURCE_ROOT")

    def test_direct_source(self):
        self.check_source("direct", "SGLANG_SIMULATOR_DIRECT_SOURCE_ROOT")
