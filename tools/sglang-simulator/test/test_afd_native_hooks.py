"""Native constructor/splitting contracts, independent of hardware measurements."""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang_simulator.afd.adapters import create_adapter
from sglang_simulator.afd.costs import template
from sglang_simulator.afd.runner import simulate
from sglang_simulator.afd.native_config import NativeAFDConfig
from sglang_simulator.afd.serving import CostCatalog


class NativeHookTests(unittest.TestCase):
    def setUp(self):
        root = os.environ.get("SGLANG_SIMULATOR_DIRECT_SOURCE_ROOT")
        if not root:
            self.skipTest("Set SGLANG_SIMULATOR_DIRECT_SOURCE_ROOT")
        self.source = Path(root)

    def document(self, attention=4, ffn=4, stages=2):
        adapter = create_adapter(self.source, "afd", routing="direct")
        cfg = adapter.template_config(
            batch_size=attention * stages * 4, effective_kv=256, ffn_ranks=ffn
        )
        cfg.update(
            layers=3,
            native_config=dict(
                ffn_backend="direct",
                attention_lanes=attention,
                lanes=ffn,
                stages=stages,
                experimental_direct_stage_rows=4,
            ),
            model_config=dict(
                hidden_size=256,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                num_experts=16,
                num_experts_per_tok=2,
                vocab_size=1536,
            ),
        )
        return adapter, template(cfg, adapter.cost_mapping(cfg))

    def test_native_geometry_and_rank_placement(self):
        for attention, ffn in ((2, 4), (4, 4), (4, 2), (3, 4)):
            with self.subTest(attention=attention, ffn=ffn):
                adapter, doc = self.document(attention, ffn)
                result = simulate(doc, adapter=adapter)
                actors = {
                    e["actor"]
                    for e in result["events"]
                    if e["kind"] == "graph_replay_overhead"
                }
                self.assertEqual(len(actors), attention + ffn)
                messages = [
                    e for e in result["events"] if e["kind"] == "direct_A2F_message"
                ]
                self.assertEqual(len(messages), attention * ffn * 2 * 3)
                self.assertEqual(
                    {e["metadata"]["bytes"] for e in messages}, {4 * (256 * 2 + 2 * 12)}
                )
                self.assertEqual(len(result["request_placement"]), attention * 8)
                self.assertEqual(
                    result["request_placement"][-1],
                    dict(attention_rank=attention - 1, stage=1, slot=3),
                )

    def test_stage_limit_comes_from_native_split(self):
        adapter, doc = self.document(stages=3)
        with self.assertRaisesRegex(RuntimeError, "AFD_GRAPH_STAGE_COUNT_UNSUPPORTED"):
            simulate(doc, adapter=adapter)

    def test_request_catalog_uses_native_stage_placement(self):
        adapter, doc = self.document(attention=2, ffn=4)
        native = NativeAFDConfig.parse(doc["deployment"]["native_config"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cost.json").write_text(json.dumps(doc))
            (root / "catalog.json").write_text(
                json.dumps(dict(schema_version=1, decode=["cost.json"], prefill=[]))
            )
            catalog = CostCatalog(
                root / "catalog.json", self.source, native, allow_fixture=True
            )
            result = catalog.execute(
                [dict(id=str(i), extend=1, past=255) for i in range(16)], "decode"
            )
            self.assertEqual(
                result["placement"]["15"], dict(attention_rank=1, stage=1, slot=3)
            )
            self.assertEqual(len(result["request_finish_ns"]), 16)

    def test_constructor_is_executed_and_imports_restore_on_failure(self):
        adapter, doc = self.document()
        with tempfile.TemporaryDirectory() as tmp:
            for rel in adapter.verify():
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((self.source / rel).read_bytes())
            exchange = Path(tmp) / "python/sglang/srt/afd/expert_exchange.py"
            exchange.write_text(
                exchange.read_text().replace(
                    "        self.transport = transport",
                    '        raise AFDError("NATIVE_CTOR_PROBE")\n        self.transport = transport',
                )
            )
            other = create_adapter(tmp, "afd", routing="direct")
            sentinel = object()
            with patch.dict(sys.modules, {"sglang.srt.afd.pipeline": sentinel}):
                with self.assertRaisesRegex(RuntimeError, "NATIVE_CTOR_PROBE"):
                    simulate(doc, adapter=other)
                self.assertIs(sys.modules["sglang.srt.afd.pipeline"], sentinel)
