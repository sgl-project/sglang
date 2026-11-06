"""Quote identity, billing and provider compatibility; no hardware fixtures."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from sglang_simulator.aic_costs import (
    document_digest,
    input_identity,
    prepare_cost_document,
    quote_record,
)
from sglang_simulator.aic_providers import AICProviders, load_aic_config
from sglang_simulator.afd.costs import CostTable


class Mapping:
    def expected_shapes(self):
        return {"projection": {"m": 4}}

    def resolve(self, kind, metadata):
        return ["projection"]

    def event_call(self, event):
        return "projection", {}


def request():
    return dict(
        op="projection",
        system="device",
        query={"m": 4},
        conditions={"mode": "graph"},
        scope={"included": ["projection"], "excluded": []},
        shape_bindings={"m": {"quote_path": "query.m"}},
        billing={"group": "one_projection", "kind": "module"},
    )


def resolve(req):
    return dict(
        schema="aic.cost_quote.v1",
        op=req["op"],
        system=req["system"],
        query=req["query"],
        conditions=req["conditions"],
        scope=req["scope"],
        latency_ms=0.25,
        source="hardware_measurement",
        method="exact",
        qualification={"status": "matched", "reasons": []},
        provenance=input_identity([__file__]),
        neighbors=[],
        assumptions=[],
        derivation={},
    )


class CostContractTests(unittest.TestCase):
    def document(self):
        return prepare_cost_document(
            {"mode": "afd"},
            Mapping(),
            resolve,
            requests={"projection": request()},
            input_paths=[__file__],
        )

    def test_units_legacy_and_typed_replay(self):
        doc = self.document()
        typed = CostTable(doc, Mapping())
        self.assertEqual(typed.quote("projection"), 250000)
        self.assertEqual(typed.queries[0]["resolved_costs"][0]["method"], "exact")
        legacy = dict(
            schema_version=1,
            fixture=False,
            deployment=doc["deployment"],
            costs={
                "projection": {
                    k: v
                    for k, v in doc["costs"]["projection"].items()
                    if k != "request"
                }
            },
        )
        self.assertEqual(CostTable(legacy, Mapping()).quote("projection"), 250000)
        bad = copy.deepcopy(doc)
        bad["costs"]["projection"]["duration_ns"] += 1
        with self.assertRaises(ValueError):
            CostTable(bad, Mapping())
        bad["snapshot_sha256"] = document_digest(bad)
        with self.assertRaisesRegex(ValueError, "RECORD_MISMATCH"):
            CostTable(bad, Mapping())

    def test_scope_shape_qualification_and_unknown_are_not_upgraded(self):
        req = dict(request(), consumer_key="projection", consumer_shape={"m": 4})
        q = resolve(req)
        for field, value in (
            ("scope", {"included": ["whole_attention"], "excluded": []}),
            ("query", {"m": 8}),
        ):
            changed = dict(q, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                quote_record({"m": 4}, changed, request=req, consumer_key="projection")
        for status in ("conditional", "unknown", "unsupported"):
            changed = dict(
                q, qualification={"status": status, "reasons": ["fixture limitation"]}
            )
            with self.subTest(status=status), self.assertRaises(ValueError):
                quote_record({"m": 4}, changed, request=req, consumer_key="projection")
        q.update(method="table_lookup_unknown")
        record = quote_record({"m": 4}, q, request=req, consumer_key="projection")
        self.assertEqual(record["basis"], "table_lookup_unknown")
        self.assertNotIn("neighbors", record)
        from sglang_simulator.afd.runner import critical_path

        doc = self.document()
        doc["costs"]["projection"] = record
        doc["snapshot_sha256"] = document_digest(doc)
        path = critical_path(
            [
                dict(
                    id=0,
                    duration=250000,
                    start_ns=0,
                    finish_ns=250000,
                    resources=[],
                    deps=[],
                )
            ],
            CostTable(doc, Mapping()),
        )
        self.assertEqual(path["contributions_ns"]["table_lookup_unknown"], 250000)
        self.assertEqual(path["declared_exact_interpolation_fraction"], 0)
        for method in ("interpolation", "formula"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                quote_record(
                    {"m": 4},
                    dict(q, method=method),
                    request=req,
                    consumer_key="projection",
                )

    def test_module_and_leaf_cannot_share_one_billing_group(self):
        class Both(Mapping):
            def expected_shapes(self):
                return {"module": {"m": 4}, "leaf": {"m": 4}}

        requests = {
            k: dict(request(), billing={"group": "A0.layer0.attention", "kind": k})
            for k in ("module", "leaf")
        }
        with self.assertRaisesRegex(ValueError, "DOUBLE_BILLING"):
            prepare_cost_document(
                {}, Both(), resolve, requests=requests, input_paths=[__file__]
            )

    def test_new_preparation_requires_billing_but_legacy_replay_remains_valid(self):
        incomplete = request()
        del incomplete["billing"]
        with self.assertRaisesRegex(ValueError, "BILLING_BOUNDARY_REQUIRED"):
            prepare_cost_document(
                {},
                Mapping(),
                resolve,
                requests={"projection": incomplete},
                input_paths=[__file__],
            )
        historical = self.document()
        del historical["costs"]["projection"]["request"]["billing"]
        historical["snapshot_sha256"] = document_digest(historical)
        self.assertEqual(CostTable(historical, Mapping()).quote("projection"), 250000)

    def test_relative_paths_are_independent_of_cwd_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SDK_CONFIG.json"
            for value in ("../elsewhere", "/absolute"):
                path.write_text(
                    json.dumps(
                        dict(paths_relative_to="SDK_CONFIG.json parent", root=value)
                    )
                )
                with self.assertRaises(ValueError):
                    load_aic_config(path)
            path.write_text(
                json.dumps(
                    dict(paths_relative_to="SDK_CONFIG.json parent", root="data")
                )
            )
            self.assertEqual(
                load_aic_config(path)["root"], str((Path(tmp) / "data").resolve())
            )

    @unittest.skipUnless(
        os.environ.get("SGLANG_SIMULATOR_AIC_CONFIG"),
        "Requires published native SDK loader",
    )
    def test_repository_anchor_requires_root_and_verifies_published_inputs(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            source = data / "table"
            source.write_bytes(b"published")
            path = data / "SDK_CONFIG.json"
            config = dict(
                schema="aic.sdk_dataset_config.v1",
                paths_relative_to="repository root",
                root="data",
                input_files=[
                    dict(
                        path="data/table",
                        bytes=9,
                        sha256=hashlib.sha256(b"published").hexdigest(),
                    )
                ],
            )
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_aic_config(path)
            self.assertEqual(
                load_aic_config("data/SDK_CONFIG.json", repository_root=root)["root"],
                str(data.resolve()),
            )
            source.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Pinned dataset input changed"):
                load_aic_config(path, repository_root=root)


@unittest.skipUnless(
    os.environ.get("SGLANG_SIMULATOR_AIC_CONFIG"),
    "Set a published full AIC config for native provider checks",
)
class NativeProviderTests(unittest.TestCase):
    def setUp(self):
        self.providers = AICProviders(
            os.environ["SGLANG_SIMULATOR_AIC_CONFIG"],
            repository_root=os.environ.get("SGLANG_SIMULATOR_AIC_REPOSITORY_ROOT"),
            allowed_qualifications=("matched", "conditional", "unknown"),
            allow_unknown_source=True,
        )

    def test_published_module_and_archive_policy(self):
        cfg = self.providers.config["gb300_module"]
        db = self.providers._module("gb300_module", {})
        r = db.available_requests()[0]
        r.update(
            system=db.rows[next(iter(db.rows))][0]["system"],
            resolution={"provider": "module", "dataset": "gb300_module"},
        )
        q = self.providers(r)
        self.assertEqual(q["dataset"]["root"], cfg["root"])
        self.assertEqual(q["method"], "exact")
        self.providers.config["inactive_test"] = dict(cfg, active=False)
        r["resolution"]["dataset"] = "inactive_test"
        with self.assertRaisesRegex(ValueError, "ARCHIVE"):
            self.providers(r)

    def test_ppu_curve_and_outside_domain(self):
        resolver = self.providers._module("ppu_module", {}, ppu=True)
        curve = next(iter(resolver._curves.values()))
        coordinate = sum(curve["domain"]) // 2
        query = dict(curve["fixed_query"], **{curve["axis"]: coordinate})
        r = dict(
            op=curve["op"],
            system="ppu15_zw_m890p",
            query=query,
            conditions=curve["conditions"],
            scope=curve["scope"],
            resolution=dict(
                provider="ppu",
                dataset="ppu_module",
                curve_id=curve["id"],
                coordinate=coordinate,
            ),
        )
        q = self.providers(r)
        self.assertEqual(q["query"], query)
        self.assertEqual(q["method"], "interpolation")
        r["resolution"]["coordinate"] = curve["domain"][1] + 1
        with self.assertRaises(LookupError):
            self.providers(r)

    def test_core_unknown_lookup_and_formula_keep_distinct_methods(self):
        core = dict(
            op="attention",
            system="gb300",
            query=dict(
                b=16,
                s=8448,
                n=64,
                n_kv=4,
                head_size=128,
                quant="fp8",
                lane_order=["trtllm_mha"],
            ),
            conditions=dict(backend="sglang", data_version="0.5.14"),
            scope=dict(included=["attention"], excluded=[]),
            resolution=dict(
                provider="core",
                dataset="community_systems_root",
                backend="sglang",
                version="0.5.14",
                operation="generation_attention",
            ),
        )
        quote = self.providers(core)
        self.assertEqual(quote["method"], "table_lookup_unknown")
        self.assertEqual(quote["neighbors"], [])
        formula = dict(
            core,
            op="auxiliary",
            query={"bytes": 16384},
            resolution=dict(
                provider="formula",
                dataset="community_systems_root",
                backend="sglang",
                version="0.5.14",
                formula="memory",
                parameters={"bytes_touched": 16384},
            ),
        )
        quote = self.providers(formula)
        self.assertEqual(quote["method"], "formula")
        self.assertEqual(quote["source"], "analytical_assumption")
        core["conditions"]["data_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "VERSION_OR_BACKEND"):
            self.providers(core)


if __name__ == "__main__":
    unittest.main()
