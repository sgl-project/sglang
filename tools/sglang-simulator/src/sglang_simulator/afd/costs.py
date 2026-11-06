"""Model-independent cost records with explicit invocation mapping.

The execution adapter supplies expected signatures and resolves each invocation
into non-overlapping record keys. No model, backend or rank defaults live here.
"""

import copy
import math

from sglang_simulator.aic_costs import quote_record, validate_snapshot

from .contracts import CostMapping


def template(config, mapping):
    return {
        "schema_version": 1,
        "fixture": True,
        "deployment": copy.deepcopy(config),
        "costs": {
            key: {
                "shape": shape,
                "duration_ns": 10,
                "basis": "fixture",
                "source": "Synthetic control-test duration; not a benchmark",
            }
            for key, shape in mapping.expected_shapes().items()
        },
    }


class CostTable:
    def __init__(self, document, mapping: CostMapping):
        document = copy.deepcopy(document)
        schema = document["schema_version"]
        self.schema_version = schema
        if schema not in (1, 2) or type(document["fixture"]) is not bool:
            raise ValueError("AFD_COST_SCHEMA_UNSUPPORTED")
        if schema == 2 and document["fixture"]:
            raise ValueError("AFD_COST_SCHEMA2_FIXTURE_UNSUPPORTED")
        if schema == 2:
            validate_snapshot(document)
        self.config = document["deployment"]
        self.synthetic = document["fixture"]
        self.by = document["costs"]
        self.queries = []
        self.mapping = copy.deepcopy(mapping)
        expected = self.mapping.expected_shapes()
        self._shapes = expected
        if self.by.keys() != expected.keys():
            raise ValueError(
                "AFD_COST_KEYS_MISMATCH: no missing or unused costs allowed"
            )
        for key, shape in expected.items():
            record = self.by[key]
            if schema == 2:
                validated = quote_record(
                    shape,
                    record["source"],
                    request=record["request"],
                    consumer_key=key,
                    **document["cost_policy"],
                )
                if record != validated:
                    raise ValueError(f"AFD_COST_QUOTE_RECORD_MISMATCH: {key}")
            ns = record["duration_ns"]
            if (
                isinstance(ns, bool)
                or not isinstance(ns, (int, float))
                or not math.isfinite(ns)
                or ns < 1
            ):
                raise ValueError(f"AFD_COST_INVALID: {key}")
            if (
                record["shape"] != shape
                or not isinstance(record["source"], (str, dict))
                or not record["source"]
            ):
                raise ValueError(f"AFD_COST_SHAPE_OR_SOURCE_MISMATCH: {key}")
            basis = record["basis"]
            if basis not in (
                ("fixture",)
                if self.synthetic
                else ("exact", "interpolation", "formula", "table_lookup_unknown")
            ):
                raise ValueError(f"AFD_COST_BASIS_INVALID: {key}")
            if schema == 1 and basis == "table_lookup_unknown":
                raise ValueError(f"AFD_COST_BASIS_INVALID: {key}")
            if basis == "interpolation" and not record.get("neighbors"):
                raise ValueError(f"AFD_COST_INTERPOLATION_NEIGHBORS_MISSING: {key}")
            if basis == "formula" and not record.get("equation"):
                raise ValueError(f"AFD_COST_FORMULA_MISSING: {key}")

    def components(self, kind, **metadata):
        keys = list(self.mapping.resolve(kind, metadata))
        if not keys or len(set(keys)) != len(keys):
            raise ValueError(f"AFD_COST_COMPONENTS_INVALID: {kind}")
        if any(key not in self.by for key in keys):
            raise ValueError(f"AFD_COST_QUERY_MISSING: {kind}")
        return [(key, self.by[key]) for key in keys]

    def event_components(self, event):
        kind, metadata = self.mapping.event_call(event)
        return self.components(kind, **metadata)

    def quote(self, kind, **metadata):
        records = self.components(kind, **metadata)
        query = dict(kind=kind, **metadata)
        if self.schema_version == 2:
            query["resolved_costs"] = [
                {
                    "key": key,
                    "shape": record["shape"],
                    "method": record["basis"],
                    "source": record["source"]["source"],
                    "qualification": record["source"]["qualification"],
                }
                for key, record in records
            ]
        self.queries.append(query)
        return round(sum(record["duration_ns"] for _, record in records))
