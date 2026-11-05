"""Model-independent cost records with explicit invocation mapping.

The execution adapter supplies expected signatures and resolves each invocation
into non-overlapping record keys. No model, backend or rank defaults live here.
"""

import copy
import math

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
        if document["schema_version"] != 1 or type(document["fixture"]) is not bool:
            raise ValueError("AFD_COST_SCHEMA_UNSUPPORTED")
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
                else ("exact", "interpolation", "formula")
            ):
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
        self.queries.append(dict(kind=kind, **metadata))
        return round(sum(record["duration_ns"] for _, record in records))
