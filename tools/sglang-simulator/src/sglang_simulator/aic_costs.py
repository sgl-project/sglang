"""Shared AIC quote validation and replay preparation, without model defaults.

AIC owns lookup, interpolation and formulas. Callers supply model invocation
shapes; this module retains the returned measurement scope and qualifications.
"""

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path


SOURCES = {
    "hardware_measurement",
    "vendor_simulation",
    "analytical_assumption",
    "unknown",
}
METHODS = {"exact", "interpolation", "formula", "table_lookup_unknown"}
QUALIFICATIONS = {"matched", "conditional", "unknown", "unsupported"}


def validate_quote(
    quote, *, allowed_qualifications=("matched",), allow_unknown_source=False
):
    """Validate declared evidence; this does not certify it independently."""
    if quote.get("schema") != "aic.cost_quote.v1":
        raise ValueError("AIC_QUOTE_SCHEMA_UNSUPPORTED")
    ms = quote.get("latency_ms")
    if (
        isinstance(ms, bool)
        or not isinstance(ms, (int, float))
        or not math.isfinite(ms)
        or ms <= 0
    ):
        raise ValueError("AIC_QUOTE_LATENCY_INVALID")
    if quote.get("source") not in SOURCES or quote.get("method") not in METHODS:
        raise ValueError("AIC_QUOTE_SOURCE_OR_METHOD_INVALID")
    qualification = quote.get("qualification", {})
    status = qualification.get("status")
    allowed = set(allowed_qualifications)
    if not allowed <= QUALIFICATIONS - {"unsupported"}:
        raise ValueError("AIC_QUOTE_POLICY_INVALID")
    if status not in allowed or status == "unsupported":
        raise ValueError("AIC_QUOTE_QUALIFICATION_REJECTED")
    if not isinstance(qualification.get("reasons"), list):
        raise ValueError("AIC_QUOTE_QUALIFICATION_REASONS_MISSING")
    if status != "matched" and not qualification["reasons"]:
        raise ValueError("AIC_QUOTE_QUALIFICATION_REASONS_MISSING")
    if quote["source"] == "unknown" and not allow_unknown_source:
        raise ValueError("AIC_QUOTE_UNKNOWN_SOURCE_REJECTED")
    if not quote.get("op") or not quote.get("system"):
        raise ValueError("AIC_QUOTE_IDENTITY_MISSING")
    for field in ("query", "conditions", "scope"):
        if not isinstance(quote.get(field), dict) or not quote[field]:
            raise ValueError(f"AIC_QUOTE_{field.upper()}_MISSING")
    if (
        not isinstance(quote["scope"].get("included"), list)
        or not quote["scope"]["included"]
        or not isinstance(quote["scope"].get("excluded"), list)
    ):
        raise ValueError("AIC_QUOTE_SCOPE_BOUNDARY_MISSING")
    if not isinstance(quote.get("provenance"), list):
        raise ValueError("AIC_QUOTE_PROVENANCE_MISSING")
    if not isinstance(quote.get("assumptions"), list):
        raise ValueError("AIC_QUOTE_ASSUMPTIONS_MISSING")
    method = quote["method"]
    if method == "exact" and (
        not quote["provenance"] or quote["source"] == "analytical_assumption"
    ):
        raise ValueError("AIC_QUOTE_EXACT_EVIDENCE_MISSING")
    if method == "interpolation":
        neighbors = quote.get("neighbors")
        if (
            not isinstance(neighbors, list)
            or len(neighbors) < 2
            or not all(isinstance(n, dict) and n for n in neighbors)
        ):
            raise ValueError("AIC_QUOTE_INTERPOLATION_NEIGHBORS_MISSING")
    if method == "table_lookup_unknown" and quote.get("neighbors") not in (None, []):
        raise ValueError("AIC_QUOTE_UNKNOWN_LOOKUP_NEIGHBORS_INVALID")
    if method == "formula" and not (quote.get("equation") or quote.get("derivation")):
        raise ValueError("AIC_QUOTE_FORMULA_MISSING")
    return copy.deepcopy(quote)


def _at(value, path):
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _contains(actual, expected):
    return (
        all(
            key in actual
            and (
                _contains(actual[key], value)
                if isinstance(value, dict)
                else actual[key] == value
            )
            for key, value in expected.items()
        )
        if isinstance(actual, dict)
        else False
    )


def validate_binding(quote, request, shape, key):
    """Check the frozen operation request, including explicit shape aliases."""
    if request.get("consumer_key") != key or request.get("consumer_shape") != shape:
        raise ValueError("AIC_QUOTE_CONSUMER_BINDING_MISMATCH")
    for field in ("op", "system"):
        if not request.get(field) or quote[field] != request[field]:
            raise ValueError(f"AIC_QUOTE_REQUEST_{field.upper()}_MISMATCH")
    for field in ("query", "conditions", "scope"):
        if (
            not isinstance(request.get(field), dict)
            or not request[field]
            or not _contains(quote[field], request[field])
        ):
            raise ValueError(f"AIC_QUOTE_REQUEST_{field.upper()}_MISMATCH")
    bindings = request.get("shape_bindings", {})
    if bindings.keys() != shape.keys():
        raise ValueError("AIC_QUOTE_SHAPE_BINDING_INCOMPLETE")
    for field, value in shape.items():
        binding = bindings[field]
        expected = value
        if "alias" in binding:
            if isinstance(value, (int, float)) or not binding.get("reason"):
                raise ValueError("AIC_QUOTE_SHAPE_ALIAS_INVALID")
            expected = binding["alias"]
        if "factor" in binding or "offset" in binding:
            if not isinstance(value, (int, float)) or not binding.get("reason"):
                raise ValueError("AIC_QUOTE_SHAPE_TRANSFORM_INVALID")
            expected = value * binding.get("factor", 1) + binding.get("offset", 0)
        try:
            actual = _at(quote, binding["quote_path"])
        except (KeyError, TypeError, IndexError, ValueError) as error:
            raise ValueError("AIC_QUOTE_SHAPE_PATH_MISSING") from error
        if actual != expected:
            raise ValueError(f"AIC_QUOTE_SHAPE_BINDING_MISMATCH: {field}")


def quote_record(shape, quote, *, request, consumer_key, **policy):
    quote = validate_quote(quote, **policy)
    validate_binding(quote, request, shape, consumer_key)
    ns = quote["latency_ms"] * 1e6
    if not math.isfinite(ns) or ns < 1:
        raise ValueError("AIC_QUOTE_DURATION_INVALID")
    record = {
        "shape": copy.deepcopy(shape),
        "duration_ns": ns,
        "basis": quote["method"],
        "source": quote,
        "request": copy.deepcopy(request),
    }
    if quote["method"] == "interpolation":
        record["neighbors"] = copy.deepcopy(quote["neighbors"])
    if quote["method"] == "formula":
        record["equation"] = quote.get("equation") or quote["derivation"]
    return record


def input_identity(paths):
    """Bind existing dataset/resolver files, without modifying their contents."""
    records = []
    for path in sorted({str(Path(p).absolute()) for p in paths}):
        file = Path(path)
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"AIC_INPUT_NOT_REGULAR: {path}")
        data = file.read_bytes()
        records.append(
            {
                "path": path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not records:
        raise ValueError("AIC_INPUT_IDENTITY_EMPTY")
    return records


def document_digest(document):
    payload = {k: v for k, v in document.items() if k != "snapshot_sha256"}

    def json_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("AIC_SNAPSHOT_NON_STRING_KEY")
            for item in value.values():
                json_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                json_keys(item)

    # JSON coerces integer keys to strings. Sorting before and after that
    # coercion can produce different digests, so reject ambiguous input before
    # publishing instead of weakening replay's content check.
    json_keys(payload)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def evidence_identities(quote):
    """Read file identities without rewriting the provider's original evidence."""
    dataset = quote.get("dataset")
    if dataset and dataset.get("root") and dataset.get("manifest_sha256"):
        # Native providers verify the published dataset. Historical raw paths
        # inside its records remain provenance metadata, not live dependencies.
        return [
            (
                str(Path(dataset["root"]) / "DATASET_MANIFEST.json"),
                dataset["manifest_sha256"],
            )
        ]
    identities = []
    for origin in quote["provenance"]:
        if not isinstance(origin, dict):
            raise ValueError("AIC_QUOTE_PROVENANCE_ENTRY_INVALID")
        if "path" in origin:
            # Historical manifest evidence used path + manifest digest. Never
            # replace an explicit raw-file sha256, even if it is invalid/None.
            digest = (
                origin["sha256"]
                if "sha256" in origin
                else origin.get("dataset_manifest_sha256")
            )
            identities.append((origin["path"], digest))
        for path_key, sha_key in (
            ("trace", "trace_sha256"),
            ("dataset_manifest", "dataset_manifest_sha256"),
        ):
            if path_key in origin:
                identities.append((origin[path_key], origin.get(sha_key)))
    dataset = quote.get("dataset")
    if dataset and dataset.get("root") and dataset.get("manifest_sha256"):
        identities.append(
            (
                str(Path(dataset["root"]) / "DATASET_MANIFEST.json"),
                dataset["manifest_sha256"],
            )
        )
    return identities


def validate_billing_boundaries(costs, *, require_declarations=False):
    """An explicitly mapped invocation uses a module or its leaves, never both."""
    groups = {}
    for record in costs.values():
        billing = record.get("request", {}).get("billing")
        if billing is None:
            if require_declarations:
                raise ValueError("AIC_BILLING_BOUNDARY_REQUIRED")
            continue  # Existing schema2 snapshots retain their original mapping.
        if (
            not isinstance(billing, dict)
            or not isinstance(billing.get("group"), str)
            or not billing["group"]
            or billing.get("kind") not in ("module", "leaf")
        ):
            raise ValueError("AIC_BILLING_BOUNDARY_INVALID")
        groups.setdefault(billing["group"], []).append(billing["kind"])
    if any("module" in kinds and len(kinds) != 1 for kinds in groups.values()):
        raise ValueError("AIC_MODULE_AND_LEAF_DOUBLE_BILLING")


def _validate_evidence_binding(costs, identities, *, phase):
    """Cross-check retained evidence identities without opening origin paths.

    Only the quote's explicit top-level evidence and dataset manifest count as
    bound inputs. Nested historical environment/source metadata stays intact;
    replay does not require those original machines or paths to remain online.
    """
    by_path = {record["path"]: record for record in identities}
    for record in costs.values():
        evidence = evidence_identities(record["source"])
        if not evidence:
            raise ValueError(f"{phase}_PROVENANCE_EMPTY")
        for path, digest in evidence:
            if path not in by_path or digest != by_path[path]["sha256"]:
                raise ValueError(f"{phase}_SOURCE_NOT_BOUND")


def validate_snapshot(document):
    validate_billing_boundaries(document["costs"])
    identities = document.get("input_identity")
    if not isinstance(identities, list) or not identities:
        raise ValueError("AIC_SNAPSHOT_INPUT_IDENTITY_MISSING")
    for record in identities:
        if (
            not isinstance(record.get("path"), str)
            or not record["path"]
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sha256"])
        ):
            raise ValueError("AIC_SNAPSHOT_INPUT_IDENTITY_INVALID")
    if document.get("resolver_identity") not in identities:
        raise ValueError("AIC_SNAPSHOT_RESOLVER_IDENTITY_MISSING")
    if document.get("snapshot_sha256") != document_digest(document):
        raise ValueError("AIC_SNAPSHOT_CONTENT_CHANGED")
    _validate_evidence_binding(document["costs"], identities, phase="AIC_SNAPSHOT")


def prepare_cost_document(
    deployment,
    mapping,
    resolver,
    *,
    requests,
    input_paths,
    allowed_qualifications=("matched",),
    allow_unknown_source=False,
):
    """Resolve every expected invocation and make an automatic schema2 replay.

    resolver(request) returns an AIC typed quote, not a duration
    stripped of metadata. It may call a core op, measured module or vendor
    provider; none of those numerical algorithms are implemented here.
    """
    before = input_identity(input_paths)
    resolver_file = inspect.getsourcefile(
        resolver
        if inspect.isfunction(resolver) or inspect.ismethod(resolver)
        else type(resolver)
    )
    if not resolver_file:
        raise ValueError("AIC_PREPARATION_RESOLVER_IDENTITY_MISSING")
    resolver_path = str(Path(resolver_file).absolute())
    resolver_identity = next((x for x in before if x["path"] == resolver_path), None)
    if resolver_identity is None:
        raise ValueError("AIC_PREPARATION_RESOLVER_NOT_BOUND")
    policy = {
        "allowed_qualifications": list(allowed_qualifications),
        "allow_unknown_source": allow_unknown_source,
    }
    expected = mapping.expected_shapes()
    if not expected:
        raise ValueError("AIC_COST_MAPPING_EMPTY")
    if requests.keys() != expected.keys():
        raise ValueError("AIC_PREPARATION_REQUEST_KEYS_MISMATCH")
    costs = {}
    for key, shape in expected.items():
        request = copy.deepcopy(requests[key])
        request.update(consumer_key=key, consumer_shape=copy.deepcopy(shape))
        costs[key] = quote_record(
            shape,
            resolver(copy.deepcopy(request)),
            request=request,
            consumer_key=key,
            **policy,
        )
    validate_billing_boundaries(costs, require_declarations=True)
    # Quote sources must be among the bound input files, not an unrelated token
    # passed to satisfy a nonempty input_paths argument. Replay needs only this
    # retained identity; it does not require old absolute paths to stay online.
    _validate_evidence_binding(costs, before, phase="AIC_PREPARATION")
    if input_identity(input_paths) != before:
        raise ValueError("AIC_PREPARATION_INPUT_CHANGED")
    document = {
        "schema_version": 2,
        "fixture": False,
        "deployment": copy.deepcopy(deployment),
        "cost_policy": policy,
        "input_identity": before,
        "resolver_identity": copy.deepcopy(resolver_identity),
        "costs": costs,
    }
    document["snapshot_sha256"] = document_digest(document)
    return document


def write_snapshot(document, path):
    """Publish one prepared replay, refusing to overwrite prior evidence."""
    with Path(path).open("x") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")
