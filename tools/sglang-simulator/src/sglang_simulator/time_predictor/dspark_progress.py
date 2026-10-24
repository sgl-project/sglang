"""Independent empirical DSPark progress from retained training/heldout rows.

This loader fits conditional commit-count distributions from training rows only.
It reports heldout distribution/mean errors without inventing an acceptance
threshold. A successfully loaded artifact is not an accuracy admission.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sglang_simulator.simulation.sglang.dspark import DSparkProgress
from sglang_simulator.time_predictor.base import PredictorError

DEFINITION = "dspark_recipe_prefix_history_empirical_commit_v1"


def _error(message, **details):
    raise PredictorError("invalid_progress_artifact", message, **details)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                _error("duplicate progress JSON field")
            result[k] = v
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _canonical(value)
        return value
    except (ValueError, TypeError) as error:
        raise PredictorError(
            "invalid_progress_artifact", "progress JSON must be finite and valid"
        ) from error


class EmpiricalDSparkProgress:
    """Sample modeled commits from before-work, never from a live accept result.

    A row's trajectory identifies an independent request/conversation campaign;
    training and heldout trajectories cannot overlap. Its retained native key and
    full recipe are provenance. Prefix/history binning is an explicit model
    hypothesis whose heldout errors remain visible.
    """

    def __init__(self, path, *, expected_sha256, runtime_profile_json):
        from infercast.sdk.native_sources import NATIVE_SOURCE_0522
        from infercast.sdk.native_windows import MAX_NATIVE_WINDOW_BYTES

        path = Path(path)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 64 * 1024**2
        ):
            _error("progress artifact must be a regular file")
        path = path.resolve(strict=True)
        raw = path.read_bytes()
        if (
            not _digest(expected_sha256)
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            _error("progress artifact digest differs")
        data = _strict_json(raw)
        fields = {
            "schema_version",
            "definition",
            "runtime_profile_sha256",
            "source_pin",
            "seed",
            "prefix_edges",
            "history_edges",
            "training",
            "validation",
        }
        if (
            not isinstance(data, dict)
            or set(data) != fields
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["definition"] != DEFINITION
        ):
            _error("progress artifact fields or definition differ")
        if (
            data["runtime_profile_sha256"]
            != hashlib.sha256(runtime_profile_json.encode()).hexdigest()
            or data["source_pin"] != NATIVE_SOURCE_0522.name
        ):
            _error("progress model must match the exact runtime/source cohort")
        if type(data["seed"]) is not int or data["seed"] < 0:
            _error("progress seed must be explicit and nonnegative")
        for name in ("prefix_edges", "history_edges"):
            edges = data[name]
            if (
                not isinstance(edges, list)
                or any(type(v) is not int or v < 1 for v in edges)
                or edges != sorted(set(edges))
            ):
                _error("progress bin edges must be sorted distinct positive integers")
        self.model_ref = "sha256:" + expected_sha256
        self.profile_digest = data["runtime_profile_sha256"]
        self.source_pin = data["source_pin"]
        self.seed = data["seed"]
        self.prefix_edges = tuple(data["prefix_edges"])
        self.history_edges = tuple(data["history_edges"])
        self.distributions = defaultdict(Counter)
        trajectories = {"training": set(), "validation": set()}
        seen = set()
        rows = {}
        request_splits = {}
        native_cache = {}

        def read_evidence(ref):
            if (
                not isinstance(ref, dict)
                or set(ref) != {"path", "sha256"}
                or not _digest(ref["sha256"])
            ):
                _error("invalid native progress evidence descriptor")
            relative = ref["path"]
            if (
                type(relative) is not str
                or re.fullmatch(r"[a-zA-Z0-9_.-]+\.json", relative) is None
            ):
                _error("native progress evidence must be a retained sibling JSON")
            key = relative, ref["sha256"]
            if key not in native_cache:
                p = path.parent / relative
                if (
                    p.is_symlink()
                    or not p.is_file()
                    or p.stat().st_size > MAX_NATIVE_WINDOW_BYTES
                ):
                    _error("native progress evidence must be bounded regular bytes")
                with p.open("rb") as stream:
                    content = stream.read(MAX_NATIVE_WINDOW_BYTES + 1)
                if len(content) > MAX_NATIVE_WINDOW_BYTES:
                    _error("native progress evidence must be bounded regular bytes")
                if hashlib.sha256(content).hexdigest() != ref["sha256"]:
                    _error("native progress evidence digest differs")
                native_cache[key] = _strict_json(content)
            return native_cache[key]

        def read_rows(split):
            descriptors = data[split]
            if not isinstance(descriptors, list) or not descriptors:
                _error(
                    "progress needs independent nonempty training and validation originals"
                )
            result = []
            for ref in descriptors:
                if (
                    not isinstance(ref, dict)
                    or set(ref) != {"path", "sha256"}
                    or not _digest(ref["sha256"])
                ):
                    _error("progress original descriptor differs")
                relative = ref["path"]
                if (
                    type(relative) is not str
                    or re.fullmatch(
                        r"progress-observations-" + ref["sha256"] + r"\.json", relative
                    )
                    is None
                ):
                    _error("progress observations must be retained beside the artifact")
                original = path.parent / relative
                if (
                    original.is_symlink()
                    or not original.is_file()
                    or original.stat().st_size > 64 * 1024**2
                ):
                    _error("progress observations must be bounded regular originals")
                raw_rows = original.read_bytes()
                if hashlib.sha256(raw_rows).hexdigest() != ref["sha256"]:
                    _error("progress observation bytes differ")
                document = _strict_json(raw_rows)
                if (
                    not isinstance(document, dict)
                    or set(document)
                    != {
                        "schema_version",
                        "source_pin",
                        "runtime_profile_sha256",
                        "observations",
                        "native",
                        "participants",
                        "phase_policy",
                        "causal_history_evidence",
                    }
                    or type(document["schema_version"]) is not int
                    or document["schema_version"] != 1
                    or document["source_pin"] != self.source_pin
                    or document["runtime_profile_sha256"] != self.profile_digest
                    or not isinstance(document["observations"], list)
                ):
                    _error(
                        "progress observations differ from their runtime/source cohort"
                    )
                if (
                    not isinstance(document["causal_history_evidence"], list)
                    or not document["causal_history_evidence"]
                ):
                    _error("progress history projection needs retained causal evidence")
                for ref in document["causal_history_evidence"]:
                    read_evidence(ref)
                identities, batch_sizes = _verify_native_progress_context(
                    document, read_evidence
                )
                topology = tuple(
                    document["participants"][0][k]
                    for k in ("tp_size", "attn_dp_size", "pp_size")
                )
                if hasattr(self, "topology") and self.topology != topology:
                    _error("progress originals mix different execution topologies")
                self.topology = topology
                for row in document["observations"]:
                    self._validate_row(row)
                    previous_split = request_splits.setdefault(row["rid"], split)
                    if previous_split != split:
                        _error(
                            "native request lineage overlaps training and validation"
                        )
                    key = identities[row["native_key"]]
                    if key in seen:
                        _error(
                            "duplicate native progress observation or train/heldout leakage"
                        )
                    seen.add(key)
                    trajectories[split].add(row["trajectory_id"])
                    result.append((row, batch_sizes[row["native_key"]]))
            if not result:
                _error("progress split must contain observations")
            return result

        for split in ("training", "validation"):
            rows[split] = read_rows(split)
        if trajectories["training"] & trajectories["validation"]:
            _error("progress training and validation trajectories overlap")
        batch_support = {"training": set(), "validation": set()}
        for split, observations in rows.items():
            for row, batch_size in observations:
                batch_support[split].add((self._row_key(row), batch_size))
        self.validated_batch_keys = (
            batch_support["training"] & batch_support["validation"]
        )
        for row, _ in rows["training"]:
            self.distributions[self._row_key(row)][row["commit_len"]] += 1
        heldout = defaultdict(Counter)
        for row, _ in rows["validation"]:
            heldout[self._row_key(row)][row["commit_len"]] += 1
        self.validation = []
        self.validated_keys = set(heldout) & set(self.distributions)
        for key in sorted(set(heldout) | set(self.distributions)):
            train = self.distributions.get(key, Counter())
            val = heldout.get(key, Counter())
            nt, nv = sum(train.values()), sum(val.values())
            train_mean = sum(k * n for k, n in train.items()) / nt if nt else None
            val_mean = sum(k * n for k, n in val.items()) / nv if nv else None
            self.validation.append(
                {
                    "cell": key,
                    "training_count": nt,
                    "validation_count": nv,
                    "training_batch_sizes": sorted(
                        size for cell, size in batch_support["training"] if cell == key
                    ),
                    "validation_batch_sizes": sorted(
                        size
                        for cell, size in batch_support["validation"]
                        if cell == key
                    ),
                    "supported_batch_sizes": sorted(
                        size for cell, size in self.validated_batch_keys if cell == key
                    ),
                    "training_mean": train_mean,
                    "validation_mean": val_mean,
                    "mean_error": train_mean - val_mean if nt and nv else None,
                    "total_variation": sum(
                        abs(train[k] / nt - val[k] / nv) for k in range(1, 7)
                    )
                    / 2
                    if nt and nv
                    else None,
                }
            )

    @staticmethod
    def _validate_row(row):
        from infercast.sdk.native_sampling import SAMPLING_FIELDS

        fields = {
            "native_key",
            "forward_ct",
            "rid",
            "trajectory_id",
            "prefix_len",
            "draft_history_tokens",
            "sampling_params",
            "verify_width",
            "commit_len",
        }
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or type(row["forward_ct"]) is not int
            or row["forward_ct"] < 0
            or any(
                type(row[name]) is not str or not row[name]
                for name in ("native_key", "trajectory_id", "rid")
            )
        ):
            _error("progress observation identity/fields differ")
        if any(
            type(row[name]) is not int or row[name] < 0
            for name in ("prefix_len", "draft_history_tokens")
        ) or row["draft_history_tokens"] > min(128, row["prefix_len"]):
            _error("progress observation requires causal prefix/history")
        if (
            type(row["verify_width"]) is not int
            or row["verify_width"] != 6
            or type(row["commit_len"]) is not int
            or not 1 <= row["commit_len"] <= 6
        ):
            _error("progress observation requires source Q6 bounded commits")
        recipe = row["sampling_params"]
        if (
            not isinstance(recipe, dict)
            or set(recipe) != SAMPLING_FIELDS
            or recipe["is_normalized"] is not True
        ):
            _error("progress observations require the complete normalized recipe")

    def _key(self, prefix, history, recipe):
        return (
            bisect.bisect_right(self.prefix_edges, prefix),
            bisect.bisect_right(self.history_edges, history),
            hashlib.sha256(_canonical(recipe)).hexdigest(),
        )

    def _row_key(self, row):
        return self._key(
            row["prefix_len"], row["draft_history_tokens"], row["sampling_params"]
        )

    def conditional_commit_counts(self, step):
        """Return immutable training frequencies for causal phase preparation.

        This is a requestwise native-commit distribution, not a sampled outcome
        or a joint batch law. A source-owned cost preparer may integrate a
        measured outcome-dependent cost only with its separately justified
        composition; independence across requests is not implied. The same
        runtime/topology/recipe/heldout-support gates apply as to prediction.
        """
        if (
            hashlib.sha256(step.runtime_profile_json.encode()).hexdigest()
            != self.profile_digest
        ):
            raise PredictorError(
                "data_unavailable", "progress model has no matching runtime profile"
            )
        conditioned = []
        if (
            step.topology.tp_size,
            step.topology.attention_dp_size,
            step.topology.pp_size,
        ) != self.topology:
            raise PredictorError(
                "data_unavailable", "progress originals do not support this topology"
            )
        for group in step.groups:
            batch_size = len(group.requests)
            for request in group.requests:
                if request.extend_len != 6:
                    raise PredictorError(
                        "data_unavailable",
                        "progress model has no matching verify width",
                    )
                key = self._key(
                    request.target_prefix_len,
                    request.bounded_draft_history_tokens(128),
                    json.loads(request.sampling_params_json),
                )
                if key not in self.validated_keys:
                    raise PredictorError(
                        "data_unavailable",
                        "progress condition has no independent training/heldout support",
                        cell=key,
                    )
                if (key, batch_size) not in self.validated_batch_keys:
                    raise PredictorError(
                        "data_unavailable",
                        "progress local batch has no independent training/heldout support",
                        cell=key,
                        batch_size=batch_size,
                    )
                conditioned.append(
                    (request.rid, tuple(sorted(self.distributions[key].items())))
                )
        return tuple(conditioned)

    def predict_progress(self, step):
        conditioned = self.conditional_commit_counts(step)
        predictions = []
        for request, (_, counts) in zip(step.requests, conditioned, strict=True):
            identity = _canonical(
                {
                    "model": self.model_ref,
                    "seed": self.seed,
                    "rid": request.rid,
                    "prefix": request.target_prefix_len,
                    "output_before": request.output_tokens_before,
                }
            )
            # Integer selection is deterministic and has no mutable RNG/retry
            # state. The choice uses no outcome from the current forward.
            choice = int.from_bytes(hashlib.sha256(identity).digest(), "big") % sum(
                frequency for _, frequency in counts
            )
            selected = None
            for count, frequency in counts:
                choice -= frequency
                if choice < 0:
                    selected = count
                    break
            # The retained labels are native commit_lens, not visible output
            # counts. Native Q6 can cross max_new_tokens; SGLang's scheduler
            # updates KV with the full commit and owns the finish boundary.
            predictions.append((request.rid, selected))
        return DSparkProgress(
            step.forward_id, tuple(predictions), self.model_ref
        ).validate_for(step)


def verify_native_progress_rows(document, read_evidence):
    """Check offline commit labels against complete native participant bytes.

    Timing qualification is separate. Current acceptance is read only while
    loading an immutable offline model; live prediction receives before-work.
    Causal-history derivation retains its own review/evidence contract.
    """
    identities, _ = _verify_native_progress_context(document, read_evidence)
    return identities


def _verify_native_progress_context(document, read_evidence):
    """Return original row identities and local batch sizes from one SDK join."""
    from infercast.sdk.native_sampling import native_sampling_params
    from infercast.sdk.native_windows import join_native_frames

    rows = document["observations"]
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(document["native"], list)
        or not document["native"]
    ):
        _error("progress requires selected rows and complete original native windows")
    for row in rows:
        EmpiricalDSparkProgress._validate_row(row)
    wanted = {row["forward_ct"] for row in rows}
    windows = []
    for ref in document["native"]:
        window = read_evidence(ref)
        windows.append(
            {
                **window,
                "records": [
                    r
                    for r in window.get("records", [])
                    if r.get("forward_ct") in wanted
                ],
                "metadata": [
                    r
                    for r in window.get("metadata", [])
                    if r.get("forward_ct") in wanted
                ],
            }
        )
    try:
        frames = join_native_frames(
            windows,
            participants=document["participants"],
            phase_policy=document["phase_policy"],
            source_pin=document["source_pin"],
        )
    except (ValueError, KeyError, TypeError) as error:
        raise PredictorError(
            "invalid_progress_artifact",
            "native progress source/participant join failed",
        ) from error
    byfid = {frame["forward_ct"]: frame for frame in frames}
    identities, batch_sizes, selected = {}, {}, set()
    for row in rows:
        key = row["forward_ct"], row["rid"]
        if key in selected or key[0] not in byfid:
            _error("duplicate or missing selected native progress request")
        selected.add(key)
        matches = 0
        leaders = 0
        local_batch_sizes = set()
        origin_parts = []
        for participant in byfid[key[0]]["participants"]:
            meta, record = participant["metadata"], participant["native_step"]
            if row["rid"] not in meta["request_ids"]:
                continue
            if meta["mode"] != "DECODE" or record is None:
                _error("progress observations require original decode commits")
            index = meta["request_ids"].index(row["rid"])
            recipe = native_sampling_params(meta)[index]
            if _canonical(recipe) != _canonical(row["sampling_params"]):
                _error("progress before-recipe differs from native peer original")
            matches += 1
            local_batch_sizes.add(len(meta["request_ids"]))
            origin_parts.append(
                {
                    "metadata_sha256": hashlib.sha256(_canonical(meta)).hexdigest(),
                    "record_sha256": hashlib.sha256(_canonical(record)).hexdigest(),
                }
            )
            # The native observer publishes per-request acceptance only from
            # the attention-TP leader. Peers retain null/empty reqs; the shared
            # SDK join already enforces that source contract.
            if participant["topology"]["attn_tp_rank"] != 0:
                continue
            leaders += 1
            reqs = record.get("reqs", [])
            if len(reqs) != len(meta["request_ids"]):
                _error("native progress request count differs")
            observed = reqs[index]
            if (
                observed.get("rid") != row["rid"]
                or observed.get("prefix_len") != row["prefix_len"]
                or observed.get("verify_len") != row["verify_width"]
                or type(observed.get("acc_len")) is not int
                or observed["acc_len"] != row["commit_len"]
                or _canonical(recipe) != _canonical(row["sampling_params"])
            ):
                _error(
                    "progress training label or before-input differs from native original"
                )
        if matches != document["participants"][0]["attn_tp_size"] or leaders != 1:
            _error("native progress request omits an attention-TP peer")
        if len(local_batch_sizes) != 1:
            _error("native progress peers disagree on local batch size")
        if row["native_key"] in identities:
            _error("duplicate native progress observation label")
        identities[row["native_key"]] = hashlib.sha256(
            _canonical({"participants": origin_parts, "rid": row["rid"]})
        ).hexdigest()
        batch_sizes[row["native_key"]] = local_batch_sizes.pop()
    return identities, batch_sizes
