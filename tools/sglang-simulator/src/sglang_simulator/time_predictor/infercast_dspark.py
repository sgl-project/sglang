"""Explicit DSPark assembly with measured or opt-in hybrid phase costs."""

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from sglang_simulator.time_predictor.base import InferTimePredictor, PredictorError


def _require(condition, message, *, code="provider_initialization_failed", **details):
    if not condition:
        raise PredictorError(code, message, **details)


def _asset(descriptor, name):
    _require(
        type(descriptor) is dict and set(descriptor) == {"path", "sha256"},
        f"{name} requires exactly path and sha256",
    )
    path, digest = descriptor["path"], descriptor["sha256"]
    _require(type(path) is str and bool(path.strip()), f"{name} path must be explicit")
    _require(
        type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
        f"{name} sha256 must be a full lowercase digest",
    )
    try:
        source = Path(path)
        _require(
            not source.is_symlink()
            and source.is_file()
            and source.stat().st_size <= 64 * 1024**2,
            f"{name} must be a bounded regular file without a symlink",
        )
        with source.open("rb") as stream:
            raw = stream.read(64 * 1024**2 + 1)
        _require(len(raw) <= 64 * 1024**2, f"{name} exceeds its byte bound")
        _require(
            hashlib.sha256(raw).hexdigest() == digest,
            f"{name} bytes differ from sha256",
        )
        return source.resolve(strict=True), raw, digest
    except OSError as error:
        raise PredictorError(
            "data_unavailable", f"{name} is unavailable: {error}"
        ) from error


def _runtime_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate runtime JSON field")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=unique)
        _require(
            type(value) is dict and bool(value),
            "runtime profile must be a nonempty object",
            code="invalid_execution_profile",
        )
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _require(
            raw == canonical,
            "runtime profile JSON must be canonical",
            code="invalid_execution_profile",
        )
        return text
    except (UnicodeError, ValueError, TypeError) as error:
        raise PredictorError(
            "invalid_execution_profile", f"runtime profile JSON is invalid: {error}"
        ) from error


def _native_joint_runtime(
    database,
    *,
    source_policy,
    runtime_profile_json,
    profiles,
    analytic_models,
    model_config_sha256,
    model_seed,
    output_token_id=1,
    cost_evidence_path=None,
):
    """Assemble the typed joint library route under the existing predictor owner.

    The caller retains the independent runtime/capture/allocated-source custody.
    A progress policy attests the local law and mirror observations, not that
    physical context. Public native factory/clock activation remains guarded.
    """
    from sglang_simulator.simulation.sglang.dspark import DSparkSession, DSparkTopology
    from sglang_simulator.simulation.sglang.dspark_joint_execution import (
        NativeJointExecutionResolver,
    )
    from sglang_simulator.simulation.sglang.dspark_runtime import DSparkNativeRuntime
    from sglang_simulator.time_predictor.dspark_golden_progress import (
        SourceGoldenDPProgress,
    )
    from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel

    # All preconditions precede any cost-ledger creation. Never default a
    # missing physical mirror observation to the CPU engine's backend choice.
    _require(
        type(output_token_id) is int and output_token_id >= 0,
        "joint output_token_id must be a nonnegative integer",
    )
    _require(
        type(getattr(source_policy, "reference_needs_cpu_seq_lens", None)) is bool,
        "joint runtime requires original reference CPU mirror observations",
        code="data_unavailable",
        missing_source_facts=["reference_backend_cpu_mirror_policy"],
    )
    _require(
        type(model_config_sha256) is str
        and re.fullmatch("[0-9a-f]{64}", model_config_sha256) is not None
        and type(analytic_models) is dict
        and set(analytic_models) == {"EXTEND", "DECODE"},
        "joint runtime requires both explicit canonical-model analytic assets",
    )
    from infercast.sdk.native_segment_analytic import NativeSegmentAnalyticModel

    _require(
        all(
            type(model) is NativeSegmentAnalyticModel
            and model.model_config_sha256 == model_config_sha256
            for model in analytic_models.values()
        ),
        "joint analytic assets differ from the canonical model",
        code="incompatible_runtime",
    )
    resolver = NativeJointExecutionResolver(
        source_policy=source_policy,
        runtime_profile_json=runtime_profile_json,
        profiles=profiles,
    )
    progress = SourceGoldenDPProgress(
        source_policy,
        runtime_profile_json=runtime_profile_json,
        model_seed=model_seed,
    )
    costs = UMDDSparkCostModel.for_native_joint(
        database,
        source_policy=source_policy,
        runtime_profile_json=runtime_profile_json,
        profiles=profiles,
        analytic_models=analytic_models,
        cost_evidence_path=cost_evidence_path,
    )
    return DSparkNativeRuntime(
        session=DSparkSession(costs, progress),
        topology=DSparkTopology(2, 2),
        runtime_profile_json=runtime_profile_json,
        output_token_id=output_token_id,
        reference_events=True,
        joint_execution_resolver=resolver,
    )


def _native_joint_source_from_binding(binding, runtime_profile_json, model_revision):
    """Rebuild physical coordinates from bounded original source material.

    These receipts establish source/resource facts, not timing qualification.
    Supplemental FILEs and configured environment defaults remain conditional
    semantics; they do not attest that every module was loaded in the process.
    """
    from infercast._serialization import canonical_json_bytes
    from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
    from infercast.sdk.native_segment_analytic import _JOINT_SOURCES
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522
    from infercast.sdk.native_windows import MAX_NATIVE_WINDOW_BYTES
    from infercast.sdk.perf_database.dsv41_dspark import (
        NATIVE_RUNTIME_ENVIRONMENT,
        decode_dspark_artifact,
        decode_dspark_json,
        validate_allocated_pools,
        validate_native_startup,
    )
    from infercast.sdk.perf_database.dsv41_native_forward import (
        parse_forward_profile,
        validate_native_graph_captures,
    )
    from infercast.sdk.perf_database.dsv41_native_runtime import (
        forward_environment,
        select_forward_runtime,
    )
    from infercast.sdk.perf_database.slice_manifest import _read_regular_metadata_file

    fields = {
        "root",
        "source_runtime",
        "startup",
        "proposal",
        "windows",
        "captures",
        "allocated",
        "profiles",
        "container",
        "supplemental_files",
    }
    _require(
        type(binding) is dict and set(binding) == fields,
        "native_joint_source_binding requires its exact original asset fields",
    )
    _require(
        type(binding["root"]) is str and bool(binding["root"].strip()),
        "joint source root must be explicit",
    )
    root = Path(binding["root"])
    _require(
        root.is_absolute()
        and root.is_dir()
        and not root.is_symlink()
        and root.resolve(strict=True) == root,
        "joint source root must be a resolved directory without links",
    )

    def read(ref):
        _require(
            type(ref) is dict
            and set(ref) == {"path", "sha256"}
            and type(ref["path"]) is str
            and type(ref["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is not None,
            "joint source requires exact standard path/sha256 descriptors",
        )
        raw = _read_regular_metadata_file(root, ref["path"], MAX_NATIVE_WINDOW_BYTES)
        _require(
            hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "joint source original bytes differ from descriptor",
            code="invalid_execution_profile",
        )
        return raw

    def document(ref):
        return decode_dspark_artifact(read(ref))

    originals = {}
    for name, count in (
        ("startup", 4),
        ("proposal", 2),
        ("windows", 2),
        ("captures", 4),
        ("allocated", 2),
    ):
        refs = binding[name]
        _require(
            type(refs) is list and len(refs) == count,
            "joint source omits original " + name,
        )
        originals[name] = [document(ref) for ref in refs]
        _require(
            len({ref["path"] for ref in refs}) == count,
            "joint source repeats original " + name,
        )
    startup, proposal, windows = (
        originals[n] for n in ("startup", "proposal", "windows")
    )
    captures, allocated = originals["captures"], originals["allocated"]
    raw_runtime = document(binding["source_runtime"])
    selected = select_forward_runtime(raw_runtime)
    _require(
        selected.source == NATIVE_SOURCE_0522,
        "joint source requires the supported native realization",
    )
    _require(
        type(binding["profiles"]) is dict
        and set(binding["profiles"]) == {"EXTEND", "DECODE"},
        "joint source requires distinct EXTEND and DECODE profile descriptors",
    )
    profiles = {
        mode: parse_forward_profile(document(ref))
        for mode, ref in binding["profiles"].items()
    }
    first = startup[0]
    for key, original in (
        ("execution", "exec_runtime"),
        ("server_args", "server_args"),
        ("runtime_environment", "runtime_environment"),
        ("resolved_model_config", "resolved_model_config"),
        ("model_metadata_files", "model_metadata_files"),
    ):
        _require(
            raw_runtime[key] == first[original],
            "original runtime differs from its startup " + original,
            code="incompatible_runtime",
        )
    _require(
        first["sglang_version"] == raw_runtime["framework_runtime_version"],
        "joint source framework version differs",
        code="incompatible_runtime",
    )
    environment = first["runtime_environment"]
    startup_environment = {
        key: environment[key]
        for key in forward_environment(NATIVE_RUNTIME_ENVIRONMENT, selected)
    }
    for profile in profiles.values():
        validate_native_startup(
            startup,
            profile,
            runtime_version=raw_runtime["framework_runtime_version"],
            model_snapshot_revision=model_revision,
            tp_size=2,
            attention_dp_size=2,
            runtime_environment=startup_environment,
            source_pin=selected.source.name,
        )
    validate_allocated_pools(
        allocated, tp_size=2, attention_dp_size=2, source_pin=selected.source.name
    )
    validate_native_graph_captures(
        captures,
        participants=[row["topology"] for row in proposal],
        profile=profiles["DECODE"],
        startup=startup,
        windows=windows,
        source_pin=selected.source.name,
    )
    policy = NativeDPProgressPolicy(
        startup=tuple(startup), proposal=tuple(proposal), windows=tuple(windows)
    )
    _require(
        type(policy.reference_needs_cpu_seq_lens) is bool,
        "joint source omits original reference mirror observations",
        code="data_unavailable",
    )

    roles, modules = [], {}
    by_rank = {row["topology"]["tp_rank"]: row for row in proposal}
    for start in startup:
        rank, draft = start["parallel"]["tp_rank"], start["is_draft_worker"]
        _require(
            all(
                start[k] == first[k]
                for k in (
                    "server_args",
                    "exec_runtime",
                    "runtime_environment",
                    "actual_model_dtype",
                )
            )
            and start["resolved_model_config"]["quantization_config"]
            == first["resolved_model_config"]["quantization_config"],
            "joint startup execution/model coordinates differ",
            code="incompatible_runtime",
        )
        allocation = next(
            row for row in allocated if row["topology"]["tp_rank"] == rank
        )
        allocated_ref = binding["allocated"][allocated.index(allocation)]
        owner = by_rank[rank]
        _require(
            owner["allocated_runtime_artifact"] == Path(allocated_ref["path"]).name
            and max(row["time_ns"] for row in startup if row["rank"] == rank)
            <= allocation["time_ns"]
            <= owner["time_ns"],
            "joint allocated pool does not belong to its proposal epoch",
        )
        runner = next(
            row for row in allocation["runners"] if row["is_draft_worker"] is draft
        )
        capture = next(
            row
            for row in captures
            if row["pid"] == start["pid"] and row["is_draft_worker"] is draft
        )
        roles.append(
            dict(
                role="draft" if draft else "target",
                parallel={
                    key: value
                    for key, value in start["parallel"].items()
                    if key not in ("launch_world_rank", "launch_world_size")
                },
                monokernel_installed=runner["monokernel_route"]["installed"],
                graph={
                    key: capture[key]
                    for key in (
                        "captured_req_width",
                        "request_pool_size",
                        "padding_policy",
                        "graph_keys",
                        "capture_batches",
                    )
                },
            )
        )
        for module, row in start["loaded_realization_modules"].items():
            if module in _JOINT_SOURCES:
                digest = row["sha256"]
                _require(
                    module not in modules or modules[module] == {"sha256": digest},
                    "joint startup module identities differ",
                )
                modules[module] = {"sha256": digest}
    for row in proposal:
        for module, source in (
            (
                "sglang.srt.environ",
                row["facts"]["golden_acceptance"]["sources"]["environment"],
            ),
            (
                "sglang.srt.speculative.dspark_components.dspark_worker_v2",
                row["facts"]["sources"]["worker"],
            ),
        ):
            value = {"sha256": source["sha256"]}
            _require(
                module not in modules or modules[module] == value,
                "joint proposal module identities differ",
            )
            modules[module] = value
    missing = set(_JOINT_SOURCES) - modules.keys()
    _require(
        type(binding["supplemental_files"]) is dict
        and set(binding["supplemental_files"]) == missing,
        "joint source requires exactly its missing conditional FILE originals",
    )
    for module, ref in binding["supplemental_files"].items():
        modules[module] = {"sha256": hashlib.sha256(read(ref)).hexdigest()}
    _require(
        modules
        == {name: {"sha256": digest} for name, digest in _JOINT_SOURCES.items()},
        "joint conditional source FILE identities differ",
        code="incompatible_runtime",
    )

    inspect = decode_dspark_json(read(binding["container"]))
    _require(
        type(inspect) is list
        and len(inspect) == 1
        and inspect[0]["Config"]["Image"] == selected.image,
        "joint container image differs from its native source",
    )
    configured = {}
    for item in inspect[0]["Config"]["Env"]:
        key, value = item.split("=", 1)
        _require(key not in configured, "joint container environment repeats a name")
        configured[key] = value
    env = dict(environment)
    defaults = []
    for name in (
        "SGLANG_USE_ROCM700A",
        "SGLANG_AITER_MEGA_RANK_SYNC",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_DP_USE_REDUCE_SCATTER",
        "SGLANG_SHARED_EXPERT_TP1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL",
        "SGLANG_DP_USE_GATHERV",
        "SGLANG_ENABLE_DP_GATHER_FP8",
    ):
        if name in env:
            _require(
                name not in configured or configured[name] == env[name],
                "joint configured environment differs from original startup",
            )
        else:
            env[name] = configured.get(name)
            defaults.append(name)
    sections = {
        key: copy.deepcopy(first["exec_runtime"][key])
        for key in ("kernel", "comm", "moe", "offload", "overlap", "graph", "features")
    }
    for mode in ("decode", "prefill"):
        sections["graph"]["cuda_graph_backend_" + mode] = first["server_args"][
            "cuda_graph_config"
        ][mode]["backend"]
    physical = dict(
        schema_version=1,
        software=dict(sglang=first["sglang_version"], modules=modules),
        model_realization=dict(
            model_dtype=first["actual_model_dtype"],
            quantization_config=first["resolved_model_config"]["quantization_config"],
        ),
        roles=sorted(roles, key=lambda row: (row["role"], row["parallel"]["tp_rank"])),
        execution=dict(sections=sections, environment=env),
    )
    _require(
        canonical_json_bytes(physical).decode() == runtime_profile_json,
        "configured joint runtime differs from original physical source",
        code="incompatible_runtime",
    )
    return (
        policy,
        profiles,
        {
            **copy.deepcopy(binding),
            "root": str(root),
            "conditional_FILE_modules": sorted(binding["supplemental_files"]),
            "container_configured_environment_coordinates": defaults,
            "source_identity_sha256": policy.source_identity_sha256,
            "timing_qualified": False,
        },
    )


def _native_source_binding_assets(binding):
    """Use the SDK's existing bounded original reader, not numerical metadata."""
    from infercast.sdk.native_windows import MAX_NATIVE_WINDOW_BYTES
    from infercast.sdk.perf_database.dsv41_native_segment import artifact_descriptor
    from infercast.sdk.perf_database.slice_manifest import _read_regular_metadata_file

    _require(
        type(binding) is dict
        and set(binding) == {"root", "profile", "cohort", "stack"},
        "native_source_binding requires exactly root, profile, cohort and stack",
    )
    _require(
        type(binding["root"]) is str and bool(binding["root"].strip()),
        "native source evidence root must be explicit",
    )
    root = Path(binding["root"])
    _require(
        root.is_absolute()
        and root.is_dir()
        and not root.is_symlink()
        and root.resolve(strict=True) == root,
        "native source evidence root must be a resolved directory without links",
    )
    refs = {}
    for name in ("profile", "cohort", "stack"):
        _require(
            type(binding[name]) is dict, "native source descriptor must be an object"
        )
        path, digest = artifact_descriptor(binding[name])
        refs[name] = {"path": path, "sha256": digest}

    def read(path, digest):
        artifact_descriptor({"path": path, "sha256": digest})
        raw = _read_regular_metadata_file(root, path, MAX_NATIVE_WINDOW_BYTES)
        _require(
            hashlib.sha256(raw).hexdigest() == digest,
            "native source original bytes differ from their descriptor",
            code="invalid_execution_profile",
        )
        return raw

    return {"root": str(root), **refs}, read


def _native_source_policy_from_binding(database, profile_id, capture_profile, binding):
    """Verify source originals independently of numerical row admission."""
    from infercast._serialization import canonical_json_bytes
    from infercast.sdk.native_segment_projection import NativeSegmentSourcePolicy
    from infercast.sdk.perf_database.dsv41_dspark import decode_dspark_artifact
    from infercast.sdk.perf_database.dsv41_native_segment import (
        artifact_descriptor,
        native_segment_stack_identity,
    )

    identity, read = _native_source_binding_assets(binding)

    def document(ref):
        return decode_dspark_artifact(read(*artifact_descriptor(ref)))

    profile = document(identity["profile"])
    cohort = document(identity["cohort"])
    profile_stack = document(profile["stack"])
    stack = document(identity["stack"])
    _require(
        identity["profile"]["sha256"] == profile_id
        and artifact_descriptor(cohort["stack"])
        == artifact_descriptor(identity["stack"])
        and native_segment_stack_identity(profile_stack)
        == native_segment_stack_identity(stack)
        == native_segment_stack_identity(database.stack_meta),
        "native source profile or stable numerical database stack differs",
        code="incompatible_runtime",
    )
    policy = NativeSegmentSourcePolicy.from_cohort(
        identity["profile"], identity["cohort"], read, stack_meta=stack
    )
    _require(
        canonical_json_bytes(capture_profile.to_dict())
        == canonical_json_bytes(document(cohort["capture_profile"])),
        "configured native capture profile differs from its source cohort",
        code="invalid_execution_profile",
    )
    return policy, identity


def _native_source_policy(database, profile_id, capture_profile):
    """Resolve existing retained slice metadata; the SDK validates its policy."""
    from infercast._serialization import canonical_json_bytes
    from infercast.sdk.native_segment_module import OP, NativeSegmentQuery
    from infercast.sdk.native_segment_projection import NativeSegmentSourcePolicy
    from infercast.sdk.native_segments import BOUNDARY
    from infercast.sdk.native_windows import MAX_NATIVE_WINDOW_BYTES
    from infercast.sdk.perf_database.dsv41_dspark import decode_dspark_artifact
    from infercast.sdk.perf_database.dsv41_native_segment import artifact_descriptor
    from infercast.sdk.perf_database.slice_manifest import read_verified_slice_artifact

    root = database.slice_ref.slice_dir

    def read(path, digest):
        return read_verified_slice_artifact(
            root, path, expected_sha256=digest, max_bytes=MAX_NATIVE_WINDOW_BYTES
        )

    def document(ref):
        return decode_dspark_artifact(read(*artifact_descriptor(ref)))

    inventory = decode_dspark_artifact(
        read_verified_slice_artifact(
            root, "_meta/kernel_identity.json", max_bytes=MAX_NATIVE_WINDOW_BYTES
        )
    )
    _require(
        type(inventory.get("schema_version")) is int
        and inventory["schema_version"] == 1
        and isinstance(inventory.get("identities"), list),
        "native regions require retained kernel identities",
        code="invalid_execution_profile",
    )
    profiles, measurements = {}, {}
    for entry in inventory["identities"]:
        if not isinstance(entry, dict):
            raise TypeError("native kernel identity must be an object")
        if entry.get("op") != OP or entry.get("config_token") != profile_id:
            continue
        _require(
            type(entry.get("shape_key")) is str,
            "native region identity requires its canonical source shape",
            code="invalid_execution_profile",
        )
        query = NativeSegmentQuery.from_shape(
            decode_dspark_artifact(entry["shape_key"].encode("utf-8"))
        )
        shape = query.to_shape()
        extra = entry.get("extra", {})
        _require(
            query.execution_profile_id == profile_id
            and entry["shape_key"] == canonical_json_bytes(shape).decode()
            and entry.get("kernel_source") == "sglang"
            and entry.get("exec_mode") == shape["kernel_impl"]
            and entry.get("kernel_name") == BOUNDARY
            and isinstance(extra, dict)
            and {"execution_profile", "measurement_manifest"} <= extra.keys(),
            "native region source/profile identity differs",
            code="invalid_execution_profile",
        )
        ref, measurement = extra["execution_profile"], extra["measurement_manifest"]
        profiles[artifact_descriptor(ref)] = ref
        measurements[artifact_descriptor(measurement)] = measurement
    _require(
        len(profiles) == 1 and next(iter(profiles))[1] == profile_id,
        "native region profile is missing or ambiguous in the finalized slice",
        code="data_unavailable",
    )
    profile_ref = next(iter(profiles.values()))
    profile = document(profile_ref)
    _require(
        isinstance(profile, dict)
        and type(profile.get("schema_version")) is int
        and profile["schema_version"] in (1, 2),
        "native source profile schema must be explicitly 1 or 2",
        code="invalid_execution_profile",
    )
    if profile["schema_version"] == 2:
        _require(
            callable(getattr(NativeSegmentSourcePolicy, "from_cohort", None)),
            "native profile2 requires SDK cohort validation",
        )
        policy = None
        for ref in measurements.values():
            value = document(ref)
            _require(
                isinstance(value, dict)
                and type(value.get("schema_version")) is int
                and value["schema_version"] == 2
                and value.get("execution_profile_id") == profile_id,
                "native source measurement belongs to another profile or schema",
                code="invalid_execution_profile",
            )
            _require(
                "cohort" in value,
                "native profile2 measurement omits original cohort",
                code="data_unavailable",
            )
            candidate = NativeSegmentSourcePolicy.from_cohort(
                profile_ref, value["cohort"], read, stack_meta=database.stack_meta
            )
            cohort = document(value["cohort"])
            _require(
                canonical_json_bytes(capture_profile.to_dict())
                == canonical_json_bytes(document(cohort["capture_profile"])),
                "configured native capture profile differs from its source cohort",
                code="invalid_execution_profile",
            )
            for name in ("native", "proposal_paths"):
                refs = value.get(name)
                _require(
                    isinstance(refs, list)
                    and bool(refs)
                    and (name != "proposal_paths" or len(refs) == candidate.tp_size),
                    "native source metadata omits original " + name,
                    code="data_unavailable",
                )
                identities = [artifact_descriptor(original) for original in refs]
                _require(
                    len(set(identities)) == len(identities),
                    "native source metadata repeats original " + name,
                    code="invalid_execution_profile",
                )
                for original in refs:
                    read(*artifact_descriptor(original))
            _require(
                sorted(value["proposal_paths"], key=artifact_descriptor)
                == cohort["proposal_paths"],
                "native source measurement proposal cohort differs",
                code="invalid_execution_profile",
            )
            # Each original epoch must independently establish this entire
            # immutable SDK policy. Never union partial process receipts.
            _require(
                policy is None or candidate == policy,
                "native source cohorts establish inconsistent prospective policies",
                code="invalid_execution_profile",
            )
            policy = candidate
        _require(policy is not None, "native profile2 omits source measurements")
        return policy
    source = document(profile["source"])
    captures, startup = {}, {}
    for ref in source["evidence"]:
        value = document(ref)
        # Select original record kinds, including malformed candidates for the
        # SDK's strict rejection. Roles never come from artifact filenames.
        if any(
            k in value for k in ("graph_keys", "runner_class", "captured_req_width")
        ):
            captures[artifact_descriptor(ref)] = ref
        if "recorded_after_successful_init" in value:
            startup[artifact_descriptor(ref)] = ref
    windows, proposal = {}, {}
    for ref in measurements.values():
        value = document(ref)
        _require(
            type(value.get("schema_version")) is int
            and value["schema_version"] == 1
            and value.get("execution_profile_id") == profile_id,
            "native source measurement belongs to another profile",
            code="invalid_execution_profile",
        )
        for name, target in (("native", windows), ("proposal_paths", proposal)):
            refs = value.get(name)
            _require(
                isinstance(refs, list)
                and bool(refs)
                and (name != "proposal_paths" or len(refs) == 2),
                "native source metadata omits original " + name,
                code="data_unavailable",
            )
            identities = [artifact_descriptor(original) for original in refs]
            _require(
                len(set(identities)) == len(identities),
                "native source metadata repeats original " + name,
                code="invalid_execution_profile",
            )
            for original in refs:
                target[artifact_descriptor(original)] = original
    return NativeSegmentSourcePolicy(
        profile_ref,
        read,
        stack_meta=database.stack_meta,
        capture_profile=capture_profile,
        capture_descriptors=tuple(captures[k] for k in sorted(captures)),
        startup_descriptors=tuple(startup[k] for k in sorted(startup)),
        window_descriptors=tuple(windows[k] for k in sorted(windows)),
        proposal_descriptors=tuple(proposal[k] for k in sorted(proposal)),
    )


def _source_initial_draft_buffer(
    database, policy, capture_profile, descriptor, binding
):
    """Verify an observed warm resource through its original source cohort."""
    from infercast._serialization import canonical_json_bytes
    from infercast.sdk.native_forward_observation import native_forward_inputs
    from infercast.sdk.native_segment_module import (
        NativeSegmentQuery,
        native_draft_read_extent,
    )
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522
    from infercast.sdk.native_verify_plan import verify_plan_cpu_max_seq_len
    from infercast.sdk.native_windows import join_native_frames
    from infercast.sdk.perf_database.dsv41_dspark import decode_dspark_artifact
    from infercast.sdk.perf_database.dsv41_native_segment import (
        _verify_before_buffer,
        artifact_descriptor,
        json_variant,
    )

    candidate, identity = _native_source_policy_from_binding(
        database, policy.execution_profile_id, capture_profile, binding
    )
    _require(candidate == policy, "initial resource source policy differs")
    path, raw, digest = _asset(descriptor, "native_initial_draft_buffer")
    query = NativeSegmentQuery.from_shape(decode_dspark_artifact(raw))
    shape, variant = query.to_shape(), json_variant(query)
    capacity = variant["buffer_capacity"]
    _require(
        query.execution_profile_id == policy.execution_profile_id
        and shape["tp_size"] == policy.tp_size
        and shape["attention_dp_size"] == 1
        and shape["worker_phase"] == "DECODE"
        and shape["segment_kind"] == "decode_preamble"
        and type(capacity) is int
        and capacity >= shape["batch_size"]
        and variant["buffer_device_matches"] is True
        and "draft_read_extent_tokens" in variant
        and capture_profile.decode_graph_backend == "full",
        "initial source resource requires a matching warm native DECODE query",
        code="invalid_execution_profile",
    )
    capture_profile.validate_batch(shape["batch_size"])
    policy._require_capture(True, variant["draft_capture_key"])
    policy._require_capture(False, variant["target_capture_key"])
    _, read = _native_source_binding_assets(identity)
    cohort = decode_dspark_artifact(read(*artifact_descriptor(identity["cohort"])))
    windows = [
        decode_dspark_artifact(read(*artifact_descriptor(ref)))
        for ref in cohort["policy_windows"]
    ]
    participants = [
        {
            "tp_rank": rank,
            "tp_size": policy.tp_size,
            "attn_tp_rank": rank,
            "attn_tp_size": policy.tp_size,
            "attn_dp_rank": 0,
            "attn_dp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
        }
        for rank in range(policy.tp_size)
    ]
    join = {
        "participants": participants,
        "phase_policy": "serialized",
        "source_pin": NATIVE_SOURCE_0522.name,
    }
    frames = join_native_frames(windows, **join)
    native = dict(native_forward_inputs(windows, candidate_span=16384, **join))
    matched = []
    for frame in frames:
        fid = frame["forward_ct"]
        if native[fid] != query.native_input:
            continue
        observed = []
        for peer in frame["participants"]:
            row = peer["metadata"]
            buffer = (
                row.get("native_proposal_call", {}).get("before", {}).get("buffer", {})
            )
            state = buffer.get("state")
            _require(
                state == "none"
                or state == "present"
                and type(buffer.get("shape")) is list
                and len(buffer["shape"]) == 2,
                "initial source resource omits actual buffer state",
                code="invalid_execution_profile",
            )
            actual = dict(variant)
            actual["buffer_capacity"] = 0 if state == "none" else buffer["shape"][0]
            actual["buffer_device_matches"] = state == "present" and buffer.get(
                "device"
            ) == buffer.get("prefix_device")
            _verify_before_buffer(row, actual)
            actual["verify_plan_cpu_max_seq_len"] = verify_plan_cpu_max_seq_len(row)
            actual["draft_read_extent_tokens"] = native_draft_read_extent(native[fid])
            for role, name in (
                (True, "draft_capture_key"),
                (False, "target_capture_key"),
            ):
                keys = [
                    replay["shape_key"]
                    for replay in row["actual_full_graph_replays"]
                    if replay["is_draft_worker"] is role
                ]
                _require(
                    len(keys) == 1,
                    "initial resource requires both actual graph replays",
                )
                actual[name] = keys[0]
            observed.append(canonical_json_bytes(actual))
        if len(observed) == policy.tp_size and all(
            value == canonical_json_bytes(variant) for value in observed
        ):
            matched.append(fid)
    _require(
        bool(matched),
        "initial resource has no complete matching original source forward",
        code="data_unavailable",
    )
    return capacity, {
        "path": str(path),
        "sha256": digest,
        "query_sha256": hashlib.sha256(canonical_json_bytes(shape)).hexdigest(),
        "execution_profile_id": policy.execution_profile_id,
        "capture_profile_id": capture_profile.profile_id,
        "cohort": identity["cohort"],
        "forward_ids": sorted(matched),
        "capacity": capacity,
        "device_matches": True,
        "source": "native_source_cohort",
        "state": "observed_warm_before_work",
        "resolution": "observed_before_work",
        "latency_used": False,
        "physical_CPU_buffer_claimed": False,
        "source_reference_initial_state_equivalence": False,
        "whole_workflow_point": False,
    }


def _initial_draft_buffer(database, policy, capture_profile, descriptor):
    """Verify an explicit warm resource witness through the existing SDK owner.

    The asset is an existing complete NativeSegmentQuery shape, not a latency
    override. Exact silicon validation checks its original before-work buffer
    observations. Its cost value is never used to initialize the runtime.
    """
    if descriptor is None:
        return 0, None
    _require(
        policy is not None and capture_profile is not None,
        "initial draft buffer requires the native source/capture profile",
        code="invalid_execution_profile",
    )
    from infercast._serialization import canonical_json_bytes
    from infercast.sdk.diagnostics.prediction_evidence import (
        capture_prediction_evidence,
        prediction_node,
        record_node,
    )
    from infercast.sdk.native_segment_module import OP, NativeSegmentQuery
    from infercast.sdk.operations.compute.dsv41_native_segment import DSV41NativeSegment
    from infercast.sdk.perf_database.dsv41_dspark import decode_dspark_artifact

    path, raw, digest = _asset(descriptor, "native_initial_draft_buffer")
    query = NativeSegmentQuery.from_shape(decode_dspark_artifact(raw))
    shape = query.to_shape()
    variant = json.loads(query.module_variant)
    capacity = variant["buffer_capacity"]
    _require(
        query.execution_profile_id == policy.execution_profile_id
        and shape["tp_size"] == policy.tp_size
        and shape["attention_dp_size"] == 1
        and shape["worker_phase"] == "DECODE"
        and shape["segment_kind"] == "decode_preamble"
        and type(capacity) is int
        and capacity >= shape["batch_size"]
        and variant["buffer_device_matches"] is True
        and capture_profile.decode_graph_backend == "full"
        and variant["draft_capture_key"] is not None
        and variant["target_capture_key"] is not None,
        "initial draft buffer witness must be a matching warm native DECODE context",
        code="invalid_execution_profile",
    )
    capture_profile.validate_batch(shape["batch_size"])
    node_id = "initial-modeled-draft-buffer-witness"
    with (
        capture_prediction_evidence() as recorder,
        prediction_node(node_id, "initialization"),
    ):
        result = DSV41NativeSegment(node_id, shape=shape).query(database)
        _require(
            result.source == "silicon",
            "initial draft buffer witness requires an exact qualified silicon cell",
            code="data_unavailable",
        )
        record_node(
            node_id=node_id,
            phase="initialization",
            op=OP,
            shape=shape,
            layer_group="initialization",
            repeat=1,
            count=1,
            result=result,
            status="resolved",
        )
    queries = recorder.nodes[0]["queries"]
    _require(
        recorder.dropped_queries == 0
        and len(queries) == 1
        and queries[0]["component"] == OP
        and queries[0]["source"] == "silicon"
        and queries[0]["resolution"] == "exact"
        and canonical_json_bytes(queries[0]["requested_key"])
        == canonical_json_bytes(shape)
        and canonical_json_bytes(queries[0]["resolved_key"])
        == canonical_json_bytes(shape),
        "initial draft buffer witness requires an exact qualified silicon cell",
        code="data_unavailable",
    )
    return capacity, {
        "path": str(path),
        "sha256": digest,
        "query_sha256": hashlib.sha256(canonical_json_bytes(shape)).hexdigest(),
        "execution_profile_id": policy.execution_profile_id,
        "capture_profile_id": capture_profile.profile_id,
        "capacity": capacity,
        "device_matches": True,
        "source": "silicon",
        "state": "observed_warm_before_work",
        "latency_used": False,
        "physical_CPU_buffer_claimed": False,
        "resolution": "exact",
        "source_reference_initial_state_equivalence": False,
        "whole_workflow_point": False,
    }


def _canonical_config(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_metadata(value):
    return {
        name: getattr(value, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }


def _local_config_bytes(model_path, model_config_root):
    """Read a bounded config within an explicit root and verify read stability."""
    model_dir = Path(model_path).absolute()
    root = Path(model_config_root).absolute()
    source = model_dir / "config.json"

    def snapshot():
        anchor = root.resolve(strict=True)
        model = model_dir.resolve(strict=True)
        anchor_stat, model_stat = anchor.stat(), model.stat()
        _require(
            stat.S_ISDIR(anchor_stat.st_mode) and stat.S_ISDIR(model_stat.st_mode),
            "model_config_root and local model path must be existing directories",
            code="incompatible_runtime",
        )
        _require(
            model.is_relative_to(anchor),
            "local model directory escapes model_config_root",
            code="incompatible_runtime",
        )
        target = source.resolve(strict=True)
        _require(
            target.is_relative_to(anchor),
            "resolved local model config escapes model_config_root",
            code="incompatible_runtime",
        )
        source_stat, target_stat = source.lstat(), target.lstat()
        _require(
            stat.S_ISREG(target_stat.st_mode) and target_stat.st_size <= 64 * 1024**2,
            "resolved local model config must be a bounded regular file",
            code="incompatible_runtime",
        )

        # Directory contents may change independently in a shared HF cache.
        # Bind directory identity and link resolution, not directory mtimes.
        def directory_identity(path, resolved_stat):
            value = path.lstat()
            return {
                "requested_identity": [value.st_dev, value.st_ino, value.st_mode],
                "resolved_identity": [
                    resolved_stat.st_dev,
                    resolved_stat.st_ino,
                    resolved_stat.st_mode,
                ],
                "link_target": str(path.readlink())
                if stat.S_ISLNK(value.st_mode)
                else None,
            }

        return {
            "path": str(source),
            "resolved_path": str(target),
            "requested_root": str(root),
            "root": str(anchor),
            "resolved_model_dir": str(model),
            "root_identity": directory_identity(root, anchor_stat),
            "model_dir_identity": directory_identity(model_dir, model_stat),
            "source_stat": _file_metadata(source_stat),
            "target_stat": _file_metadata(target_stat),
            "link_target": str(source.readlink())
            if stat.S_ISLNK(source_stat.st_mode)
            else None,
        }

    before = snapshot()
    with Path(before["resolved_path"]).open("rb") as stream:
        opened = _file_metadata(os.fstat(stream.fileno()))
        _require(
            # Windows path stat and descriptor fstat can expose different
            # ctime meanings. Compare ctime within each API below; cross-check
            # file identity, mode, size and mtime across the opened descriptor.
            all(
                opened[name] == before["target_stat"][name]
                for name in opened
                if name != "st_ctime_ns"
            ),
            "local model config target changed before read",
            code="incompatible_runtime",
        )
        raw = stream.read(64 * 1024**2 + 1)
        _require(
            len(raw) <= 64 * 1024**2
            and len(raw) == opened["st_size"]
            and _file_metadata(os.fstat(stream.fileno())) == opened,
            "local model config changed during bounded read",
            code="incompatible_runtime",
        )
    _require(
        snapshot() == before,
        "local model config path or target changed during read",
        code="incompatible_runtime",
    )
    return raw, {**before, "read_stat": opened}


def _bind_local_model_config(model, authoritative, model_config_root=None):
    """Bind local checkpoint bytes and native HF text fields to SDK source JSON."""
    path = getattr(model, "model_path", None)
    _require(
        type(path) is str and bool(path.strip()),
        "DSPark requires an explicit local model.model_path",
        code="incompatible_runtime",
    )
    _require(
        model_config_root is None
        or type(model_config_root) is str
        and bool(model_config_root.strip()),
        "model_config_root must be None or an explicit nonempty path",
        code="incompatible_runtime",
    )
    root = path if model_config_root is None else model_config_root
    try:
        raw, binding = _local_config_bytes(path, root)

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate local model config field")
                result[key] = value
            return result

        actual = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
        _require(
            type(actual) is dict
            and type(authoritative) is dict
            and _canonical_config(actual) == _canonical_config(authoritative),
            "local checkpoint config differs from authoritative SDK source config",
            code="incompatible_runtime",
        )
        expected = authoritative.get("text_config")
        _require(
            type(expected) is dict and bool(expected),
            "authoritative SDK config requires text_config",
            code="incompatible_runtime",
        )
        native = getattr(model, "hf_config", None)
        if not isinstance(native, dict) and callable(getattr(native, "to_dict", None)):
            native = native.to_dict()
        _require(
            type(native) is dict,
            "native HF text config must be a dict or expose to_dict()",
            code="incompatible_runtime",
        )
        # Delegate the source-authored text/root/vision mapping to the same
        # native adapter that loads the checkpoint. Do not copy its field map
        # or fill missing actual fields with source defaults.
        from sglang.srt.configs.deepseek_v41 import (
            DeepseekV41Config,
            normalize_deepseek_v41_config,
        )
        from sglang.srt.utils.hf_transformers_patches import (
            normalize_rope_scaling_compat,
        )

        expected = normalize_deepseek_v41_config(copy.deepcopy(authoritative))
        rope_fields = {"rope_scaling", "rope_parameters", "rope_theta"}
        # HF writes its installed transformers_version on serialization. This
        # provenance field does not describe model work; the local file above
        # still binds the original authored value.
        structural = set(expected) - {
            "dtype",
            "torch_dtype",
            "transformers_version",
            *rope_fields,
        }
        missing = sorted(structural - native.keys())
        _require(
            not missing,
            "native HF config omits authoritative structural keys",
            code="incompatible_runtime",
            fields=missing,
        )
        _require(
            _canonical_config({key: native[key] for key in structural})
            == _canonical_config({key: expected[key] for key in structural}),
            "native HF structural config differs from SDK source",
            code="incompatible_runtime",
        )
        if rope_fields & expected.keys():
            # Transformers versions can serialize rope_scaling/rope_theta as
            # rope_parameters. Ask the source-owned native class for that exact
            # representation, including the runtime's legacy "type" alias.
            # Never normalize or repair the actual loaded config here.
            reference = DeepseekV41Config(**copy.deepcopy(authoritative))
            normalize_rope_scaling_compat(reference)
            reference = reference.to_dict()
            _require(
                type(reference) is dict and bool(rope_fields & reference.keys()),
                "native reference cannot represent authoritative RoPE config",
                code="incompatible_runtime",
            )
            _require(
                _canonical_config(
                    {key: native[key] for key in rope_fields & native.keys()}
                )
                == _canonical_config(
                    {key: reference[key] for key in rope_fields & reference.keys()}
                ),
                "native HF RoPE config differs from SDK source conversion",
                code="incompatible_runtime",
            )
        expected_dtype = expected.get(
            "dtype",
            expected.get(
                "torch_dtype",
                authoritative.get("dtype", authoritative.get("torch_dtype")),
            ),
        )
        if expected_dtype is not None:
            # HF can serialize torch.dtype as 'bfloat16' and older versions use
            # torch_dtype instead of dtype. No numeric or structural coercion.
            def dtype_name(value):
                return str(value).removeprefix("torch.")

            dtypes = [
                native[name] for name in ("dtype", "torch_dtype") if name in native
            ]
            _require(
                bool(dtypes)
                and all(
                    dtype_name(value) == dtype_name(expected_dtype) for value in dtypes
                ),
                "native HF dtype differs from SDK source",
                code="incompatible_runtime",
            )
        quant = authoritative.get("quantization_config")
        if (
            "expert_dtype" in native
            and "expert_dtype" not in expected
            and isinstance(quant, dict)
        ):
            _require(
                native["expert_dtype"] == quant.get("expert_dtype"),
                "native HF expert dtype differs from SDK source",
                code="incompatible_runtime",
            )
        final_raw, final_binding = _local_config_bytes(path, root)
        _require(
            final_binding == binding and final_raw == raw,
            "local model config path or target changed during native binding",
            code="incompatible_runtime",
        )
        return {
            **binding,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "stable_binding": True,
            "sdk_canonical_sha256": hashlib.sha256(
                _canonical_config(authoritative)
            ).hexdigest(),
        }
    except PredictorError:
        raise
    except (
        ImportError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        raise PredictorError(
            "incompatible_runtime", f"local/native model config is unsupported: {error}"
        ) from error


def _geometry(model, description, model_id):
    geometry = getattr(description, "geometry", None)
    _require(
        isinstance(geometry, Mapping)
        and getattr(description, "model", None) == model_id
        and getattr(description, "backend", None) == "sglang",
        "SDK model description identity or geometry differs",
        code="incompatible_runtime",
    )
    names = {
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "head_dim": "head_dim",
        "vocab_size": "vocab_size",
        "qk_rope_head_dim": "qk_rope_head_dim",
        "indexer_head_dim": "index_head_dim",
        "window_size": "sliding_window",
    }
    mismatches = {}
    for field, key in names.items():
        actual, expected = getattr(model, field, None), geometry.get(key)
        if type(actual) is not int or type(expected) is not int or actual != expected:
            mismatches[field] = {"sglang": actual, "infercast": expected}
    ratios = getattr(model, "compression_ratios", None)
    expected_ratios = geometry.get("compression_ratios_source")
    if (
        not isinstance(ratios, (tuple, list))
        or not isinstance(expected_ratios, (tuple, list))
        or any(type(value) is not int for value in (*ratios, *expected_ratios))
        or tuple(ratios) != tuple(expected_ratios)
    ):
        mismatches["compression_ratios"] = {
            "sglang": ratios,
            "infercast": expected_ratios,
        }
    _require(
        not mismatches,
        "SDK geometry differs from the loaded SGLang model",
        code="incompatible_runtime",
        mismatches=mismatches,
    )


class InferCastDSparkTimePredictor(InferTimePredictor):
    """Assemble native before-work/progress and the existing measured UMD path.

    Construction establishes identity, not complete costs or accuracy. No
    configured callback, phase constant or ordinary-forward fallback is accepted.
    Synthetic output content does not broaden the native cohort's stop/EOS semantics.
    """

    supports_dspark = True

    def __init__(
        self,
        model,
        hw,
        config,
        *,
        model_id=None,
        model_revision=None,
        provider_revision=None,
        system=None,
        systems_root=None,
        runtime_profile=None,
        progress_artifact=None,
        progress_kind="empirical",
        model_seed=None,
        proposal_profile_id=None,
        native_segment_profile_id=None,
        native_capture_profile=None,
        native_initial_draft_buffer=None,
        native_source_work_policy=None,
        cost_mode="SILICON",
        native_analytic_model=None,
        native_cost_evidence_path=None,
        native_source_binding=None,
        native_joint_source_binding=None,
        output_token_id=1,
        model_config_root=None,
        **kwargs,
    ):
        _require(
            not kwargs, "unknown DSPark predictor configuration", fields=sorted(kwargs)
        )
        _require(
            type(cost_mode) is str and cost_mode in {"SILICON", "HYBRID"},
            "cost_mode must be SILICON or HYBRID",
        )
        _require(
            (cost_mode == "HYBRID") == (native_analytic_model is not None),
            "HYBRID requires an explicit native_analytic_model asset",
        )
        _require(
            native_cost_evidence_path is None or cost_mode == "HYBRID",
            "native_cost_evidence_path requires explicit HYBRID costs",
        )
        _require(
            native_source_binding is None
            or cost_mode == "HYBRID"
            and type(native_source_binding) is dict,
            "native_source_binding requires explicit HYBRID analytic costs",
        )
        joint = native_joint_source_binding is not None
        _require(
            not joint
            or cost_mode == "HYBRID"
            and progress_kind == "source_golden351"
            and type(native_joint_source_binding) is dict
            and all(
                value is None
                for value in (
                    native_source_binding,
                    native_segment_profile_id,
                    native_capture_profile,
                    native_initial_draft_buffer,
                    native_source_work_policy,
                    proposal_profile_id,
                    progress_artifact,
                )
            ),
            "native joint source requires explicit HYBRID/source_golden351 without regional selectors",
        )
        _require(
            cost_mode != "HYBRID"
            or (native_segment_profile_id is not None or joint)
            and progress_kind == "source_golden351",
            "HYBRID requires source-bound native golden regions",
        )
        _require(
            type(progress_kind) is str
            and progress_kind in {"empirical", "source_golden351"},
            "progress_kind must be empirical or source_golden351",
        )
        if progress_kind == "empirical":
            _require(model_seed is None, "empirical progress uses its artifact seed")
        else:
            _require(
                type(model_seed) is int and model_seed >= 0,
                "source_golden351 requires an explicit nonnegative integer model_seed",
            )
            _require(
                progress_artifact is None
                and (native_segment_profile_id is not None or joint),
                "source_golden351 requires native regions and no empirical progress_artifact",
            )
        _require(
            native_source_work_policy is None
            or progress_kind == "source_golden351"
            and type(native_source_work_policy) is str,
            "native source work requires explicit source_golden351 progress",
        )
        super().__init__(model, hw, config)
        _require(
            model_id == "deepseek-ai/DeepSeek-V4.1-Flash",
            "DSPark predictor requires deepseek-ai/DeepSeek-V4.1-Flash",
            code="unsupported_model",
        )
        for name, revision in (
            ("model_revision", model_revision),
            ("provider_revision", provider_revision),
        ):
            _require(
                type(revision) is str
                and re.fullmatch("[0-9a-f]{40}", revision) is not None,
                f"{name} must be a full lowercase commit",
            )
        tp_size = getattr(config, "tp_size", None)
        _require(
            type(tp_size) is int
            and tp_size in (2, 4)
            and (
                tp_size == 2
                or native_segment_profile_id is not None
                and progress_kind == "source_golden351"
            ),
            "TP4 requires native golden regions",
            code="unsupported_topology",
        )
        for name, expected in (
            ("dp_size", 2 if joint else 1),
            ("pp_size", 1),
            ("ep_size", 1),
            ("cp_size", 1),
        ):
            value = getattr(config, name, None)
            _require(
                type(value) is int and value == expected,
                "DSPark predictor requires its selected modeled attention-DP/PP1/EP1/CP1 topology",
                code="unsupported_topology",
            )
        _require(
            not joint or tp_size == 2,
            "joint DSPark requires two chips, shared MLP TP2 and singleton attention TP1",
            code="unsupported_topology",
        )
        _require(
            config.backend_name == "sglang"
            and type(config.backend_version) is str
            and bool(config.backend_version.strip()),
            "DSPark predictor requires sglang and an explicit framework version",
            code="incompatible_runtime",
        )
        _require(
            type(system) is str and bool(system.strip()), "system must be explicit"
        )
        _require(
            getattr(hw, "name", None) == system,
            "system differs from configured hardware",
            code="incompatible_runtime",
        )
        _require(
            type(systems_root) is str and bool(systems_root.strip()),
            "systems_root must be explicit",
        )
        _require(
            model_config_root is None
            or type(model_config_root) is str
            and bool(model_config_root.strip()),
            "model_config_root must be None or an explicit nonempty path",
        )
        _require(
            proposal_profile_id is None
            or type(proposal_profile_id) is str
            and re.fullmatch("[0-9a-f]{64}", proposal_profile_id) is not None,
            "proposal_profile_id must be a full lowercase digest",
        )
        _require(
            native_segment_profile_id is None
            or type(native_segment_profile_id) is str
            and re.fullmatch("[0-9a-f]{64}", native_segment_profile_id) is not None,
            "native_segment_profile_id must be a full lowercase digest",
        )
        _require(
            (native_segment_profile_id is None) == (native_capture_profile is None),
            "native_segment_profile_id and native_capture_profile must be selected together",
        )
        _require(
            native_initial_draft_buffer is None
            or native_segment_profile_id is not None,
            "initial draft buffer requires native region profiles",
        )
        _require(
            native_segment_profile_id is None or proposal_profile_id is None,
            "native regions and legacy proposal profiles select conflicting cost grains",
        )
        _require(
            type(output_token_id) is int
            and type(getattr(model, "vocab_size", None)) is int
            and 0 <= output_token_id < model.vocab_size,
            "output_token_id must be an integer inside the model vocabulary",
        )
        runtime_path, runtime_bytes, runtime_digest = _asset(
            runtime_profile, "runtime_profile"
        )
        runtime_text = _runtime_json(runtime_bytes)
        progress_path, progress_digest = None, None
        if progress_kind == "empirical":
            progress_path, _, progress_digest = _asset(
                progress_artifact, "progress_artifact"
            )

        try:
            import infercast
            from infercast.sdk import PerfDatabase
            from infercast.sdk.models.model_config import load_model_config
            from infercast.sdk.models.umd import build_model_description
            from infercast.sdk.perf_database.dsv41_dspark_proposal import MODEL_REVISION

            if native_source_work_policy is not None:
                from infercast.sdk.native_segment_module import SOURCE_WORK_POLICY

                _require(
                    native_source_work_policy == SOURCE_WORK_POLICY,
                    "native_source_work_policy is not the SDK versioned policy",
                )
        except ImportError as error:
            raise PredictorError(
                "provider_initialization_failed", "InferCast SDK is unavailable"
            ) from error
        _require(
            getattr(infercast, "__revision__", None) == provider_revision,
            "installed InferCast revision differs from provider_revision",
            code="incompatible_runtime",
        )
        _require(
            model_revision == MODEL_REVISION,
            "model_revision differs from the SDK-owned DSPark source cohort",
            code="incompatible_runtime",
        )
        try:
            authoritative = load_model_config(
                model_id, allow_download=False, revision=model_revision
            )
            model_config_ref = _bind_local_model_config(
                model, authoritative, model_config_root=model_config_root
            )
            description = build_model_description(
                model_id,
                backend="sglang",
                revision=model_revision,
                allow_download=False,
            )
            _geometry(model, description, model_id)
            database = PerfDatabase.open_fidb(
                system,
                "sglang",
                config.backend_version,
                systems_root=systems_root,
                database_mode=cost_mode,
            )
            stack_path = Path(database.slice_ref.stack_path)
            stack_digest = hashlib.sha256(stack_path.read_bytes()).hexdigest()
        except PredictorError:
            raise
        except ImportError as error:
            raise PredictorError(
                "provider_initialization_failed",
                f"DSPark SDK model/slice import failed: {error}",
            ) from error
        except (OSError, KeyError, ValueError, TypeError, NotImplementedError) as error:
            raise PredictorError(
                "data_unavailable", f"DSPark SDK model/slice is unavailable: {error}"
            ) from error

        if joint:
            try:
                from infercast.sdk.native_segment_analytic import (
                    JOINT_POLICY,
                    NativeSegmentAnalyticModel,
                )
                from infercast.sdk.perf_database.dsv41_dspark import (
                    decode_dspark_artifact,
                )

                policy, profiles, source_identity = _native_joint_source_from_binding(
                    native_joint_source_binding, runtime_text, model_revision
                )
                _require(
                    type(native_analytic_model) is dict
                    and set(native_analytic_model) == {"EXTEND", "DECODE"},
                    "joint costs require two phase analytic model descriptors",
                )
                models, model_identity = {}, {}
                for mode, ref in native_analytic_model.items():
                    path, raw, digest = _asset(ref, "native_analytic_model." + mode)
                    models[mode] = value = NativeSegmentAnalyticModel.from_dict(
                        decode_dspark_artifact(raw)
                    )
                    _require(
                        value.policy == JOINT_POLICY
                        and value.calibration == "none"
                        and value.tp_size == 2
                        and value.execution_profile_id == profiles[mode].profile_id,
                        "joint analytic model differs from its phase/source profile",
                        code="incompatible_runtime",
                    )
                    model_identity[mode] = dict(
                        path=str(path), sha256=digest, parameters=value.to_dict()
                    )
                self.dspark_runtime = _native_joint_runtime(
                    database,
                    source_policy=policy,
                    runtime_profile_json=runtime_text,
                    profiles=profiles,
                    analytic_models=models,
                    model_config_sha256=model_config_ref["sdk_canonical_sha256"],
                    model_seed=model_seed,
                    output_token_id=output_token_id,
                    cost_evidence_path=native_cost_evidence_path,
                )
            except PredictorError:
                raise
            except (
                ImportError,
                OSError,
                KeyError,
                ValueError,
                TypeError,
                StopIteration,
            ) as error:
                raise PredictorError(
                    "data_unavailable", f"joint source is unavailable: {error}"
                ) from error
            self._identity = dict(
                model_id=model_id,
                model_revision=model_revision,
                native_model_config=model_config_ref,
                provider_revision=provider_revision,
                provider_version=getattr(infercast, "__version__", None),
                system=system,
                framework_version=config.backend_version,
                topology=dict(tp=2, dp=2, pp=1, ep=1, cp=1),
                runtime_profile=dict(path=str(runtime_path), sha256=runtime_digest),
                progress_kind=progress_kind,
                progress_model_ref=self.dspark_runtime.session.progress_model.model_ref,
                native_joint_source_binding=source_identity,
                native_analytic_model=model_identity,
                slice=dict(
                    path=str(database.slice_ref.slice_dir),
                    stack_path=str(stack_path),
                    stack_sha256=stack_digest,
                    database_mode=cost_mode,
                ),
            )
            return

        from sglang_simulator.simulation.sglang.dspark import (
            DSparkSession,
            DSparkTopology,
        )
        from sglang_simulator.simulation.sglang.dspark_runtime import (
            DSparkNativeRuntime,
        )
        from sglang_simulator.time_predictor.dspark_progress import (
            EmpiricalDSparkProgress,
        )
        from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel

        native_policy, capture_identity, capture_profile = None, None, None
        source_binding_identity = None
        if native_segment_profile_id is not None:
            try:
                from infercast.sdk.decode_forward import DecodeExecutionProfile
                from infercast.sdk.perf_database.dsv41_dspark import (
                    decode_dspark_artifact,
                )

                capture_path, capture_bytes, capture_digest = _asset(
                    native_capture_profile, "native_capture_profile"
                )
                capture_profile = DecodeExecutionProfile.from_dict(
                    decode_dspark_artifact(capture_bytes)
                )
                if native_source_binding is None:
                    native_policy = _native_source_policy(
                        database, native_segment_profile_id, capture_profile
                    )
                else:
                    native_policy, source_binding_identity = (
                        _native_source_policy_from_binding(
                            database,
                            native_segment_profile_id,
                            capture_profile,
                            native_source_binding,
                        )
                    )
                _require(
                    runtime_text == native_policy.runtime_profile_json,
                    "configured runtime differs from the verified native physical source",
                    code="incompatible_runtime",
                )
                _require(
                    tp_size == native_policy.tp_size,
                    "configured topology differs from the verified native physical source",
                    code="incompatible_runtime",
                )
                capture_identity = {
                    "path": str(capture_path),
                    "sha256": capture_digest,
                    "profile_id": capture_profile.profile_id,
                }
            except PredictorError:
                raise
            except ImportError as error:
                raise PredictorError(
                    "provider_initialization_failed",
                    "native projection SDK is unavailable",
                ) from error
            except (
                OSError,
                KeyError,
                ValueError,
                TypeError,
                NotImplementedError,
            ) as error:
                raise PredictorError(
                    "data_unavailable", f"native source policy is unavailable: {error}"
                ) from error
        analytic_model, analytic_identity = None, None
        if native_analytic_model is not None:
            try:
                from infercast.sdk.native_segment_analytic import (
                    NativeSegmentAnalyticModel,
                )
                from infercast.sdk.perf_database.dsv41_dspark import (
                    decode_dspark_artifact,
                )

                analytic_path, analytic_bytes, analytic_digest = _asset(
                    native_analytic_model, "native_analytic_model"
                )
                analytic_model = NativeSegmentAnalyticModel.from_dict(
                    decode_dspark_artifact(analytic_bytes)
                )
                document = analytic_model.to_dict()
                _require(
                    document["execution_profile_id"]
                    == native_policy.execution_profile_id
                    and document["tp_size"] == tp_size
                    and document["model_config_sha256"]
                    == model_config_ref["sdk_canonical_sha256"],
                    "analytic model differs from the verified source/model topology",
                    code="incompatible_runtime",
                )
                analytic_identity = {
                    "path": str(analytic_path),
                    "sha256": analytic_digest,
                    "parameters": document,
                }
            except PredictorError:
                raise
            except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
                raise PredictorError(
                    "data_unavailable", f"native analytic model is unavailable: {error}"
                ) from error
        # An explicit source binding may establish resource facts without a
        # numerical latency row. The default silicon witness owner is unchanged.
        if (
            source_binding_identity is not None
            and native_initial_draft_buffer is not None
        ):
            initial_capacity, initial_buffer_identity = _source_initial_draft_buffer(
                database,
                native_policy,
                capture_profile,
                native_initial_draft_buffer,
                source_binding_identity,
            )
        else:
            initial_capacity, initial_buffer_identity = _initial_draft_buffer(
                database, native_policy, capture_profile, native_initial_draft_buffer
            )
        if progress_kind == "empirical":
            _require(
                getattr(native_policy, "golden_law", None) is None,
                "golden native regions require source_golden351 progress",
            )
            progress = EmpiricalDSparkProgress(
                progress_path,
                expected_sha256=progress_digest,
                runtime_profile_json=runtime_text,
            )
        else:
            from sglang_simulator.time_predictor.dspark_golden_progress import (
                SourceGoldenDSparkProgress,
            )

            progress = SourceGoldenDSparkProgress(native_policy, model_seed=model_seed)
        _require(
            progress.topology == (tp_size, 1, 1),
            "progress originals require the same modeled TP/DP1/PP1 topology",
            code="unsupported_topology",
        )
        cost_options = {}
        if native_source_work_policy is not None:
            cost_options["native_source_work_policy"] = native_source_work_policy
        if analytic_model is not None:
            cost_options.update(
                native_analytic_model=analytic_model,
                cost_evidence_path=native_cost_evidence_path,
            )
        costs = UMDDSparkCostModel(
            database,
            proposal_profile_id=proposal_profile_id,
            native_source_policy=native_policy,
            **cost_options,
        )
        self.dspark_runtime = DSparkNativeRuntime(
            session=DSparkSession(costs, progress),
            topology=DSparkTopology(tp_size, 1, 1),
            runtime_profile_json=runtime_text,
            output_token_id=output_token_id,
            reference_events=native_policy is not None,
            initial_draft_buffer_capacity=initial_capacity,
        )
        self._identity = {
            "model_id": model_id,
            "model_revision": model_revision,
            "native_model_config": model_config_ref,
            "provider_revision": provider_revision,
            "provider_version": getattr(infercast, "__version__", None),
            "system": system,
            "framework_version": config.backend_version,
            "topology": {"tp": tp_size, "dp": 1, "pp": 1, "ep": 1, "cp": 1},
            "runtime_profile": {"path": str(runtime_path), "sha256": runtime_digest},
            "progress_artifact": {
                "path": str(progress_path),
                "sha256": progress_digest,
            }
            if progress_kind == "empirical"
            else None,
            "progress_model_ref": progress.model_ref,
            "proposal_profile_id": proposal_profile_id,
            "native_segment_profile_id": native_segment_profile_id,
            "native_capture_profile": capture_identity,
            "slice": {
                "path": str(database.slice_ref.slice_dir),
                "stack_path": str(stack_path),
                "stack_sha256": stack_digest,
                "database_mode": cost_mode,
            },
        }
        if progress_kind == "source_golden351":
            self._identity.update(progress_kind=progress_kind, **progress.identity)
        if native_source_work_policy is not None:
            self._identity["native_source_work_policy"] = native_source_work_policy
        if initial_buffer_identity is not None:
            self._identity["native_initial_draft_buffer"] = initial_buffer_identity
        if analytic_identity is not None:
            self._identity["native_analytic_model"] = analytic_identity
        if source_binding_identity is not None:
            self._identity["native_source_binding"] = source_binding_identity

    def predict_infer_time(self, batch):
        raise PredictorError(
            "unsupported_execution_profile",
            "DSPark timing requires an attached native DSPark step result",
        )

    def get_metrics(self):
        if "native_joint_source_binding" in self._identity:
            # One joint whole-worker price, never seven fabricated regional costs.
            runtime = self.dspark_runtime
            return {
                "infercast_dspark": {
                    **copy.deepcopy(self._identity),
                    "cost_grain": "whole_worker",
                    "source_resolution": "sdk_native_joint_source_originals",
                    "hybrid_cost_queries": runtime.session.cost_model.get_cost_metrics(),
                    "timing_qualified": False,
                    "accuracy_admitted": False,
                    "public_factory_clock_supported": False,
                }
            }
        from sglang_simulator.simulation.sglang.dspark import (
            _DECODE_PHASES,
            _PREFILL_PHASES,
        )

        available = self.dspark_runtime.session.cost_model.preparers
        coverage = {
            name: {
                "available": [phase for phase in phases if phase in available],
                "missing": [phase for phase in phases if phase not in available],
            }
            for name, phases in (
                ("prefill", _PREFILL_PHASES),
                ("decode", _DECODE_PHASES),
            )
        }
        driver = self.dspark_runtime.reference_driver
        reference_policy_available = (
            type(
                getattr(
                    self.dspark_runtime.session.cost_model.native_source_policy,
                    "reference_needs_cpu_seq_lens",
                    None,
                )
            )
            is bool
        )
        result = {
            "infercast_dspark": {
                **copy.deepcopy(self._identity),
                "phase_preparers": coverage,
                "source_resolution": (
                    "sdk_native_segment_projection"
                    if self._identity["native_segment_profile_id"] is not None
                    else "proposal_capture_only"
                ),
                "unresolved_source_facts": (
                    (
                        []
                        if reference_policy_available
                        else ["reference_backend_cpu_mirror_policy"]
                    )
                    if self._identity["native_segment_profile_id"] is not None
                    else [
                        "target_graph_key",
                        "target_graph_padding",
                        "target_epilogue_folding",
                        "target_monokernel_source_state_ledger",
                    ]
                ),
                "reference_event_driver": (
                    None
                    if driver is None
                    else {
                        "implementation": "native_scheduler_events_v1",
                        "scheduler_bound": driver.owner is not None,
                        "reference_backend_policy_available": reference_policy_available,
                        "published_forwards": self.dspark_runtime._forward_id,
                        "pending_results": len(driver.adapter.ledger.pending),
                        "failure": copy.deepcopy(driver.failure),
                    }
                ),
                "complete_worker_costs": False,
                "timing_qualified": False,
                "accuracy_admitted": False,
            }
        }
        if "native_analytic_model" in self._identity:
            cost_metrics = self.dspark_runtime.session.cost_model.get_cost_metrics()
            result["infercast_dspark"]["hybrid_cost_queries"] = cost_metrics
        return result

    def reset_metrics(self):
        # Identity and coverage are not interval counters or simulation state.
        return None
