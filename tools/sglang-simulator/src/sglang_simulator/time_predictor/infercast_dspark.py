"""Explicit DSPark assembly; complete worker phase measurements remain required."""

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
        proposal_profile_id=None,
        output_token_id=1,
        model_config_root=None,
        **kwargs,
    ):
        _require(
            not kwargs, "unknown DSPark predictor configuration", fields=sorted(kwargs)
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
        for name, expected in (
            ("tp_size", 2),
            ("dp_size", 1),
            ("pp_size", 1),
            ("ep_size", 1),
            ("cp_size", 1),
        ):
            value = getattr(config, name, None)
            _require(
                type(value) is int and value == expected,
                "DSPark predictor requires modeled TP2/DP1/PP1/EP1/CP1",
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
            type(output_token_id) is int
            and type(getattr(model, "vocab_size", None)) is int
            and 0 <= output_token_id < model.vocab_size,
            "output_token_id must be an integer inside the model vocabulary",
        )
        runtime_path, runtime_bytes, runtime_digest = _asset(
            runtime_profile, "runtime_profile"
        )
        runtime_text = _runtime_json(runtime_bytes)
        progress_path, _, progress_digest = _asset(
            progress_artifact, "progress_artifact"
        )

        try:
            import infercast
            from infercast.sdk import PerfDatabase
            from infercast.sdk.models.model_config import load_model_config
            from infercast.sdk.models.umd import build_model_description
            from infercast.sdk.perf_database.dsv41_dspark_proposal import MODEL_REVISION
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
                database_mode="SILICON",
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

        progress = EmpiricalDSparkProgress(
            progress_path,
            expected_sha256=progress_digest,
            runtime_profile_json=runtime_text,
        )
        _require(
            progress.topology == (2, 1, 1),
            "progress originals require the same modeled TP2/DP1/PP1 topology",
            code="unsupported_topology",
        )
        costs = UMDDSparkCostModel(database, proposal_profile_id=proposal_profile_id)
        self.dspark_runtime = DSparkNativeRuntime(
            session=DSparkSession(costs, progress),
            topology=DSparkTopology(2, 1, 1),
            runtime_profile_json=runtime_text,
            output_token_id=output_token_id,
        )
        self._identity = {
            "model_id": model_id,
            "model_revision": model_revision,
            "native_model_config": model_config_ref,
            "provider_revision": provider_revision,
            "provider_version": getattr(infercast, "__version__", None),
            "system": system,
            "framework_version": config.backend_version,
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 1, "cp": 1},
            "runtime_profile": {"path": str(runtime_path), "sha256": runtime_digest},
            "progress_artifact": {
                "path": str(progress_path),
                "sha256": progress_digest,
            },
            "progress_model_ref": progress.model_ref,
            "proposal_profile_id": proposal_profile_id,
            "slice": {
                "path": str(database.slice_ref.slice_dir),
                "stack_path": str(stack_path),
                "stack_sha256": stack_digest,
                "database_mode": "SILICON",
            },
        }

    def predict_infer_time(self, batch):
        raise PredictorError(
            "unsupported_execution_profile",
            "DSPark timing requires an attached native DSPark step result",
        )

    def get_metrics(self):
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
        return {
            "infercast_dspark": {
                **copy.deepcopy(self._identity),
                "phase_preparers": coverage,
                "source_resolution": "proposal_capture_only",
                "unresolved_source_facts": [
                    "target_graph_key",
                    "target_graph_padding",
                    "target_epilogue_folding",
                    "target_monokernel_source_state_ledger",
                ],
                "complete_worker_costs": False,
                "timing_qualified": False,
                "accuracy_admitted": False,
            }
        }

    def reset_metrics(self):
        # Identity and coverage are not interval counters or simulation state.
        return None
