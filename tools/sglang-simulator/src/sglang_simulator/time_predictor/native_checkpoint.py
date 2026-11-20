"""Stable local checkpoint binding for the modeled native realization."""

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from sglang_simulator.time_predictor.base import PredictorError


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
            mismatches[field] = {"sglang": actual, "simopus": expected}
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
            "simopus": expected_ratios,
        }
    _require(
        not mismatches,
        "SDK geometry differs from the loaded SGLang model",
        code="incompatible_runtime",
        mismatches=mismatches,
    )
