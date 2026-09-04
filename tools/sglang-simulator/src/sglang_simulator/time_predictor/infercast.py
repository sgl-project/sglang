"""InferCast model-forward latency predictor."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    PredictorError,
    ScheduleBatch,
)

CONTRACT_VERSION = 1
REDUCTION_POLICY = "mean_attention_flops_v1"
_CONTEXT_MODES = {"EXTEND", "MIXED"}
_SUPPORTED_MODES = _CONTEXT_MODES | {"DECODE"}
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReducedForward:
    method: str
    arguments: dict[str, int | float]
    raw_attention_scale: float | None = None


def _error(code: str, message: str, **details: Any) -> PredictorError:
    return PredictorError(code, message, **details)


def reduce_batch(batch: ScheduleBatch) -> ReducedForward:
    """Apply the frozen ``mean_attention_flops_v1`` reduction."""
    mode = batch.forward_mode
    if mode not in _SUPPORTED_MODES:
        raise _error(
            "unsupported_forward_mode",
            f"unsupported SGLang forward mode {mode!r}",
            forward_mode=mode,
        )
    if not batch.reqs:
        raise _error("invalid_batch", "provider batch must not be empty")

    for index, request in enumerate(batch.reqs):
        extend = request.extend_length
        prefix = request.past_kv_length
        if type(extend) is not int or type(prefix) is not int:
            raise _error(
                "invalid_batch",
                "request lengths must be integers",
                request_index=index,
            )
        if extend < 1 or prefix < 0:
            raise _error(
                "invalid_batch",
                "extend length must be positive and prefix length non-negative",
                request_index=index,
                extend_length=extend,
                past_kv_length=prefix,
            )

    batch_size = len(batch.reqs)
    sum_prefix = sum(request.past_kv_length for request in batch.reqs)
    if mode == "DECODE":
        if any(request.extend_length != 1 for request in batch.reqs):
            raise _error("invalid_batch", "decode requires extend length 1")
        return ReducedForward(
            "estimate_decode_forward_ms",
            {
                "batch_size": batch_size,
                "history_len": sum_prefix // batch_size,
            },
        )

    sum_extend = sum(request.extend_length for request in batch.reqs)
    prefix_len = sum_prefix // batch_size
    extend_len = (sum_prefix + sum_extend) // batch_size - prefix_len
    raw_scale = (
        batch_size
        * sum(
            request.extend_length * (2 * request.past_kv_length + request.extend_length)
            for request in batch.reqs
        )
        / (sum_extend * (2 * sum_prefix + sum_extend))
    )
    applied_scale = raw_scale if raw_scale >= 0.4 else 1.0
    return ReducedForward(
        "estimate_extend_forward_ms",
        {
            "batch_size": batch_size,
            "extend_len": extend_len,
            "prefix_len": prefix_len,
            "seq_imbalance_correction_scale": applied_scale,
        },
        raw_attention_scale=raw_scale,
    )


def validate_topology(config: SchedulerConfig) -> None:
    topology = {
        "tp": config.tp_size,
        "ep": config.ep_size,
        "pp": config.pp_size,
        "dp": config.dp_size,
        "cp": config.cp_size,
    }
    if any(type(value) is not int or value < 1 for value in topology.values()):
        raise _error(
            "unsupported_topology",
            f"topology values must be positive integers: {topology}",
            **topology,
        )
    if config.dp_size != 1 or config.cp_size != 1:
        raise _error(
            "unsupported_topology",
            "InferCast provider v1 requires dp_size=cp_size=1",
            **topology,
        )
    if config.tp_size % config.ep_size:
        raise _error(
            "unsupported_topology",
            "tp_size must be divisible by ep_size",
            **topology,
        )


def milliseconds_to_seconds(value: Any) -> float:
    try:
        latency_ms = float(value)
    except (TypeError, ValueError) as error:
        raise _error(
            "invalid_provider_output",
            f"InferCast returned non-numeric latency {value!r}",
        ) from error
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise _error(
            "invalid_provider_output",
            f"InferCast latency must be finite and positive, got {latency_ms!r}",
            latency_ms=latency_ms,
        )
    return latency_ms / 1000.0


def _canonical_dtype(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(getattr(value, "value", value)).lower().removeprefix("torch.")
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "float8": "fp8",
    }.get(normalized, normalized)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferCastTimePredictor(InferTimePredictor):
    """Call one InferCast UMD estimate for one prepared SGLang forward."""

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *,
        model_id: str | None,
        system: str | None,
        systems_root: str | None,
        database_mode: str = "SILICON",
        attn_kernel_impl: str | None,
        attn_dtype: str | None,
        kv_cache_dtype: str | None,
        provider_revision: str | None,
        _provider: Any | None = None,
        _provider_version: str = "test",
        _stack_digest: str = "0" * 64,
        **kwargs,
    ) -> None:
        super().__init__(model, hw, config)
        validate_topology(config)

        required = {
            "model_id": model_id,
            "system": system,
            "systems_root": systems_root,
            "backend_version": config.backend_version,
            "attn_kernel_impl": attn_kernel_impl,
            "attn_dtype": attn_dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "provider_revision": provider_revision,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise _error(
                "provider_initialization_failed",
                f"missing InferCast configuration: {', '.join(missing)}",
                missing_fields=missing,
            )
        if config.backend_name != "sglang":
            raise _error(
                "incompatible_runtime",
                f"expected backend 'sglang', got {config.backend_name!r}",
            )
        if str(database_mode).upper() != "SILICON":
            raise _error(
                "incompatible_runtime",
                "InferCast provider v1 requires database_mode='SILICON'",
            )
        if not _FULL_COMMIT.fullmatch(str(provider_revision)):
            raise _error(
                "provider_initialization_failed",
                "provider_revision must be a full lowercase commit",
            )

        self._validate_dtypes(model, config, attn_dtype, kv_cache_dtype)
        self._model_id = str(model_id)
        self._system = str(system)
        self._provider_revision = str(provider_revision)
        self._runtime = {
            "tp_size": config.tp_size,
            "pp_size": config.pp_size,
            "moe_tp_size": config.moe_tp_size,
            "moe_ep_size": config.moe_ep_size,
            "attention_dp_size": config.attn_dp_size,
            "attn_kernel_impl": attn_kernel_impl,
            "attn_dtype": attn_dtype,
            "kv_cache_dtype": kv_cache_dtype,
        }

        if _provider is None:
            (
                self._provider,
                self._provider_version,
                self._stack_digest,
            ) = self._build_provider(str(systems_root))
            self._validate_model_geometry(model)
        else:
            self._provider = _provider
            self._provider_version = _provider_version
            self._stack_digest = _stack_digest
        self._calls = dict.fromkeys(("total", "context", "mixed", "decode"), 0)

    @staticmethod
    def _validate_dtypes(model, config, attn_dtype, kv_cache_dtype) -> None:
        expected_attn = _canonical_dtype(model.torch_dtype)
        actual_attn = _canonical_dtype(attn_dtype)
        if expected_attn and expected_attn != actual_attn:
            raise _error(
                "incompatible_runtime",
                "attn_dtype does not match the loaded model",
                model_torch_dtype=expected_attn,
                attn_dtype=actual_attn,
            )
        expected_kv = _canonical_dtype(config.kv_cache_data_type)
        actual_kv = _canonical_dtype(kv_cache_dtype)
        if expected_kv and expected_kv != actual_kv:
            raise _error(
                "incompatible_runtime",
                "kv_cache_dtype does not match the scheduler",
                scheduler_kv_cache_dtype=expected_kv,
                kv_cache_dtype=actual_kv,
            )

    def _validate_model_geometry(self, model: ModelInfo) -> None:
        geometry = getattr(getattr(self._provider, "desc", None), "geometry", None)
        if not isinstance(geometry, dict):
            raise _error(
                "incompatible_runtime",
                "InferCast provider does not expose model geometry",
            )
        fields = {
            "hidden_size": ("hidden", "hidden_size"),
            "num_hidden_layers": ("num_layers",),
            "num_attention_heads": ("n_heads",),
            "num_key_value_heads": ("n_kv",),
            "head_dim": ("head_dim",),
        }
        mismatches = {}
        for model_field, geometry_fields in fields.items():
            expected = getattr(model, model_field)
            actual = next(
                (geometry[name] for name in geometry_fields if name in geometry),
                None,
            )
            if (
                expected is not None
                and actual is not None
                and int(expected) != int(actual)
            ):
                mismatches[model_field] = {"sglang": expected, "infercast": actual}
        if mismatches:
            raise _error(
                "incompatible_runtime",
                "InferCast model geometry does not match the loaded SGLang model",
                mismatches=mismatches,
            )

    def _build_provider(self, systems_root: str) -> tuple[Any, str, str]:
        try:
            import infercast
            from infercast.sdk import PerfDatabase, build_umd_static_model

            database = PerfDatabase.open_fidb(
                self._system,
                "sglang",
                str(self.config.backend_version),
                systems_root=systems_root,
                database_mode="SILICON",
            )
            provider = build_umd_static_model(
                database,
                self._model_id,
                backend="sglang",
            )
            actual_revision = getattr(infercast, "__revision__", None)
            if actual_revision and actual_revision != self._provider_revision:
                raise _error(
                    "incompatible_runtime",
                    "configured provider_revision does not match InferCast",
                    configured_revision=self._provider_revision,
                    actual_revision=actual_revision,
                )
            return (
                provider,
                infercast.__version__,
                _sha256(database.slice_ref.stack_path),
            )
        except ImportError as error:
            raise _error(
                "provider_initialization_failed",
                "InferCast is not installed",
            ) from error
        except FileNotFoundError as error:
            raise _error(
                "data_unavailable",
                f"InferCast FIDB data is unavailable: {error}",
            ) from error
        except PredictorError:
            raise
        except Exception as error:
            raise _error(
                "provider_initialization_failed",
                f"failed to initialize InferCast: {error}",
            ) from error

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        reduced = reduce_batch(batch)
        phase = {
            "DECODE": "decode",
            "MIXED": "mixed",
            "EXTEND": "context",
        }[batch.forward_mode]
        self._calls["total"] += 1
        self._calls[phase] += 1
        try:
            latency_ms = getattr(self._provider, reduced.method)(
                **reduced.arguments,
                **self._runtime,
            )
        except NotImplementedError as error:
            raise _error(
                "prediction_failed",
                f"InferCast does not support this forward yet: {error}",
                forward_mode=batch.forward_mode,
                **reduced.arguments,
            ) from error
        except Exception as error:
            infercast_code = getattr(error, "code", None)
            code = (
                "data_unavailable"
                if infercast_code == "perf_data_not_available"
                else "prediction_failed"
            )
            raise _error(
                code,
                f"InferCast prediction failed: {error}",
                infercast_code=infercast_code,
                infercast_details=getattr(error, "details", {}),
            ) from error
        return milliseconds_to_seconds(latency_ms)

    def get_metrics(self) -> dict:
        return {
            "infercast": {
                "contract_version": CONTRACT_VERSION,
                "reduction_policy": REDUCTION_POLICY,
                "provider_version": self._provider_version,
                "provider_revision": self._provider_revision,
                "model_id": self._model_id,
                "system": self._system,
                "backend": self.config.backend_name,
                "framework_version": str(self.config.backend_version),
                "database_profile": "framework",
                "database_mode": "silicon",
                "stack_digest": self._stack_digest,
                "topology": {
                    "tp": self.config.tp_size,
                    "ep": self.config.ep_size,
                    "pp": self.config.pp_size,
                    "dp": self.config.dp_size,
                    "cp": self.config.cp_size,
                },
                "runtime": {
                    name: self._runtime[name]
                    for name in (
                        "attn_kernel_impl",
                        "attn_dtype",
                        "kv_cache_dtype",
                    )
                },
                "calls": dict(self._calls),
            }
        }

    def reset_metrics(self) -> None:
        self._calls = dict.fromkeys(self._calls, 0)
