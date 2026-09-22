"""Configuration primitives for the local multi-model server supervisor.

This module intentionally contains no SGLang runtime imports.  Keeping the
configuration parser dependency-free makes invalid topology configurations
fail before CUDA processes are started and lets its tests run without GPUs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


class MultiModelConfigError(ValueError):
    """Raised when a multi-model supervisor configuration is invalid."""


@dataclasses.dataclass(frozen=True)
class ModelReplica:
    """One independent SGLang runtime replica owned by the supervisor."""

    model_id: str
    model_path: str
    server_args: dict[str, Any]
    gpu_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class RoutingProfile:
    """A virtual model resolved to compatible concrete model IDs."""

    model_id: str
    candidates: tuple[str, ...]
    max_kv_utilization: float
    max_waiting_requests: int
    min_free_tokens: int


@dataclasses.dataclass(frozen=True)
class ModelResolverConfig:
    """Settings for the optional public resource-aware Router edge."""

    host: str
    port: int
    refresh_interval_secs: float
    stale_after_secs: float
    request_timeout_secs: float
    profiles: tuple[RoutingProfile, ...]


@dataclasses.dataclass(frozen=True)
class MultiModelConfig:
    """Validated input for the static local multi-model supervisor MVP."""

    router_args: dict[str, Any]
    worker_host: str
    worker_base_port: int
    startup_timeout_secs: int
    allow_gpu_sharing: bool
    replicas: tuple[ModelReplica, ...]
    model_resolver: ModelResolverConfig | None


_TOP_LEVEL_KEYS = {
    "models",
    "router",
    "startup_timeout_secs",
    "worker_base_port",
    "worker_host",
    "allow_gpu_sharing",
    "model_resolver",
}
_MODEL_KEYS = {"gpu_groups", "model_id", "model_path", "server_args"}
_MODEL_RESOLVER_KEYS = {
    "host",
    "port",
    "refresh_interval_secs",
    "stale_after_secs",
    "request_timeout_secs",
    "profiles",
}
_ROUTING_PROFILE_KEYS = {
    "model_id",
    "candidates",
    "max_kv_utilization",
    "max_waiting_requests",
    "min_free_tokens",
}
_SUPERVISOR_OWNED_SERVER_ARGS = {
    "base_gpu_id",
    "dp_size",
    "host",
    "model_path",
    "port",
    "served_model_name",
}
_UNSUPPORTED_SERVER_ARGS = {
    "disaggregation_mode",
    "enable_dp_attention",
    "nnodes",
    "node_rank",
}


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MultiModelConfigError(f"{path} must be an object")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise MultiModelConfigError(f"{path} must be a non-empty string")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MultiModelConfigError(f"{path} must be a positive integer")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise MultiModelConfigError(f"{path} must be a boolean")
    return value


def _require_positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise MultiModelConfigError(f"{path} must be a positive number")
    return float(value)


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultiModelConfigError(f"{path} must be a non-negative integer")
    return value


def _parse_gpu_group(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MultiModelConfigError(f"{path} must be a non-empty list of GPU IDs")

    gpu_ids: list[int] = []
    for offset, gpu_id in enumerate(value):
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise MultiModelConfigError(
                f"{path}[{offset}] must be a non-negative integer GPU ID"
            )
        gpu_ids.append(gpu_id)

    if len(set(gpu_ids)) != len(gpu_ids):
        raise MultiModelConfigError(f"{path} contains the same GPU more than once")
    return tuple(gpu_ids)


def _parse_model_replicas(
    model: dict[str, Any],
    model_index: int,
    used_gpu_ids: set[int],
    allow_gpu_sharing: bool,
) -> list[ModelReplica]:
    path = f"models[{model_index}]"
    unknown_keys = set(model) - _MODEL_KEYS
    if unknown_keys:
        raise MultiModelConfigError(
            f"{path} contains unsupported keys: {', '.join(sorted(unknown_keys))}"
        )

    model_id = _require_nonempty_string(model.get("model_id"), f"{path}.model_id")
    model_path = _require_nonempty_string(
        model.get("model_path"), f"{path}.model_path"
    )
    server_args = _require_mapping(model.get("server_args", {}), f"{path}.server_args")

    controlled_args = set(server_args) & _SUPERVISOR_OWNED_SERVER_ARGS
    if controlled_args:
        raise MultiModelConfigError(
            f"{path}.server_args cannot set supervisor-owned fields: "
            f"{', '.join(sorted(controlled_args))}"
        )
    unsupported_args = set(server_args) & _UNSUPPORTED_SERVER_ARGS
    if unsupported_args:
        raise MultiModelConfigError(
            f"{path}.server_args cannot set MVP-unsupported fields: "
            f"{', '.join(sorted(unsupported_args))}"
        )

    tp_size = _require_positive_int(server_args.get("tp_size", 1), f"{path}.server_args.tp_size")
    pp_size = _require_positive_int(server_args.get("pp_size", 1), f"{path}.server_args.pp_size")
    expected_gpus = tp_size * pp_size

    gpu_groups = model.get("gpu_groups")
    if not isinstance(gpu_groups, list) or not gpu_groups:
        raise MultiModelConfigError(f"{path}.gpu_groups must be a non-empty list")

    replicas: list[ModelReplica] = []
    for replica_index, gpu_group in enumerate(gpu_groups):
        gpu_ids = _parse_gpu_group(gpu_group, f"{path}.gpu_groups[{replica_index}]")
        if len(gpu_ids) != expected_gpus:
            raise MultiModelConfigError(
                f"{path}.gpu_groups[{replica_index}] has {len(gpu_ids)} GPUs, but "
                f"tp_size * pp_size requires {expected_gpus}"
            )
        overlap = used_gpu_ids.intersection(gpu_ids)
        if overlap and not allow_gpu_sharing:
            raise MultiModelConfigError(
                f"{path}.gpu_groups[{replica_index}] reuses GPU IDs already assigned "
                f"to another replica: {', '.join(map(str, sorted(overlap)))}. "
                "Set config.allow_gpu_sharing to true only for explicitly "
                "oversubscribed process-level deployments."
            )
        used_gpu_ids.update(gpu_ids)
        replicas.append(
            ModelReplica(
                model_id=model_id,
                model_path=model_path,
                server_args=dict(server_args),
                gpu_ids=gpu_ids,
            )
        )
    return replicas


def _parse_model_resolver(
    raw_resolver: Any,
    *,
    router_args: dict[str, Any],
    model_ids: set[str],
) -> ModelResolverConfig | None:
    if raw_resolver is None:
        return None

    resolver = _require_mapping(raw_resolver, "config.model_resolver")
    unknown_keys = set(resolver) - _MODEL_RESOLVER_KEYS
    if unknown_keys:
        raise MultiModelConfigError(
            "config.model_resolver contains unsupported keys: "
            + ", ".join(sorted(unknown_keys))
        )

    host = _require_nonempty_string(
        resolver.get("host", router_args.get("host", "127.0.0.1")),
        "config.model_resolver.host",
    )
    port = _require_positive_int(
        resolver.get("port"), "config.model_resolver.port"
    )
    router_port = router_args.get("port", 30000)
    if port == router_port:
        raise MultiModelConfigError(
            "config.model_resolver.port must differ from config.router.port; "
            "the resolver is the public endpoint and the Rust Router is its backend"
        )

    refresh_interval_secs = _require_positive_float(
        resolver.get("refresh_interval_secs", 1.0),
        "config.model_resolver.refresh_interval_secs",
    )
    stale_after_secs = _require_positive_float(
        resolver.get("stale_after_secs", 3.0),
        "config.model_resolver.stale_after_secs",
    )
    if stale_after_secs < refresh_interval_secs:
        raise MultiModelConfigError(
            "config.model_resolver.stale_after_secs must be at least "
            "refresh_interval_secs"
        )
    request_timeout_secs = _require_positive_float(
        resolver.get("request_timeout_secs", 120.0),
        "config.model_resolver.request_timeout_secs",
    )

    raw_profiles = resolver.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise MultiModelConfigError(
            "config.model_resolver.profiles must be a non-empty list"
        )

    profile_ids: set[str] = set()
    profiles: list[RoutingProfile] = []
    for profile_index, raw_profile in enumerate(raw_profiles):
        path = f"config.model_resolver.profiles[{profile_index}]"
        profile = _require_mapping(raw_profile, path)
        unknown_profile_keys = set(profile) - _ROUTING_PROFILE_KEYS
        if unknown_profile_keys:
            raise MultiModelConfigError(
                f"{path} contains unsupported keys: "
                + ", ".join(sorted(unknown_profile_keys))
            )
        model_id = _require_nonempty_string(profile.get("model_id"), f"{path}.model_id")
        if model_id in model_ids:
            raise MultiModelConfigError(
                f"{path}.model_id {model_id!r} conflicts with a concrete model ID"
            )
        if model_id in profile_ids:
            raise MultiModelConfigError(f"{path}.model_id duplicates {model_id!r}")
        profile_ids.add(model_id)

        candidates = profile.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise MultiModelConfigError(f"{path}.candidates must be a non-empty list")
        if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
            raise MultiModelConfigError(
                f"{path}.candidates must contain non-empty model ID strings"
            )
        if len(set(candidates)) != len(candidates):
            raise MultiModelConfigError(f"{path}.candidates contains duplicates")
        missing_models = sorted(set(candidates) - model_ids)
        if missing_models:
            raise MultiModelConfigError(
                f"{path}.candidates references unknown concrete models: "
                + ", ".join(missing_models)
            )

        max_kv_utilization = _require_positive_float(
            profile.get("max_kv_utilization", 0.9),
            f"{path}.max_kv_utilization",
        )
        if max_kv_utilization > 1.0:
            raise MultiModelConfigError(
                f"{path}.max_kv_utilization must be less than or equal to 1"
            )
        max_waiting_requests = _require_positive_int(
            profile.get("max_waiting_requests", 64),
            f"{path}.max_waiting_requests",
        )
        min_free_tokens = _require_nonnegative_int(
            profile.get("min_free_tokens", 1), f"{path}.min_free_tokens"
        )
        profiles.append(
            RoutingProfile(
                model_id=model_id,
                candidates=tuple(candidates),
                max_kv_utilization=max_kv_utilization,
                max_waiting_requests=max_waiting_requests,
                min_free_tokens=min_free_tokens,
            )
        )

    return ModelResolverConfig(
        host=host,
        port=port,
        refresh_interval_secs=refresh_interval_secs,
        stale_after_secs=stale_after_secs,
        request_timeout_secs=request_timeout_secs,
        profiles=tuple(profiles),
    )


def parse_multi_model_config(raw_config: Any) -> MultiModelConfig:
    """Validate the JSON document accepted by ``--multi-model-config``.

    The MVP deliberately accepts a static, single-node deployment.  Every GPU
    belongs to one and only one replica.  A model may have several replicas,
    which implements standard data parallelism at the process level.
    """

    config = _require_mapping(raw_config, "config")
    unknown_keys = set(config) - _TOP_LEVEL_KEYS
    if unknown_keys:
        raise MultiModelConfigError(
            "config contains unsupported keys: " + ", ".join(sorted(unknown_keys))
        )

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise MultiModelConfigError("config.models must be a non-empty list")

    router_args = _require_mapping(config.get("router", {}), "config.router")
    if "worker_urls" in router_args:
        raise MultiModelConfigError(
            "config.router.worker_urls is managed by the multi-model supervisor"
        )
    if router_args.get("enable_igw") is False:
        raise MultiModelConfigError(
            "config.router.enable_igw cannot be false in multi-model mode"
        )

    worker_host = _require_nonempty_string(
        config.get("worker_host", "127.0.0.1"), "config.worker_host"
    )
    worker_base_port = _require_positive_int(
        config.get("worker_base_port", 31000), "config.worker_base_port"
    )
    startup_timeout_secs = _require_positive_int(
        config.get("startup_timeout_secs", 1800), "config.startup_timeout_secs"
    )
    allow_gpu_sharing = _require_bool(
        config.get("allow_gpu_sharing", False), "config.allow_gpu_sharing"
    )

    model_ids: set[str] = set()
    used_gpu_ids: set[int] = set()
    replicas: list[ModelReplica] = []
    for model_index, raw_model in enumerate(models):
        model = _require_mapping(raw_model, f"models[{model_index}]")
        model_id = _require_nonempty_string(
            model.get("model_id"), f"models[{model_index}].model_id"
        )
        if model_id in model_ids:
            raise MultiModelConfigError(
                f"models[{model_index}].model_id duplicates {model_id!r}"
            )
        model_ids.add(model_id)
        replicas.extend(
            _parse_model_replicas(
                model,
                model_index,
                used_gpu_ids,
                allow_gpu_sharing,
            )
        )

    model_resolver = _parse_model_resolver(
        config.get("model_resolver"),
        router_args=router_args,
        model_ids=model_ids,
    )

    return MultiModelConfig(
        router_args=dict(router_args),
        worker_host=worker_host,
        worker_base_port=worker_base_port,
        startup_timeout_secs=startup_timeout_secs,
        allow_gpu_sharing=allow_gpu_sharing,
        replicas=tuple(replicas),
        model_resolver=model_resolver,
    )


def load_multi_model_config(config_path: str | Path) -> MultiModelConfig:
    """Load and validate a JSON multi-model supervisor configuration file."""

    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError as exc:
        raise MultiModelConfigError(f"multi-model config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MultiModelConfigError(
            f"multi-model config file is not valid JSON: {path}: {exc.msg}"
        ) from exc
    return parse_multi_model_config(raw_config)
