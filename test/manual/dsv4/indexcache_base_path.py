"""Shared base-path validation helpers for DSV4 IndexCache manual tools."""

from __future__ import annotations

import requests


def _as_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def speculative_config_paths(value, path: str = "") -> list[str]:
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key == "speculative_algorithm" and item:
                paths.append(f"{child_path}={item}")
            elif key in {
                "speculative_num_steps",
                "speculative_eagle_topk",
                "speculative_num_draft_tokens",
            }:
                int_value = _as_int(item)
                if int_value is not None and int_value > 0:
                    paths.append(f"{child_path}={item}")
            elif key == "enable_multi_layer_eagle" and item:
                paths.append(f"{child_path}={item}")
            paths.extend(speculative_config_paths(item, child_path))
        return paths
    if isinstance(value, list):
        paths = []
        for i, item in enumerate(value):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            paths.extend(speculative_config_paths(item, child_path))
        return paths
    return []


def validate_server_info_for_base_path(
    server_info: dict, context: str = "server /server_info"
) -> list[str]:
    speculative_paths = speculative_config_paths(server_info)
    if speculative_paths:
        raise RuntimeError(
            f"{context} reports speculative decoding enabled; "
            "disable EAGLE/spec decode for base-path IndexCache validation: "
            + ", ".join(speculative_paths)
        )
    return speculative_paths


def fetch_server_info(base_url: str, timeout: int) -> dict:
    response = requests.get(base_url.rstrip("/") + "/server_info", timeout=timeout)
    response.raise_for_status()
    return response.json()
