"""GEMM configs keyed by provider, geometry, device, and nearest expected M."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Mapping

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PACKAGE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs"
)


class GemmTile(msgspec.Struct, frozen=True, kw_only=True):
    token_width: int
    persistent_clusters: int


class GemmConfigTable(msgspec.Struct, kw_only=True):
    buckets: dict[int, dict[str, int]]
    tiles: tuple[GemmTile, ...] = ()
    version: dict[str, str] = {}

    def pick(self, expected_m: int) -> dict[str, int]:
        return self.buckets[min(self.buckets, key=lambda m: abs(m - expected_m))]


def _validate(table: GemmConfigTable) -> None:
    if not table.buckets:
        raise ValueError("buckets must be non-empty")
    for bucket_m, payload in table.buckets.items():
        if bucket_m <= 0:
            raise ValueError(f"bucket key {bucket_m} must be positive")
        if not payload or any(value <= 0 for value in payload.values()):
            raise ValueError(f"bucket {bucket_m} payload must be positive ints")
    widths = [tile.token_width for tile in table.tiles]
    if len(set(widths)) != len(widths):
        raise ValueError("tiles must declare one persistent_clusters per token_width")
    for tile in table.tiles:
        if tile.token_width <= 0 or tile.persistent_clusters <= 0:
            raise ValueError("tile fields must be positive")


def config_file_name(
    provider_key: str,
    *,
    num_local_experts: int,
    n_gemm1: int,
    n_gemm2: int,
    k: int,
    device_name: str,
) -> str:
    device = device_name.replace(" ", "_")
    return (
        f"provider={provider_key},E={num_local_experts},N1={n_gemm1},"
        f"N2={n_gemm2},K={k},device_name={device}.json"
    )


@functools.lru_cache(maxsize=None)
def _load(
    path: str, expected_versions: tuple[tuple[str, str], ...]
) -> GemmConfigTable | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            table = msgspec.json.decode(f.read(), type=GemmConfigTable)
        _validate(table)
    except (msgspec.DecodeError, ValueError) as exc:
        logger.warning("Ignoring malformed MoE LoRA GEMM config %s: %s", path, exc)
        return None
    for name, expected in expected_versions:
        recorded = table.version.get(name)
        if recorded is not None and recorded != expected:
            logger.warning(
                "Ignoring MoE LoRA GEMM config %s: %s version %s does not "
                "match installed %s",
                path,
                name,
                recorded,
                expected,
            )
            return None
    logger.info("Using MoE LoRA GEMM config from %s.", path)
    return table


def load_config_table(
    provider_key: str,
    *,
    num_local_experts: int,
    n_gemm1: int,
    n_gemm2: int,
    k: int,
    device_name: str | None = None,
    expected_versions: Mapping[str, str] | None = None,
) -> GemmConfigTable | None:
    if device_name is None:
        from sglang.srt.utils import get_device_name

        device_name = get_device_name()
    name = config_file_name(
        provider_key,
        num_local_experts=num_local_experts,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        k=k,
        device_name=device_name,
    )
    versions = tuple(sorted((expected_versions or {}).items()))
    roots = (envs.SGLANG_LORA_MOE_CONFIG_DIR.get(), _PACKAGE_CONFIG_DIR)
    for root in filter(None, roots):
        path = os.path.join(root, "base_gemm", name)
        if os.path.isfile(path):
            return _load(path, versions)
    return None


def cutedsl_version() -> str | None:
    try:
        import cutlass
    except ImportError:
        return None
    return getattr(cutlass, "__version__", None)
