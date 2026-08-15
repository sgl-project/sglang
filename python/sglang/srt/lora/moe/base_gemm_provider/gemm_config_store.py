"""M-bucketed launch-config store for the MoE LoRA base GEMM providers.

Mirrors sglang's Triton fused-MoE JSON pattern
(``fused_moe_triton_config.py``): one JSON file per (provider, geometry,
device), keyed by ``expected_m`` buckets with nearest-M selection.  The
loader returns ``None`` whenever no valid table exists, and every consumer
then falls back to its built-in heuristic — behavior without config files is
byte-identical to today's.

File name::

    provider={key},E={E},N1={n1},N2={n2},K={k},device_name={name},dtype=bf16.json

Payload::

    {"version": {"cutedsl": "<pkg version>", "generated_on": "GB300-152SM"},
     "tiles":   [{"token_width": 64, "persistent_clusters": 128}, ...],
     "buckets": {"16": {"token_width": 64}, "96": {"token_width": 128}}}

``tiles`` (CuTeDSL only) declares the tile set to compile at attach, one
``persistent_clusters`` per ``token_width``.  Bucket payloads carry
``token_width`` for the CuTeDSL provider or ``expected_m`` for DeepGEMM.
Files are generated on the target device by
``benchmark/kernels/lora_moe/sweep_masked_gemm_configs.py``, never written
by hand.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Mapping

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class GemmTile(msgspec.Struct, frozen=True, kw_only=True):
    """One compiled-tile declaration for the CuTeDSL provider."""

    token_width: int
    persistent_clusters: int


class GemmConfigTable(msgspec.Struct, kw_only=True):
    """One provider+geometry+device table of expected_m-bucketed configs."""

    buckets: dict[int, dict[str, int]]
    tiles: tuple[GemmTile, ...] = ()
    version: dict[str, str] = {}

    def pick(self, expected_m: int) -> dict[str, int]:
        """Nearest-M bucket payload — same rule as the Triton fused-MoE
        ``try_get_optimal_moe_config`` lookup."""
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
        f"N2={n_gemm2},K={k},device_name={device},dtype=bf16.json"
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
    """Load the bucket table for one provider+geometry, or ``None``.

    ``None`` always means "use the built-in heuristics unchanged".  Tables
    live under ``base_gemm/`` inside the config root: the directory named by
    ``SGLANG_LORA_MOE_CONFIG_DIR`` when set, else the package-local
    ``lora/moe/configs/`` directory.
    """
    if device_name is None:
        from sglang.srt.utils import get_device_name

        device_name = get_device_name()
    root = envs.SGLANG_LORA_MOE_CONFIG_DIR.get() or os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs"
    )
    directory = os.path.join(root, "base_gemm")
    path = os.path.join(
        directory,
        config_file_name(
            provider_key,
            num_local_experts=num_local_experts,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            k=k,
            device_name=device_name,
        ),
    )
    return _load(path, tuple(sorted((expected_versions or {}).items())))


def cutedsl_version() -> str | None:
    """Installed CuTeDSL version for table version pinning, if importable."""
    try:
        import cutlass
    except ImportError:
        return None
    return getattr(cutlass, "__version__", None)
