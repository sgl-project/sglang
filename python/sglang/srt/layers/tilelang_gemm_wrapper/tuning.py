"""Shared tuning helpers for TileLang FP8 GEMM tooling."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

from sglang.srt.layers.tilelang_gemm_wrapper.configs import SelectedConfigStore


def concrete_shapes(
    nk_shapes: Iterable[Tuple[int, int]],
    m_values: Iterable[int],
) -> list[tuple[int, int, int]]:
    return [(M, N, K) for N, K in sorted(nk_shapes) for M in m_values]


def make_autotune_metadata(
    autotune_backend: str,
    autotune_policy: str,
    autotune_warmup: int,
    autotune_rep: int,
    autotune_max_configs: Optional[int],
    kernel_types: Optional[Iterable[str]],
    *,
    autotune: Optional[bool] = None,
) -> dict:
    metadata = {
        "autotune_backend": autotune_backend,
        "autotune_search_policy": autotune_policy,
        "autotune_warmup": autotune_warmup,
        "autotune_rep": autotune_rep,
        "autotune_max_configs": autotune_max_configs,
        "autotune_kernel_types": list(kernel_types) if kernel_types else None,
    }
    if autotune is not None:
        metadata["autotune"] = autotune
    return metadata


def load_selected_config_store(
    paths: Iterable[Optional[str]],
    *,
    skip_missing: bool = True,
) -> SelectedConfigStore:
    store = SelectedConfigStore()
    for path in paths:
        if not path:
            continue
        if skip_missing and not os.path.exists(path):
            continue
        store.update(SelectedConfigStore.from_path(path))
    return store


def warmup_tilelang_shapes(
    nk_shapes: Iterable[Tuple[int, int]],
    m_values: Iterable[int],
    config_path: Optional[str] = None,
    export_config_path: Optional[str] = None,
    autotune: bool = False,
    autotune_backend: str = "cudagraph",
    autotune_policy: str = "family_pruned",
    autotune_warmup: int = 25,
    autotune_rep: int = 100,
    autotune_max_configs: Optional[int] = None,
    kernel_types: Optional[Iterable[str]] = None,
    checkpoint_config_path: Optional[str] = None,
    resume_config_path: Optional[str] = None,
) -> None:
    from sglang.srt.layers import tilelang_gemm_wrapper

    if config_path:
        tilelang_gemm_wrapper.load_selected_configs(config_path)

    shapes = concrete_shapes(nk_shapes, m_values)
    if not shapes:
        raise RuntimeError("No TileLang GEMM shapes to warm up.")

    tilelang_gemm_wrapper.warmup_or_autotune_shapes(
        shapes,
        autotune=autotune,
        warmup=autotune_warmup,
        rep=autotune_rep,
        backend=autotune_backend,
        max_configs=autotune_max_configs,
        kernel_types=kernel_types,
        search_policy=autotune_policy,
        checkpoint_config_path=checkpoint_config_path,
        resume_config_path=resume_config_path,
    )

    if export_config_path:
        tilelang_gemm_wrapper.export_selected_configs(
            export_config_path,
            metadata=make_autotune_metadata(
                autotune_backend,
                autotune_policy,
                autotune_warmup,
                autotune_rep,
                autotune_max_configs,
                kernel_types,
                autotune=autotune,
            ),
        )
