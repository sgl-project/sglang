"""Tenant directory helpers for NIXL FILE HiCache storage."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from hashlib import sha1
from typing import Optional

logger = logging.getLogger(__name__)

STORAGE_LAYOUT_VERSION = "v1"
_TENANT_DIR_RE = re.compile(r".+-v[0-9]+$")
_UNSAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._=-]+")
_MAX_COMPONENT_LEN = 128


def resolve_tenant_key(
    model_name: Optional[str],
    tp_size: int,
    num_disks: int = 1,
    explicit_tenant: Optional[str] = None,
) -> str:
    """Return a stable directory key for one NIXL FILE cache tenant."""
    name = _safe_component(explicit_tenant or model_name or "unnamed")
    return f"{name}-tp{tp_size}-n{num_disks}-{STORAGE_LAYOUT_VERSION}"


def prepare_tenant_storage_dirs(
    base_dirs: list[str],
    tenant_key: str,
    *,
    tp_rank: int,
    force_clean_all: bool,
    run_id: Optional[str] = None,
) -> tuple[list[str], Optional[threading.Thread]]:
    """Create tenant storage dirs and start best-effort cleanup on rank 0."""
    tenant_dirs = [os.path.join(base_dir, tenant_key) for base_dir in base_dirs]
    cleanup_thread = None

    if tp_rank == 0:
        cleanup_thread = start_tenant_cleanup(
            base_dirs,
            tenant_key,
            force_clean_all=force_clean_all,
        )
        if force_clean_all and cleanup_thread is not None:
            cleanup_thread.join()

        for tenant_dir in tenant_dirs:
            os.makedirs(tenant_dir, exist_ok=True)
        _write_tenant_markers(tenant_dirs, tenant_key, run_id)

    return tenant_dirs, cleanup_thread


def start_tenant_cleanup(
    base_dirs: list[str],
    tenant_key: str,
    *,
    force_clean_all: bool = False,
) -> Optional[threading.Thread]:
    """Async-delete stale NIXL tenant dirs below configured storage bases."""
    targets = _collect_cleanup_targets(base_dirs, tenant_key, force_clean_all)
    if not targets:
        return None

    thread = threading.Thread(
        target=_run_cleanup,
        args=(targets,),
        name="hicache-nixl-tenant-cleanup",
        daemon=True,
    )
    thread.start()
    logger.info(
        "NIXL tenant cleanup started: %d dir(s), tenant=%s, force_clean_all=%s",
        len(targets),
        tenant_key,
        force_clean_all,
    )
    return thread


def _safe_component(value: str) -> str:
    value = value.strip().strip("/")
    value = _UNSAFE_COMPONENT_RE.sub("-", value).strip("-")
    if not value:
        value = "unnamed"
    if len(value) <= _MAX_COMPONENT_LEN:
        return value
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[: _MAX_COMPONENT_LEN - len(digest) - 1]}-{digest}"


def _collect_cleanup_targets(
    base_dirs: list[str], tenant_key: str, force_clean_all: bool
) -> list[str]:
    targets: list[str] = []
    for base_dir in base_dirs:
        try:
            entries = list(os.scandir(base_dir))
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning(
                "NIXL tenant cleanup failed to scan %s", base_dir, exc_info=True
            )
            continue

        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not is_dir or _TENANT_DIR_RE.fullmatch(entry.name) is None:
                continue
            if not os.path.exists(os.path.join(entry.path, ".sglang-nixl-tenant")):
                continue
            if not force_clean_all and entry.name == tenant_key:
                continue
            targets.append(entry.path)
    return targets


def _run_cleanup(targets: list[str]) -> None:
    start = time.perf_counter()
    deleted = 0
    for target in targets:
        try:
            shutil.rmtree(target)
            deleted += 1
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning(
                "NIXL tenant cleanup failed to delete %s", target, exc_info=True
            )
    logger.info(
        "NIXL tenant cleanup done: deleted %d/%d dir(s) in %.1fs",
        deleted,
        len(targets),
        time.perf_counter() - start,
    )


def _write_tenant_markers(
    tenant_dirs: list[str], tenant_key: str, run_id: Optional[str]
) -> None:
    marker = f"tenant={tenant_key}\n"
    if run_id:
        marker += f"run_id={run_id}\n"

    for tenant_dir in tenant_dirs:
        try:
            with open(os.path.join(tenant_dir, ".sglang-nixl-tenant"), "w") as f:
                f.write(marker)
        except OSError:
            logger.warning(
                "Failed to write NIXL tenant marker in %s", tenant_dir, exc_info=True
            )
