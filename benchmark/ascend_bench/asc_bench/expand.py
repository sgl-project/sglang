"""Grid expansion: config → deterministic list of benchmark cells."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any

from asc_bench.config import BenchConfig


@dataclass
class Cell:
    """One fully resolved benchmark unit (a single server config x workload
    x repeat)."""

    cell_id: str
    cell_hash: str
    model: dict[str, Any]
    server_args: dict[str, Any]
    env: dict[str, str]
    workload_args: dict[str, Any]
    dataset_name: str
    concurrency: int | None
    num_prompts: int | None
    tp_size: int | None
    repeat: int
    meta: dict[str, Any] = field(default_factory=dict)


def _model_slug(path: str) -> str:
    base = path.rstrip("/").rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", base) or "model"


def _axis_combos(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product over axes, in sorted-key order for determinism.

    A single axis value may itself be a dict of ``{flag: value}`` to vary
    several flags together (e.g. speculative decoding bundles).
    """
    if not axes:
        return [{}]
    keys = sorted(axes)
    combos = []
    for values in itertools.product(*(axes[k] for k in keys)):
        merged: dict[str, Any] = {}
        for key, value in zip(keys, values):
            if isinstance(value, dict):
                merged.update(value)
            else:
                merged[key] = value
        combos.append(merged)
    return combos


def _canonical_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def expand_cells(cfg: BenchConfig) -> list[Cell]:
    """Expand the config grids into cells with stable ids, order, hashes."""
    slug = _model_slug(cfg.model.path)
    server_combos = _axis_combos(cfg.server.axes)
    workload_combos = _axis_combos(cfg.workload.axes)

    cells: list[Cell] = []
    for si, scombo in enumerate(server_combos):
        server_args = {**cfg.server.args, **scombo}
        for wi, wcombo in enumerate(workload_combos):
            workload_args = {**cfg.workload.args, **wcombo}
            concurrency = workload_args.get("--max-concurrency")
            concurrency = int(concurrency) if concurrency is not None else None
            mult = cfg.workload.num_prompts_mult
            num_prompts = mult * concurrency if (mult and concurrency) else None
            payload = {
                "model": {
                    "path": cfg.model.path,
                    "dtype": cfg.model.dtype,
                    "quantization": cfg.model.quantization,
                },
                "server_args": server_args,
                "server_env": cfg.server.env,
                "dataset_name": cfg.workload.dataset_name,
                "workload_args": workload_args,
                "num_prompts": num_prompts,
            }
            cell_hash = _canonical_hash(payload)
            for ri in range(cfg.run.repeats):
                cells.append(
                    Cell(
                        cell_id=f"{slug}__s{si:02d}__w{wi:02d}__r{ri}",
                        cell_hash=cell_hash,
                        model=payload["model"],
                        server_args=server_args,
                        env=dict(cfg.server.env),
                        workload_args=workload_args,
                        dataset_name=cfg.workload.dataset_name,
                        concurrency=concurrency,
                        num_prompts=num_prompts,
                        tp_size=server_args.get("--tp-size"),
                        repeat=ri,
                        meta={"server_combo": si, "workload_combo": wi},
                    )
                )
    return cells
