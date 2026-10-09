# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Validation and loading helpers for EP rank topology descriptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _validate_rank_cost_matrix(
    rank_cost_matrix: torch.Tensor, expected_num_ranks: int | None = None
) -> torch.Tensor:
    if (
        rank_cost_matrix.ndim != 2
        or rank_cost_matrix.shape[0] != rank_cost_matrix.shape[1]
    ):
        raise ValueError("rank_cost_matrix must be square")
    if rank_cost_matrix.shape[0] <= 0:
        raise ValueError("rank_cost_matrix must contain at least one rank")
    if expected_num_ranks is not None and rank_cost_matrix.shape[0] != (
        expected_num_ranks
    ):
        raise ValueError(
            "rank_cost_matrix size does not match the EP group: "
            f"matrix={rank_cost_matrix.shape[0]} expected={expected_num_ranks}"
        )

    rank_cost_matrix = rank_cost_matrix.to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(rank_cost_matrix).all() or torch.any(rank_cost_matrix < 0):
        raise ValueError("rank_cost_matrix must be finite and non-negative")
    if torch.any(torch.diagonal(rank_cost_matrix) != 0):
        raise ValueError("rank_cost_matrix diagonal must be zero")
    return rank_cost_matrix.contiguous()


def load_rank_cost_matrix(
    topology: str | Path | Mapping[str, Any],
    *,
    expected_num_ranks: int | None = None,
) -> torch.Tensor:
    """Load a rank cost matrix from a path or a JSON object.

    The JSON form is intentionally small and hardware-agnostic::

        {"rank_cost_matrix": [[0.0, 1.0], [1.0, 0.0]]}

    Values are relative costs; only their ordering affects placement.  The
    diagonal must be zero because sending to the local rank does not require
    inter-rank communication.
    """
    if isinstance(topology, (str, Path)):
        path = Path(topology)
        try:
            data = json.loads(path.read_text())
        except OSError as exc:
            raise ValueError(f"unable to read EPLB topology file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"EPLB topology file is not valid JSON: {path}") from exc
    else:
        data = topology

    if isinstance(data, Mapping):
        if "rank_cost_matrix" not in data:
            raise ValueError("topology JSON must contain rank_cost_matrix")
        data = data["rank_cost_matrix"]

    try:
        rank_cost_matrix = torch.as_tensor(data, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rank_cost_matrix must be a rectangular numeric array"
        ) from exc
    return _validate_rank_cost_matrix(rank_cost_matrix, expected_num_ranks)


__all__ = ["load_rank_cost_matrix"]
