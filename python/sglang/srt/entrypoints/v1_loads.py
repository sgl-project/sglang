# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
/v1/loads API endpoint for comprehensive load metrics.

This module provides the /v1/loads endpoint which returns detailed scheduler
metrics for load balancing, monitoring, and capacity planning.
"""

import dataclasses
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.version import __version__

router = APIRouter()

_OPTIONAL_METRIC_SECTIONS = {
    "memory": ("memory", MemoryMetrics),
    "speculative": ("spec", SpeculativeMetrics),
    "lora": ("lora", LoRAMetrics),
    "disaggregation": ("disagg", DisaggregationMetrics),
    "queues": ("queues", QueueMetrics),
}


def _get_tokenizer_manager():
    """Dependency to get tokenizer_manager from global state."""
    from sglang.srt.entrypoints.http_server import get_global_state

    return get_global_state().tokenizer_manager


def _loads_dict_factory(items):
    """Factory for dataclasses.asdict() that excludes None values and timestamp."""
    return {k: v for k, v in items if v is not None and k != "timestamp"}


def _compute_aggregate(load_dicts: list) -> dict:
    """Compute aggregate metrics from load dicts."""
    if not load_dicts:
        return {
            "total_running_reqs": 0,
            "total_waiting_reqs": 0,
            "total_reqs": 0,
            "avg_token_usage": 0.0,
            "avg_throughput": 0.0,
            "avg_utilization": 0.0,
        }

    n = len(load_dicts)
    return {
        "total_running_reqs": sum(d["num_running_reqs"] for d in load_dicts),
        "total_waiting_reqs": sum(d["num_waiting_reqs"] for d in load_dicts),
        "total_reqs": sum(
            d["num_running_reqs"] + d["num_waiting_reqs"] for d in load_dicts
        ),
        "avg_token_usage": round(sum(d["token_usage"] for d in load_dicts) / n, 4),
        "avg_throughput": round(sum(d["gen_throughput"] for d in load_dicts) / n, 2),
        "avg_utilization": round(sum(d["utilization"] for d in load_dicts) / n, 4),
    }


def _format_loads_prometheus(load_results) -> Response:
    """Format load metrics in Prometheus text exposition format.

    Metrics are derived from dataclass field metadata, providing a single source of truth.
    """
    lines = []

    for f in dataclasses.fields(GetLoadsReqOutput):
        if "metric" not in f.metadata:
            continue
        metric_type, description = f.metadata["metric"]
        metric_name = f"sglang_{f.name}"
        lines.append(f"# HELP {metric_name} {description}")
        lines.append(f"# TYPE {metric_name} {metric_type}")
        for load in load_results:
            value = getattr(load, f.name, None)
            if value is not None:
                lines.append(f'{metric_name}{{dp_rank="{load.dp_rank}"}} {value}')

    for attr_name, (prefix, dataclass_type) in _OPTIONAL_METRIC_SECTIONS.items():
        if not any(getattr(load, attr_name, None) for load in load_results):
            continue
        for f in dataclasses.fields(dataclass_type):
            if "metric" not in f.metadata:
                continue
            metric_type, description = f.metadata["metric"]
            metric_name = f"sglang_{prefix}_{f.name}"
            lines.append(f"# HELP {metric_name} {description}")
            lines.append(f"# TYPE {metric_name} {metric_type}")
            for load in load_results:
                section = getattr(load, attr_name, None)
                if section:
                    value = getattr(section, f.name, None)
                    if value is not None:
                        lines.append(
                            f'{metric_name}{{dp_rank="{load.dp_rank}"}} {value}'
                        )

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/v1/loads")
async def get_loads(
    dp_rank: Optional[int] = None,
    include: Optional[str] = None,
    format: Optional[str] = None,
    tokenizer_manager=Depends(_get_tokenizer_manager),
):
    """
    Get comprehensive load metrics for all DP ranks.

    Query Parameters:
        dp_rank: Filter to specific DP rank (optional)
        include: Comma-separated sections to include (optional)
                 Options: core, memory, spec, lora, disagg, queues, all
                 Default: all
        format: Response format - 'json' (default) or 'prometheus'

    Returns:
        JSON response with timestamp, version, dp_rank_count, per-DP-rank loads, and aggregates
    """
    include_list = [s.strip() for s in include.split(",")] if include else None

    try:
        load_results = await tokenizer_manager.get_loads(
            include=include_list,
            dp_rank=dp_rank,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if format == "prometheus":
        return _format_loads_prometheus(load_results)

    loads = []
    for load in load_results:
        d = dataclasses.asdict(load, dict_factory=_loads_dict_factory)
        d["num_total_reqs"] = d["num_running_reqs"] + d["num_waiting_reqs"]
        loads.append(d)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "dp_rank_count": len(loads),
        "loads": loads,
        "aggregate": _compute_aggregate(loads),
    }
