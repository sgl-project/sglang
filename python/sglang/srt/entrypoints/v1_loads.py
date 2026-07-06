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

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from sglang.version import __version__

router = APIRouter()


def _get_tokenizer_manager():
    """Dependency to get tokenizer_manager from global state."""
    from sglang.srt.entrypoints.http_server import get_global_state

    return get_global_state().tokenizer_manager


def _format_loads_prometheus(load_results, include=None) -> Response:
    """Format load metrics in Prometheus text exposition format."""
    section_prefixes = {"speculative": "spec", "disaggregation": "disagg"}
    metric_samples = {}

    for load in load_results:
        load_dict = load.to_dict(include)
        dp_rank = load_dict.pop("dp_rank")

        for key, value in load_dict.items():
            if isinstance(value, dict):
                prefix = section_prefixes.get(key, key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metric_samples.setdefault(
                            f"sglang_{prefix}_{sub_key}", []
                        ).append((dp_rank, sub_value))
            elif isinstance(value, (int, float)):
                metric_samples.setdefault(f"sglang_{key}", []).append((dp_rank, value))

    lines = []
    for metric_name, samples in metric_samples.items():
        lines.append(f"# TYPE {metric_name} gauge")
        for dp_rank, value in samples:
            lines.append(f'{metric_name}{{dp_rank="{dp_rank}"}} {value}')

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
        JSON response with timestamp, version, and per-DP-rank loads
    """
    include_list = [s.strip() for s in include.split(",")] if include else None

    start = time.perf_counter()
    try:
        load_results = await tokenizer_manager.get_loads(
            include=include_list,
            dp_rank=dp_rank,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        mc = getattr(tokenizer_manager, "metrics_collector", None)
        if mc is not None:
            mc.get_loads_duration_seconds.labels(**mc.labels).observe(
                time.perf_counter() - start
            )

    include_set = set(include_list) if include_list else None

    if format == "prometheus":
        return _format_loads_prometheus(load_results, include_set)

    loads = []
    for load in load_results:
        d = load.to_dict(include_set)
        loads.append(d)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "loads": loads,
    }
