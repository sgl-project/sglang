# SPDX-License-Identifier: Apache-2.0
"""Launcher admission before starting workers or importing any CUDA handles."""

import json
import time

from sglang.multimodal_gen.runtime.pipelines_core import resolve_pipeline_class
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weight_cache.client import WeightCacheClient
from sglang.multimodal_gen.runtime.weight_cache.identity import compatibility_plan

logger = init_logger(__name__)


def preflight(args):
    start = time.perf_counter()
    prepared = prepare_pipeline(resolve_pipeline_class(args), args, required=True)
    resolved = time.perf_counter()
    plan = compatibility_plan(prepared, args)
    identified = time.perf_counter()
    with WeightCacheClient(plan, args) as client:
        generation, _ = client.manifest()
    args._prepared_pipeline = prepared
    args._weight_cache_admission = (plan, generation)
    admitted = time.perf_counter()
    logger.info(
        "[WeightCache] launcher admission stages: %s",
        json.dumps(
            {
                "prepare": resolved - start,
                "compatibility": identified - resolved,
                "manifest": admitted - identified,
                "total": admitted - start,
            },
            sort_keys=True,
        ),
    )
