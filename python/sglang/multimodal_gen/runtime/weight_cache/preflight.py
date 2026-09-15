# SPDX-License-Identifier: Apache-2.0
"""Launcher admission before starting workers or importing any CUDA handles."""

from sglang.multimodal_gen.runtime.pipelines_core import resolve_pipeline_class
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.weight_cache.client import WeightCacheClient
from sglang.multimodal_gen.runtime.weight_cache.identity import compatibility_plan


def preflight(args):
    prepared = prepare_pipeline(resolve_pipeline_class(args), args, required=True)
    plan = compatibility_plan(prepared, args)
    with WeightCacheClient(plan, args) as client:
        generation, _ = client.manifest()
    args._prepared_pipeline = prepared
    args._weight_cache_admission = (plan, generation)
