# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for expanding one request into per-output requests."""

import os
from copy import copy, deepcopy

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.srt.observability.trace import TraceReqContext


def normalize_output_seeds(
    seed: int | list[int],
    *,
    num_outputs_per_prompt: int,
    num_prompts: int = 1,
    prompt_index: int = 0,
) -> list[int]:
    """Return the seeds for one prompt's outputs."""
    if num_outputs_per_prompt <= 0:
        raise ValueError(
            f"num_outputs_per_prompt must be positive, got {num_outputs_per_prompt}"
        )

    if isinstance(seed, list):
        seeds = [int(item) for item in seed]
        total_outputs = num_outputs_per_prompt * num_prompts
        if len(seeds) == num_outputs_per_prompt:
            return seeds
        if len(seeds) == total_outputs:
            start = prompt_index * num_outputs_per_prompt
            return seeds[start : start + num_outputs_per_prompt]
        raise ValueError(
            "seed list length must match num_outputs_per_prompt "
            f"({num_outputs_per_prompt}) or total outputs ({total_outputs}), "
            f"got {len(seeds)}"
        )

    base_seed = int(seed)
    return [base_seed + i for i in range(num_outputs_per_prompt)]


def _with_output_index_suffix(output_file_name: str, output_index: int) -> str:
    base, ext = os.path.splitext(output_file_name)
    return f"{base}_{output_index}{ext}"


def _trace_ctx_for_output(
    req: Req,
    request_id: str | None,
    output_index: int,
    *,
    reuse_parent_trace_ctx: bool,
):
    trace_ctx = req.trace_ctx
    if reuse_parent_trace_ctx or output_index == 0 or not trace_ctx.tracing_enable:
        return trace_ctx

    output_trace_ctx = TraceReqContext(
        rid=request_id,
        module_name=trace_ctx.module_name,
        external_trace_header=trace_ctx.external_trace_header,
    )
    output_trace_ctx.trace_req_start()
    return output_trace_ctx


def expand_request_outputs(
    req: Req,
    *,
    num_prompts: int = 1,
    prompt_index: int = 0,
    reuse_parent_trace_ctx: bool = False,
    preserve_parent_metrics: bool = False,
) -> list[Req]:
    """Expand one request into independent per-output requests.

    Entry points use separate trace roots because they own and finish every
    expanded request scope. Sequential pipeline execution reuses the parent
    context because the executor owns only the parent request trace lifecycle.
    """
    num_outputs = int(req.num_outputs_per_prompt)
    seeds = normalize_output_seeds(
        req.seed,
        num_outputs_per_prompt=num_outputs,
        num_prompts=num_prompts,
        prompt_index=prompt_index,
    )

    if num_outputs == 1:
        req.seed = seeds[0]
        req.seeds = None
        req.generator = None
        req.sampling_params.refresh_request_extra_after_output_expansion(req)
        return [req]

    expanded: list[Req] = []
    for output_index, seed in enumerate(seeds):
        output_request_id = (
            f"{req.request_id}:{output_index}" if req.request_id is not None else None
        )
        output_metrics = deepcopy(req.metrics) if preserve_parent_metrics else None
        output_req = copy(req)
        output_req.sampling_params = copy(req.sampling_params)
        output_req.extra = dict(req.extra)
        output_req.condition_inputs = dict(req.condition_inputs)
        output_req.trace_ctx = _trace_ctx_for_output(
            req,
            output_request_id,
            output_index,
            reuse_parent_trace_ctx=reuse_parent_trace_ctx,
        )
        output_req.seed = seed
        output_req.num_outputs_per_prompt = 1
        output_req.seeds = None
        output_req.generator = None
        output_req.extra["parent_request_id"] = req.request_id
        output_req.extra["output_index"] = output_index

        if output_request_id is not None:
            output_req.request_id = output_request_id

        if req.output_file_name:
            output_req.output_file_name = _with_output_index_suffix(
                req.output_file_name, output_index
            )
        output_req.sampling_params.refresh_request_extra_after_output_expansion(
            output_req
        )
        output_req.validate()
        if output_metrics is not None:
            output_req.metrics = output_metrics
            output_req.metrics.request_id = output_req.request_id
        expanded.append(output_req)

    return expanded
