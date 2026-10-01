"""Prepare a passive distributed collector with its own control communicator."""

from datetime import timedelta

import torch.distributed as dist

from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_coordinator import CohortCaptureCoordinator
from sglang.srt.training_capture.metrics import CaptureMetrics
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.resources import CaptureResources


def prepare_cohort_capture(
    *,
    startup_group,
    config,
    teacher,
    kv,
    layout,
    pool,
    req_to_token,
    enable_overlap,
    capture_mode,
    metrics_labels,
    timeout_seconds=120.0,
):
    """All startup-group members call after their common policy vote succeeds.

    Group creation precedes any fallible rank-local resource work. The caller
    votes preparation and activation on startup_group, never this communicator.
    """
    ranks = dist.get_process_group_ranks(startup_group)
    if ranks != list(range(dist.get_world_size())):
        raise ContractError(
            "capture startup requires the complete PP-major worker group"
        )
    control = dist.new_group(
        ranks=ranks,
        backend="gloo",
        timeout=timedelta(seconds=timeout_seconds),
    )
    resources = None
    try:
        coordinator_type = CohortCaptureCoordinator
        if capture_mode == "pd_autoregressive":
            from sglang.srt.training_capture.pd_capture import (
                CohortDecodeCaptureCoordinator,
            )

            coordinator_type = CohortDecodeCaptureCoordinator
        partition = layout.partitions[dist.get_rank(control)]
        resources = CaptureResources.prepare(
            config=config, kv=kv, partition=partition, source_pool=pool
        )
        if resources.exporter is not None and resources.exporter.device.type != "cuda":
            raise ContractError("serving capture currently requires CUDA")
        metrics = CaptureMetrics(metrics_labels) if metrics_labels is not None else None
        return coordinator_type(
            allocator=CaptureCohortAllocator(
                group=control,
                layout=layout,
                config=config,
                teacher=teacher,
                kv=kv,
                resources=resources,
                timeout_seconds=timeout_seconds,
            ),
            req_to_token=req_to_token,
            capture_mode=capture_mode,
            enable_overlap=enable_overlap,
            metrics=metrics,
            autostart=False,
            owns_control_group=True,
        )
    except Exception:
        try:
            if resources is not None:
                resources.close()
        finally:
            dist.destroy_process_group(control)
        raise
