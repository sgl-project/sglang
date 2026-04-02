import logging

from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestMetrics,
    StageProfiler,
)


def test_stage_profiler_appends_steps_across_disjoint_loops():
    metrics = RequestMetrics(request_id="req")
    logger = logging.getLogger(__name__)

    with StageProfiler(
        "ltx2_stage1_step_0",
        logger=logger,
        metrics=metrics,
        perf_dump_path_provided=True,
        record_as_step=True,
    ):
        pass

    with StageProfiler(
        "ltx2_stage1_step_1",
        logger=logger,
        metrics=metrics,
        perf_dump_path_provided=True,
        record_as_step=True,
    ):
        pass

    with StageProfiler(
        "ltx2_stage2_step_0",
        logger=logger,
        metrics=metrics,
        perf_dump_path_provided=True,
        record_as_step=True,
    ):
        pass

    assert len(metrics.steps) == 3
    assert metrics.stages == {}


def test_stage_profiler_keeps_legacy_denoising_step_name_detection():
    metrics = RequestMetrics(request_id="req")
    logger = logging.getLogger(__name__)

    with StageProfiler(
        "denoising_step_0",
        logger=logger,
        metrics=metrics,
        perf_dump_path_provided=True,
    ):
        pass

    assert len(metrics.steps) == 1
