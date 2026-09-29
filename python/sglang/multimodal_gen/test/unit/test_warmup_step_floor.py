"""Warmup requests must stay valid for models with a step floor, and a failed
request-based warmup must not be reported as a warmed-up timing."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
    DiffGenerator,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


def _req(sampling_params: SamplingParams, steps: int) -> Req:
    req = Req(sampling_params=sampling_params)
    req.num_inference_steps = steps
    return req


def test_generic_warmup_keeps_the_requested_steps():
    warmup = _req(SamplingParams(prompt="p"), 50).copy_as_warmup(1)
    assert warmup.num_inference_steps == 1
    assert warmup.extra["warmup_target_num_inference_steps"] == 50


def test_h3_warmup_is_raised_to_its_floor():
    assert MiniMaxH3SamplingParams.min_num_inference_steps == 2
    warmup = _req(MiniMaxH3SamplingParams(prompt="p"), 50).copy_as_warmup(1)
    assert warmup.num_inference_steps == 2
    assert warmup.extra["warmup_target_num_inference_steps"] == 50


def test_explicit_steps_above_the_floor_are_kept():
    warmup = _req(MiniMaxH3SamplingParams(prompt="p"), 50).copy_as_warmup(4)
    assert warmup.num_inference_steps == 4


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.metrics = None
    scheduler.return_result = Mock()
    scheduler._should_return_lightweight_warmup_result = lambda req: False
    scheduler._warmup_total = 1
    scheduler._warmup_processed = 0
    scheduler._warmup_progress_bar = None
    scheduler._show_warmup_progress = False
    scheduler._logged_server_ready_after_warmup = False
    return scheduler


def _request_warmup_req() -> Req:
    req = _req(SamplingParams(prompt="p"), 50).copy_as_warmup(1)
    req.request_id = "warmup-req"
    return req


def _output(error=None) -> OutputBatch:
    return OutputBatch(error=error, metrics=RequestMetrics("r"))


def test_failed_request_warmup_marks_only_the_next_request_cold():
    scheduler = _scheduler()
    scheduler._return_item_result((None, _request_warmup_req()), _output("boom"))

    cold, later = _output(), _output()
    scheduler._return_item_result((None, _req(SamplingParams(prompt="a"), 50)), cold)
    scheduler._return_item_result((None, _req(SamplingParams(prompt="b"), 50)), later)

    assert cold.metrics.warmup_failed is True
    assert cold.metrics.to_dict()["warmup_failed"] is True
    assert later.metrics.warmup_failed is False


def test_successful_request_warmup_leaves_the_request_warm():
    scheduler = _scheduler()
    scheduler._return_item_result((None, _request_warmup_req()), _output())

    real = _output()
    scheduler._return_item_result((None, _req(SamplingParams(prompt="a"), 50)), real)
    assert real.metrics.warmup_failed is False


def _summary_log(caplog, warmup_failed: bool) -> list[logging.LogRecord]:
    generator = DiffGenerator.__new__(DiffGenerator)
    generator.server_args = SimpleNamespace(warmup_mode="request")
    result = SimpleNamespace(
        metrics={"total_duration_ms": 80_000.0, "warmup_failed": warmup_failed},
        peak_memory_mb=None,
    )
    with caplog.at_level(logging.INFO):
        generator._log_summary([result])
    return caplog.records


def test_summary_does_not_claim_a_warm_timing_after_a_failed_warmup(caplog):
    records = _summary_log(caplog, warmup_failed=True)
    text = " ".join(r.getMessage() for r in records)
    assert "ran cold" in text
    assert "Warmed-up request processed" not in text
    assert any(r.levelno == logging.WARNING for r in records)


def test_summary_reports_the_warm_timing_after_a_successful_warmup(caplog):
    records = _summary_log(caplog, warmup_failed=False)
    text = " ".join(r.getMessage() for r in records)
    assert "Warmed-up request processed" in text
    assert "ran cold" not in text
