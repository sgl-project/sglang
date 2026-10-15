"""The scheduler drops kept encoder rows once no request is mid chunked prefill.

mm_schedule keeps the encoder rows that later prefill chunks of a request still
read. These tests check where the scheduler drops them: before each prefill
admission when no request is between two chunks, and in pause_generation,
flush_cache and release_memory_occupation.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import (
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_parallel, publish, reset_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def kept_rows():
    """One kept entry, as left by the chunk before."""
    mm_schedule.release_prefill_rows()
    mm_schedule._prefill_rows[("item", ((0, 9),))] = object()
    yield mm_schedule._prefill_rows
    mm_schedule.release_prefill_rows()


def _admission_scheduler(chunked_req):
    """A scheduler whose admission records how many entries it found."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = chunked_req
    scheduler.prefill_delayer = None
    scheduler.entries_at_admission = []

    def admit(prefill_delayer_single_pass, running_batch):
        scheduler.entries_at_admission.append(len(mm_schedule._prefill_rows))
        return None, running_batch

    scheduler._get_new_batch_prefill_raw = admit
    return scheduler


def test_rows_stay_while_a_request_is_mid_prefill_and_go_after():
    scheduler = _admission_scheduler(chunked_req=object())
    scheduler.get_new_batch_prefill(running_batch=None)

    # An abort or the last chunk leaves no chunked request for the next step.
    scheduler.chunked_req = None
    scheduler.get_new_batch_prefill(running_batch=None)

    assert scheduler.entries_at_admission == [1, 0]


def test_last_chunk_still_reads_its_rows(kept_rows):
    scheduler = _admission_scheduler(chunked_req=object())

    def last_chunk(prefill_delayer_single_pass, running_batch):
        # add_chunked_req returns None for a request's last chunk, whose
        # forward in this step still reads the kept rows.
        scheduler.chunked_req = None
        return None, running_batch

    scheduler._get_new_batch_prefill_raw = last_chunk
    scheduler.get_new_batch_prefill(running_batch=None)

    assert len(kept_rows) == 1


def test_bootstrap_failure_releases_rows_before_admission():
    scheduler = _admission_scheduler(chunked_req=object())
    scheduler.scheduler_stage_metrics = None
    scheduler.process_pending_chunked_abort = Mock()
    scheduler._process_hicache_events = Mock()
    scheduler.resolve_waiting_queue_bootstrap = Mock()

    def bootstrap_failed(last_batch, running_batch):
        # process_prefill_chunk drops a chunked request whose bootstrap failed.
        scheduler.chunked_req = None

    scheduler.process_prefill_chunk = Mock(side_effect=bootstrap_failed)
    scheduler.dp_attn_adapter = Mock()
    scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None

    scheduler.get_next_disagg_prefill_batch_to_run(
        running_batch=SimpleNamespace(batch_is_full=False), last_batch=None
    )

    assert scheduler.entries_at_admission == [0]


def _pause_scheduler(disaggregation_mode=DisaggregationMode.NULL):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.enable_overlap = False
    scheduler.enable_overlap_mlx = False
    scheduler.last_batch = None
    scheduler.cur_batch_for_debug = None
    scheduler.running_batch = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.disaggregation_mode = disaggregation_mode
    scheduler.decode_offload_manager = None
    scheduler.hisparse_coordinator = None
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.waiting_queue = []
    scheduler._add_request_to_queue = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
    scheduler.kv_events_publisher = MagicMock()
    scheduler.chunked_req = MagicMock()
    scheduler.chunked_req.finished.return_value = False
    return scheduler


@pytest.fixture
def published_server_args():
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="test")
    yield
    reset_context()


@pytest.mark.parametrize(
    "mode, disaggregation_mode, kept",
    [
        ("retract", DisaggregationMode.NULL, 0),
        # The live chunked request stays and resumes after the pause.
        ("retract", DisaggregationMode.PREFILL, 1),
        ("in_place", DisaggregationMode.NULL, 1),
    ],
)
def test_pause_drops_rows_only_with_the_chunked_request(
    published_server_args, kept_rows, mode, disaggregation_mode, kept
):
    scheduler = _pause_scheduler(disaggregation_mode)

    with patch("sglang.srt.managers.scheduler.retract_all"):
        scheduler.pause_generation(PauseGenerationReqInput(mode=mode))

    assert len(kept_rows) == kept


def test_flush_cache_drops_rows(kept_rows):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.is_fully_idle = Mock(return_value=True)
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.grammar_manager = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.draft_worker = None

    assert scheduler.flush_cache(empty_cache=False)
    assert not kept_rows


def test_weights_only_memory_release_drops_rows(kept_rows):
    with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            memory_saver_adapter=Mock(),
            # This path never reaches flush_cache; a stub that keeps the rows
            # shows the release comes from the manager itself.
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(_engine_paused=False),
        )

    with patch("torch.get_device_module"):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
        )

    manager.flush_cache.assert_not_called()
    assert not kept_rows


@pytest.mark.parametrize("prefix_len, kept", [(4, 1), (0, 0)])
def test_dllm_context_keeps_rows_until_its_prefill_is_done(kept_rows, prefix_len, kept):
    # dLLM requests encode their context over several steps without chunked_req.
    scheduler = Scheduler.__new__(Scheduler)
    req = SimpleNamespace(
        dllm_phase=DllmReqPhase.STAGING_PREFILL, prefix_len=prefix_len
    )
    scheduler.dllm_manager = SimpleNamespace(waiting_queue=[req], staging_queue=[])

    scheduler._release_kept_mm_rows_dllm()

    assert len(kept_rows) == kept


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
