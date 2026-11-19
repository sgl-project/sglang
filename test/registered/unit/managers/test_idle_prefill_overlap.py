"""CPU regression tests for automatic early idle-prefill results on gfx942."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(
    *,
    mode=ForwardMode.EXTEND,
    last_chunk=True,
    grammar=False,
    requests=1,
    tokens=128,
    prefix=0,
    spec=SpeculativeAlgorithm.NONE,
):
    return ScheduleBatch(
        reqs=[SimpleNamespace(inflight_steps=0) for _ in range(requests)],
        forward_mode=mode,
        contains_last_prefill_chunk=last_chunk,
        extend_num_tokens=tokens,
        prefix_lens=[prefix] * requests,
        has_grammar=grammar,
        spec_algorithm=SpeculativeAlgorithm.NGRAM if grammar else spec,
    )


def _model_config(
    *,
    model_type="deepseek_v2",
    hidden_size=2048,
    layers=27,
    experts=64,
    topk=6,
    dtype=torch.bfloat16,
    quantization=None,
):
    return SimpleNamespace(
        dtype=dtype,
        quantization=quantization,
        hf_config=SimpleNamespace(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            n_routed_experts=experts,
            num_experts_per_tok=topk,
        ),
    )


@contextmanager
def _platform_settings(*, gfx942=True, prefill_backend=Backend.DISABLED):
    with (
        patch.object(scheduler_module, "is_gfx942_supported", return_value=gfx942),
        patch.object(
            scheduler_module,
            "check_cuda_graph_backend",
            side_effect=lambda phase, backend: (
                phase == Phase.PREFILL and backend == prefill_backend
            ),
        ),
    ):
        yield


def _run_loop(
    *,
    steps,
    gfx942=True,
    prefill_backend=Backend.DISABLED,
    global_policy=False,
    pause_at=None,
    model_config=None,
    is_generation=True,
):
    """Run the production overlap loop and launch bookkeeping without a GPU."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.gracefully_exit = False
    scheduler.last_batch = None
    scheduler.running_batch = None
    scheduler.require_mlp_sync = False
    scheduler.model_config = _model_config() if model_config is None else model_config
    scheduler.is_generation = is_generation
    scheduler.launch_batch_sample_if_needed = Mock()
    scheduler.enable_unified_memory = False
    scheduler._sched_idled = False
    scheduler.forward_ct = 0
    scheduler.processed_tokens_counter = 0
    scheduler.scripted_scheduler_hook = None
    scheduler.token_to_kv_pool_allocator = None
    scheduler.scheduler_stage_metrics = None
    scheduler.metrics_reporter = Mock()
    scheduler.profiler_manager = Mock()
    scheduler.invariant_checker = Mock()
    scheduler.pause_state = SimpleNamespace(paused=False)
    scheduler.increment_scheduler_iterations = Mock()
    scheduler._apply_war_barrier = Mock()
    scheduler.on_idle = Mock()
    iterator = iter(steps)
    trace = []
    names = {}
    launched_names = {}
    iteration = 0

    def ingest():
        nonlocal iteration
        iteration += 1
        scheduler.pause_state.paused = iteration == pause_at

    def record_pause():
        trace.append(("pause", None))

    def schedule(**kwargs):
        item = next(iterator)
        if item is None:
            batch = None
        else:
            name, batch = item
            names[id(batch)] = name
        return NextBatchPlan(batch_to_run=batch, running_batch=None)

    def launch(batch):
        Scheduler._on_step_launch(scheduler, batch)
        name = names[id(batch)]
        launched_names[batch.forward_iter] = name
        trace.append(("launch", name))
        return SimpleNamespace()

    def process(batch, result):
        name = launched_names[batch.forward_iter]
        trace.append(("process", name))
        Scheduler._on_step_processed(scheduler, batch, result)

    scheduler.ingest_requests = ingest
    scheduler._record_scheduler_state_for_paused_engine = record_pause
    scheduler.get_next_batch_to_run = schedule
    scheduler.run_batch = launch
    scheduler.process_batch_result = process

    with (
        _platform_settings(gfx942=gfx942, prefill_backend=prefill_backend),
        envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(global_policy),
    ):
        try:
            scheduler.event_loop_overlap()
        except StopIteration:
            pass
    if scheduler.result_queue:
        raise AssertionError("The overlap loop left an unprocessed result")
    return trace


class TestIdlePrefillResultOrder(CustomTestCase):
    def test_other_architectures_keep_the_first_result_behind_the_next_launch(self):
        trace = _run_loop(
            steps=[None, ("A", _batch()), ("B", _batch()), None],
            gfx942=False,
        )
        self.assertEqual(
            trace,
            [("launch", "A"), ("launch", "B"), ("process", "A"), ("process", "B")],
        )

    def test_only_the_idle_first_prefill_is_early(self):
        trace = _run_loop(
            steps=[None, ("A", _batch()), ("B", _batch()), ("C", _batch()), None],
            gfx942=True,
        )
        self.assertEqual(
            trace,
            [
                ("launch", "A"),
                ("process", "A"),
                ("launch", "B"),
                ("launch", "C"),
                ("process", "B"),
                ("process", "C"),
            ],
        )

    def test_continuous_prefill_keeps_overlap(self):
        trace = _run_loop(steps=[("A", _batch()), ("B", _batch()), None], gfx942=True)
        self.assertEqual(
            trace,
            [("launch", "A"), ("launch", "B"), ("process", "A"), ("process", "B")],
        )

    def test_a_new_idle_gap_rearms_the_policy(self):
        trace = _run_loop(
            steps=[
                None,
                ("A", _batch()),
                None,
                None,
                ("B", _batch()),
                ("C", _batch()),
                None,
            ],
            gfx942=True,
        )
        self.assertEqual(
            trace,
            [
                ("launch", "A"),
                ("process", "A"),
                ("launch", "B"),
                ("process", "B"),
                ("launch", "C"),
                ("process", "C"),
            ],
        )

    def test_global_policy_still_processes_every_prefill_early(self):
        trace = _run_loop(
            steps=[None, ("A", _batch()), ("B", _batch()), ("C", _batch()), None],
            gfx942=True,
            global_policy=True,
        )
        self.assertEqual(
            trace,
            [
                ("launch", "A"),
                ("process", "A"),
                ("launch", "B"),
                ("process", "B"),
                ("launch", "C"),
                ("process", "C"),
            ],
        )

    def test_non_prefill_steps_keep_overlap(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.IDLE,
            ForwardMode.DLLM_EXTEND,
        ):
            with self.subTest(mode=mode):
                trace = _run_loop(
                    steps=[None, ("A", _batch(mode=mode)), ("B", _batch()), None],
                    gfx942=True,
                )
                self.assertLess(
                    trace.index(("launch", "B")), trace.index(("process", "A"))
                )

    def test_intermediate_chunk_keeps_overlap(self):
        trace = _run_loop(
            steps=[None, ("A", _batch(last_chunk=False)), ("B", _batch()), None],
            gfx942=True,
        )
        self.assertLess(trace.index(("launch", "B")), trace.index(("process", "A")))

    def test_mixed_prefill_keeps_overlap(self):
        trace = _run_loop(
            steps=[None, ("A", _batch(mode=ForwardMode.MIXED)), ("B", _batch()), None],
            gfx942=True,
        )
        self.assertLess(trace.index(("launch", "B")), trace.index(("process", "A")))

    def test_pause_preserves_the_pending_result_and_idle_marker(self):
        trace = _run_loop(
            steps=[None, ("A", _batch()), ("B", _batch()), None],
            gfx942=True,
            pause_at=3,
        )
        self.assertEqual(
            trace,
            [
                ("launch", "A"),
                ("pause", None),
                ("process", "A"),
                ("launch", "B"),
                ("process", "B"),
            ],
        )

    def test_grammar_barrier_remains_early_without_an_idle_gap(self):
        trace = _run_loop(
            steps=[
                ("A", _batch(mode=ForwardMode.DECODE, grammar=True)),
                ("B", _batch()),
                None,
            ],
            gfx942=True,
        )
        self.assertLess(trace.index(("process", "A")), trace.index(("launch", "B")))

    def test_prefill_graph_backends_keep_overlap(self):
        for backend in (Backend.BREAKABLE, Backend.FULL):
            with self.subTest(backend=backend):
                trace = _run_loop(
                    steps=[None, ("A", _batch()), ("B", _batch()), None],
                    prefill_backend=backend,
                )
                self.assertLess(
                    trace.index(("launch", "B")), trace.index(("process", "A"))
                )

    def test_multiple_request_prefills_keep_overlap(self):
        for requests in (2, 4):
            with self.subTest(requests=requests):
                trace = _run_loop(
                    steps=[
                        None,
                        ("A", _batch(requests=requests)),
                        ("B", _batch()),
                        None,
                    ],
                )
                self.assertLess(
                    trace.index(("launch", "B")), trace.index(("process", "A"))
                )

    def test_prefix_cache_hits_keep_overlap(self):
        trace = _run_loop(
            steps=[None, ("A", _batch(prefix=64)), ("B", _batch()), None],
        )
        self.assertLess(trace.index(("launch", "B")), trace.index(("process", "A")))

    def test_token_shape_boundaries(self):
        for tokens, early in (
            (1, True),
            (128, True),
            (512, True),
            (2048, True),
            (0, False),
            (2049, False),
            (4096, False),
            (None, False),
        ):
            with self.subTest(tokens=tokens):
                trace = _run_loop(
                    steps=[None, ("A", _batch(tokens=tokens)), ("B", _batch()), None],
                )
                self.assertEqual(
                    trace.index(("process", "A")) < trace.index(("launch", "B")),
                    early,
                )

    def test_speculative_prefill_keeps_overlap(self):
        for spec in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.NGRAM):
            with self.subTest(spec=spec):
                trace = _run_loop(
                    steps=[None, ("A", _batch(spec=spec)), ("B", _batch()), None],
                )
                self.assertLess(
                    trace.index(("launch", "B")), trace.index(("process", "A"))
                )

    def test_non_generation_batches_keep_overlap(self):
        trace = _run_loop(
            steps=[None, ("A", _batch()), ("B", _batch()), None],
            is_generation=False,
        )
        self.assertLess(trace.index(("launch", "B")), trace.index(("process", "A")))

    def test_unvalidated_model_shapes_and_dtypes_keep_overlap(self):
        for kwargs in (
            {"model_type": "qwen3"},
            {"hidden_size": 5120},
            {"layers": 60},
            {"experts": 160},
            {"topk": 8},
            {"quantization": "fp8"},
            {"dtype": torch.float16},
            {"dtype": torch.float32},
        ):
            with self.subTest(config=kwargs):
                trace = _run_loop(
                    steps=[None, ("A", _batch()), ("B", _batch()), None],
                    model_config=_model_config(**kwargs),
                )
                self.assertLess(
                    trace.index(("launch", "B")), trace.index(("process", "A"))
                )

    def test_global_policy_still_applies_outside_the_automatic_shapes(self):
        trace = _run_loop(
            steps=[
                None,
                ("A", _batch(requests=4, tokens=4096, prefix=128)),
                ("B", _batch()),
                None,
            ],
            gfx942=False,
            prefill_backend=Backend.FULL,
            global_policy=True,
        )
        self.assertLess(trace.index(("process", "A")), trace.index(("launch", "B")))


class TestIdlePrefillEligibility(CustomTestCase):
    def test_rank_local_idle_state_does_not_select_an_mlp_sync_boundary(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.require_mlp_sync = True
        with (
            _platform_settings(),
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
        ):
            for idle in (False, True):
                batch = _batch()
                batch.after_idle_gap = idle
                self.assertFalse(scheduler.should_process_last_batch_early(batch))

    def test_global_policy_uses_the_synchronized_extend_flag(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.require_mlp_sync = True
        with (
            _platform_settings(),
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(True),
        ):
            for local_mode, synchronized, expected in (
                (ForwardMode.IDLE, True, True),
                (ForwardMode.EXTEND, False, False),
            ):
                batch = _batch(mode=local_mode)
                batch.is_extend_in_batch = synchronized
                batch.after_idle_gap = True
                self.assertEqual(
                    scheduler.should_process_last_batch_early(batch), expected
                )

    def test_no_pending_batch_does_not_process_a_result(self):
        scheduler = Scheduler.__new__(Scheduler)
        with _platform_settings():
            self.assertFalse(scheduler.should_process_last_batch_early(None))

    def test_snapshot_preserves_the_idle_marker(self):
        original = _batch()
        original.after_idle_gap = True
        snapshot = original.copy()
        original.after_idle_gap = False
        self.assertTrue(snapshot.after_idle_gap)


if __name__ == "__main__":
    unittest.main()
