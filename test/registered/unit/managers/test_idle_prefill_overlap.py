"""CPU regression tests for automatic early idle-prefill results on gfx942."""

import unittest
from array import array
from collections import deque
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
from sglang.srt.utils import device

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(
    *,
    requests=1,
    tokens=128,
    prefix=0,
    mode=ForwardMode.EXTEND,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    **kwargs,
):
    return ScheduleBatch(
        reqs=[SimpleNamespace(inflight_steps=0) for _ in range(requests)],
        forward_mode=mode,
        after_idle_gap=True,
        extend_num_tokens=tokens,
        prefix_lens=[prefix] * requests,
        spec_algorithm=spec_algorithm,
        **kwargs,
    )


class _EndOfSteps(Exception):
    pass


class TestIdlePrefillOverlap(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.device = "cuda"
        self.scheduler.tp_worker = SimpleNamespace(gpu_id=2)
        self.scheduler.require_mlp_sync = False
        self.scheduler.is_generation = True
        self.scheduler.model_config = SimpleNamespace(
            dtype=torch.bfloat16, quantization=None, is_encoder_decoder=False
        )
        self.enterContext(envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False))
        self.enterContext(patch.object(device, "is_hip", return_value=True))
        device.is_gfx942_supported.cache_clear()
        self.addCleanup(device.is_gfx942_supported.cache_clear)
        self.enterContext(
            patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect={
                    0: SimpleNamespace(gcnArchName="gfx942"),
                    1: SimpleNamespace(gcnArchName="gfx950"),
                    2: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
                }.__getitem__,
            )
        )
        self.prefill_backend = Backend.DISABLED
        self.enterContext(
            patch.object(
                scheduler_module,
                "check_cuda_graph_backend",
                side_effect=lambda phase, backend: (
                    phase == Phase.PREFILL and backend == self.prefill_backend
                ),
            )
        )

    def _run_loop(self, batches, *, pause_at=None):
        """Run the production loop and accounting hooks with scripted forwards."""
        scheduler = self.scheduler
        scheduler.gracefully_exit = False
        scheduler.last_batch = scheduler.running_batch = None
        scheduler.enable_unified_memory = False
        scheduler._sched_idled = False
        scheduler.forward_ct = scheduler.processed_tokens_counter = 0
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
        scheduler.launch_batch_sample_if_needed = Mock()
        pending = deque(batches)
        trace = []
        iteration = 0

        def ingest():
            nonlocal iteration
            iteration += 1
            scheduler.pause_state.paused = iteration == pause_at

        def schedule(**kwargs):
            if not pending:
                raise _EndOfSteps
            return NextBatchPlan(batch_to_run=pending.popleft(), running_batch=None)

        def launch(batch):
            scheduler._on_step_launch(batch)
            trace.append(("launch", batch.forward_iter))
            return SimpleNamespace()

        def process(batch, result):
            scheduler._on_step_processed(batch, result)
            trace.append(("process", batch.forward_iter))

        scheduler.ingest_requests = ingest
        scheduler._record_scheduler_state_for_paused_engine = lambda: trace.append(
            ("pause", None)
        )
        scheduler.get_next_batch_to_run = schedule
        scheduler.run_batch = launch
        scheduler.process_batch_result = process
        with self.assertRaises(_EndOfSteps):
            scheduler.event_loop_overlap()
        self.assertFalse(scheduler.result_queue)
        for batch in batches:
            if batch is not None:
                for req in batch.reqs:
                    self.assertEqual(req.inflight_steps, 0)
        return trace

    def test_only_first_prefill_after_idle_is_early(self):
        self.assertEqual(
            self._run_loop([None, _batch(), _batch(), _batch(), None]),
            [
                ("launch", 1),
                ("process", 1),
                ("launch", 2),
                ("launch", 3),
                ("process", 2),
                ("process", 3),
            ],
        )

    def test_idle_rearms_after_continuous_work_and_survives_pause(self):
        self.assertEqual(
            self._run_loop(
                [_batch(), _batch(), None, _batch(), _batch(), None], pause_at=5
            ),
            [
                ("launch", 1),
                ("launch", 2),
                ("process", 1),
                ("process", 2),
                ("launch", 3),
                ("pause", None),
                ("process", 3),
                ("launch", 4),
                ("process", 4),
            ],
        )

    def test_global_switch_processes_consecutive_prefills_on_other_devices(self):
        self.scheduler.tp_worker.gpu_id = 1
        with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(True):
            self.assertEqual(
                self._run_loop([_batch(requests=2), _batch(), _batch(), None]),
                [
                    ("launch", 1),
                    ("process", 1),
                    ("launch", 2),
                    ("process", 2),
                    ("launch", 3),
                    ("process", 3),
                ],
            )

    def test_batch_eligibility(self):
        select = self.scheduler.should_process_last_batch_early
        self.assertFalse(select(None))
        for tokens in (None, 0, 1, 2048, 2049, 4096, 4097):
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    select(_batch(tokens=tokens)), tokens in (1, 2048, 2049, 4096)
                )
        for kwargs in (
            {"requests": 2},
            {"prefix": 64},
            {"contains_last_prefill_chunk": False},
            {"spec_algorithm": SpeculativeAlgorithm.EAGLE},
            {"spec_algorithm": SpeculativeAlgorithm.NGRAM},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(select(_batch(**kwargs)))
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.IDLE,
            ForwardMode.DLLM_EXTEND,
            ForwardMode.MIXED,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(select(_batch(mode=mode)))

    def test_model_and_backend_eligibility(self):
        select = self.scheduler.should_process_last_batch_early
        config = self.scheduler.model_config
        for obj, field, value in (
            (config, "dtype", torch.float16),
            (config, "dtype", torch.float32),
            (config, "quantization", "fp8"),
            (self.scheduler, "is_generation", False),
            (self, "prefill_backend", Backend.BREAKABLE),
            (self, "prefill_backend", Backend.FULL),
        ):
            with (
                self.subTest(field=field, value=value),
                patch.object(obj, field, value),
            ):
                self.assertFalse(select(_batch()))

    def test_worker_device_selects_architecture(self):
        for gpu_id, expected in ((1, False), (2, True)):
            with self.subTest(gpu_id=gpu_id):
                self.scheduler.tp_worker.gpu_id = gpu_id
                self.assertEqual(
                    self.scheduler.should_process_last_batch_early(_batch()), expected
                )
        self.scheduler.device = "cpu"
        self.assertFalse(self.scheduler.should_process_last_batch_early(_batch()))

    def test_encoder_decoder_shapes_after_stripping_encoder_tokens(self):
        self.scheduler.model_config.is_encoder_decoder = True
        for encoder, prefix, new_tokens in ((4096, 0, 4104), (1024, 1024, 8)):
            with self.subTest(encoder=encoder, prefix=prefix):
                batch = _batch(tokens=new_tokens, prefix=prefix, device="cpu")
                req = batch.reqs[0]
                req.multimodal_inputs = SimpleNamespace(num_image_tokens=encoder)
                req.prefix_len, req.extend_len, req.logprob_start_len = (
                    prefix,
                    new_tokens,
                    -1,
                )
                batch.extend_lens = [new_tokens]
                batch.out_cache_loc = torch.arange(new_tokens)
                batch.prepare_encoder_info_extend(
                    [array("q", range(new_tokens))], [prefix + new_tokens]
                )
                self.assertEqual((batch.prefix_lens, batch.extend_num_tokens), ([0], 8))
                self.assertFalse(self.scheduler.should_process_last_batch_early(batch))

    def test_mlp_sync_uses_only_the_global_extend_flag(self):
        self.scheduler.require_mlp_sync = True
        select = self.scheduler.should_process_last_batch_early
        for idle in (False, True):
            batch = _batch()
            batch.after_idle_gap = idle
            self.assertFalse(select(batch))
        with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(True):
            self.assertTrue(
                select(_batch(mode=ForwardMode.IDLE, is_extend_in_batch=True))
            )
            self.assertFalse(select(_batch(is_extend_in_batch=False)))

    def test_grammar_barrier_without_idle(self):
        batch = _batch(
            mode=ForwardMode.DECODE,
            has_grammar=True,
            spec_algorithm=SpeculativeAlgorithm.NGRAM,
        )
        batch.after_idle_gap = False
        self.assertTrue(self.scheduler.should_process_last_batch_early(batch))


if __name__ == "__main__":
    unittest.main()
