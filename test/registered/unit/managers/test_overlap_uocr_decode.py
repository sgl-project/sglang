import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
    UnlimitedOCRV2AttnBackend,
)
from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.uocr_doc_hooks import UocrDocHooks
from sglang.srt.model_executor.model_runner_components.uocr_greedy_graph import (
    UocrGreedyGraph,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.uocr_decode_graph_io import UOCRDecodeGraphIO
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS = 8


def _decode_batch():
    """An all-greedy overlap decode batch the graph can feed itself."""
    sampling = Mock()
    sampling.is_all_greedy = True
    sampling.logit_bias = None
    sampling.grammar_mask = None
    sampling.has_custom_logit_processor = False
    sampling.penalizer_orchestrator = None
    sampling.has_watermark_candidates = False
    batch = Mock()
    batch.forward_mode.is_decode.return_value = True
    batch.forward_mode.is_extend.return_value = False
    batch.enable_overlap = True
    batch.spec_algorithm.is_none.return_value = True
    batch.input_ids = None
    batch.beam_tail = None
    batch.return_logprob = False
    batch.sampling_info = sampling
    batch.prefill_input_ids_cpu = None
    batch.uocr_graph_io_deferred = False
    return batch


def _graph_io():
    return UOCRDecodeGraphIO(
        num_rows=ROWS, device="cpu", captured_ids=torch.zeros(4, dtype=torch.int64)
    )


def _hooks(current_io):
    """Hooks whose decode-graph I/O is whatever ``current_io()`` returns."""
    return UocrDocHooks(
        tree_cache=Mock(),
        attn_backend=object(),
        early_embeddings=None,
        device=torch.device("cpu"),
        forward_stream=None,
        tp_size=1,
        decode_graph_io=current_io,
    )


def _future_map():
    return FutureMap(
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.NONE,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(ROWS, 4)),
        needs_cpu_seq_lens=False,
    )


class TestUOCRDecodeGraphIO(CustomTestCase):
    def test_defer_rejects_non_greedy_and_sampling_transforms(self):
        graph_io = _graph_io()
        batch = _decode_batch()
        sampling = batch.sampling_info
        self.assertTrue(graph_io.can_defer(batch))

        sampling.is_all_greedy = False
        self.assertFalse(graph_io.can_defer(batch))
        sampling.is_all_greedy = True
        sampling.logit_bias = torch.ones(1)
        self.assertFalse(graph_io.can_defer(batch))
        sampling.logit_bias = None
        batch.return_logprob = True
        self.assertFalse(graph_io.can_defer(batch))

    def test_relay_result_seeds_every_non_graph_decode_transition(self):
        graph_io = _graph_io()
        batch = _decode_batch()
        result = Mock()
        result.has_sampled_token_ids = True
        result.next_token_ids = torch.tensor([41, 43])
        result.can_run_cuda_graph = False
        indices = torch.tensor([2, 5])

        graph_io.relay_result(batch, result, indices)
        self.assertEqual(graph_io.relay[indices].tolist(), [41, 43])

        result.can_run_cuda_graph = True
        batch.sampling_info.is_all_greedy = False
        result.next_token_ids = torch.tensor([47, 53])
        graph_io.relay_result(batch, result, indices)
        self.assertEqual(graph_io.relay[indices].tolist(), [47, 53])


class TestDeferredThenEagerDecode(CustomTestCase):
    def test_a_step_after_a_graph_published_step_reads_its_tokens(self):
        """A decode whose graph published the next inputs itself must still
        leave them in the future map: the following eager, mixed or
        non-greedy step gathers its inputs from there."""
        graph_io = _graph_io()
        scheduler = object.__new__(Scheduler)
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.future_map = _future_map()
        scheduler.future_map.output_tokens_buf.zero_()
        scheduler.uocr_doc_hooks = _hooks(lambda: graph_io)
        scheduler.beam_coordinator = Mock()
        scheduler.chunked_req = None

        indices = torch.tensor([2, 5])
        published = torch.tensor([41, 43], dtype=torch.int32)
        graph_io.relay[indices] = published.to(torch.int64)
        result = SimpleNamespace(
            next_draft_input=None,
            has_sampled_token_ids=True,
            next_token_ids=published,
            can_run_cuda_graph=True,
        )
        deferred = _decode_batch()
        scheduler._relay_forward_payload(deferred, indices, result)

        eager = _decode_batch()
        eager.req_pool_indices = indices
        resolve_forward_inputs(eager, scheduler.future_map)
        self.assertEqual(eager.input_ids.tolist(), [41, 43])


class TestDecodeGraphRecapture(CustomTestCase):
    def _capture(self, greedy):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.attn_backend = object.__new__(UnlimitedOCRV2AttnBackend)
        model_runner = SimpleNamespace(
            uocr_greedy_graph=greedy,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(ROWS, 4)),
            device="cpu",
            capture_tail_hooks=[],
        )
        runner.maybe_init_uocr_graph_io(model_runner)
        self.assertEqual(model_runner.capture_tail_hooks, [])
        return runner.uocr_graph_io

    def test_a_recaptured_graph_relays_into_its_own_io_only(self):
        """Each capture records the relay of the argmax ids once; a recapture
        must replace the previous runner's relay, not record both."""
        greedy = UocrGreedyGraph(max_bs=4, device="cpu")
        old_io = self._capture(greedy)
        new_io = self._capture(greedy)
        out = SimpleNamespace(next_token_logits=torch.eye(4)[[3, 1]])
        fb = SimpleNamespace(req_pool_indices=torch.tensor([2, 5]))
        greedy.capture_tail_hook(None, out, fb, 2)
        self.assertEqual(new_io.relay[[2, 5]].tolist(), [3, 1])
        self.assertEqual(old_io.relay.tolist(), [0] * ROWS)

    def test_the_scheduler_feeds_the_recaptured_graph(self):
        """After a recapture the scheduler must defer to, force inputs into and
        seed the new runner's relay; the new one starts empty."""
        ios = [_graph_io()]
        hooks = _hooks(lambda: ios[-1])
        future_map = _future_map()
        future_map.output_tokens_buf.copy_(torch.arange(ROWS) + 100)
        batch = _decode_batch()
        self.assertTrue(hooks.prepare_forward_inputs(batch, future_map))

        ios.append(_graph_io())
        hooks._reuse._input_overrides[3] = 7
        self.assertTrue(hooks.prepare_forward_inputs(batch, future_map))
        want = (torch.arange(ROWS) + 100).tolist()
        want[3] = 7
        self.assertEqual(ios[-1].relay.tolist(), want)
        self.assertEqual(future_map.output_tokens_buf.tolist(), want)


class TestUocrGreedyGraph(CustomTestCase):
    def test_recorded_ids_are_used_only_after_a_graph_replay(self):
        """A decode that fell back to eager must sample, not reuse the last replay's ids."""
        graph = UocrGreedyGraph(max_bs=4, device="cpu")
        graph.ids.copy_(torch.tensor([7, 8, 9, 10]))
        logits = Mock(next_token_logits=torch.zeros(2, 5))
        replayed = _decode_batch()
        graph.before_forward(replayed, replays_graph=True, decode_graph_runner=None)
        ids = graph.recorded_ids(logits, replayed, watermark_state=None)
        self.assertEqual(ids.tolist(), [7, 8])
        self.assertEqual(ids.dtype, torch.int32)

        eager = _decode_batch()
        graph.before_forward(eager, replays_graph=False, decode_graph_runner=None)
        self.assertIsNone(graph.recorded_ids(logits, eager, watermark_state=None))
        self.assertIsNone(graph.recorded_ids(logits, replayed, watermark_state=None))


def _scheduler_for(architecture, training_stage):
    scheduler = object.__new__(Scheduler)
    scheduler.model_config = Mock()
    scheduler.model_config.hf_config = Mock(
        architectures=[architecture],
        sparse_attention_config=(
            None if training_stage is None else {"training_stage": training_stage}
        ),
    )
    scheduler.enable_overlap = True
    return scheduler


class TestUocrCollaboratorsStayOffForOtherModels(CustomTestCase):
    def test_other_models_and_v1_build_no_collaborator(self):
        """Every uocr call site is gated on these being None; Unlimited-OCR v1
        must keep the upstream path (no crop prefetch, no document hooks)."""
        for arch, stage in (
            ("LlamaForCausalLM", None),
            ("UnlimitedOCRForCausalLM", None),
            ("UnlimitedOCRForCausalLM", "kl"),
        ):
            with self.subTest(arch=arch, stage=stage):
                scheduler = _scheduler_for(arch, stage)
                scheduler.tree_cache = Mock()
                scheduler._target_model_runner = Mock()
                scheduler.maybe_init_uocr_doc_hooks()
                scheduler.maybe_init_uocr_crop_transfer()
                scheduler.maybe_init_deferred_overlap_release()
                self.assertIsNone(scheduler.uocr_doc_hooks)
                self.assertIsNone(scheduler.uocr_crop_transfer)
                self.assertIsNone(scheduler.deferred_overlap_release)
                scheduler.tree_cache.enable_uocr_doc_reuse.assert_not_called()
                scheduler._target_model_runner.assert_not_called()
                self.assertIsNone(
                    UocrGreedyGraph.maybe_create(
                        model_config=scheduler.model_config,
                        max_running_requests=8,
                        device="cpu",
                    )
                )

    def test_v2_hands_one_reuse_state_to_the_cache_and_the_backend(self):
        """The backend runs the queued edits of the reuse state the cache
        matches against; two different objects would never run the copies."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_parallel

        scheduler = _scheduler_for("UnlimitedOCRV2ForCausalLM", "dash")
        tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        tree_cache.session = SimpleNamespace()
        scheduler.tree_cache = tree_cache
        backend = object.__new__(UnlimitedOCRV2AttnBackend)
        scheduler._target_model_runner = Mock(
            return_value=SimpleNamespace(
                attn_backend=backend,
                model=SimpleNamespace(early_embeddings=None),
                device="cpu",
                forward_stream=None,
                decode_cuda_graph_runner=None,
            )
        )
        with get_parallel().override(tp_size=1):
            scheduler.maybe_init_uocr_doc_hooks()
        reuse = tree_cache.uocr_doc_reuse
        self.assertIsNotNone(reuse)
        self.assertIs(backend.doc_reuse, reuse)
        self.assertIs(tree_cache.session.uocr_doc_reuse, reuse)


class TestFlushWithDocumentState(CustomTestCase):
    def _hooks(self):
        tree_cache = Mock()
        tree_cache.page_size = 32
        return _hooks(lambda: None), tree_cache

    def test_held_or_matched_windows_block_a_flush(self):
        """A matched follow-up owns copied prefix slots and its source window,
        so a pool clear under it leaves both dangling; held-off windows block
        like waiting requests unless the waiting queue is ignored."""
        hooks, _ = self._hooks()
        self.assertFalse(hooks.blocks_flush(ignore_waiting=False))
        hooks._reuse.sources[1] = object()
        self.assertTrue(hooks.blocks_flush(ignore_waiting=True))
        hooks._reuse.sources.clear()
        hooks._prefetch._parked["w0"] = SimpleNamespace(rid="w1")
        self.assertTrue(hooks.blocks_flush(ignore_waiting=False))
        self.assertFalse(hooks.blocks_flush(ignore_waiting=True))

    def test_flush_drops_document_state_before_the_pools_and_aborts_held_windows(self):
        """Held windows must get an answer, and the retained windows must be
        forgotten before the clear returns their KV to the pools."""
        hooks, _ = self._hooks()
        held = SimpleNamespace(rid="w1")
        hooks._prefetch._parked["w0"] = held
        hooks._reuse.docs["doc"] = object()
        seen = []
        scheduler = object.__new__(Scheduler)
        scheduler.uocr_doc_hooks = hooks
        scheduler._engine_paused = False
        scheduler.is_fully_idle = lambda **kw: True
        scheduler.tree_cache = Mock()
        scheduler.tree_cache.reset.side_effect = lambda: seen.append(
            dict(hooks._reuse.docs)
        )
        for name in (
            "req_to_token_pool",
            "token_to_kv_pool_allocator",
            "grammar_manager",
            "metrics_reporter",
        ):
            setattr(scheduler, name, Mock())
        scheduler.draft_worker = None
        aborted = []
        scheduler._abort_uocr_doc_windows = lambda reqs, **kw: aborted.extend(reqs)
        self.assertTrue(scheduler.flush_cache(empty_cache=False))
        self.assertEqual(seen, [{}])
        self.assertEqual(aborted, [held])


if __name__ == "__main__":
    unittest.main()
