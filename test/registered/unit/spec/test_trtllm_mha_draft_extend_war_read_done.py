"""TRTLLM-MHA draft-extend WAR read-done capability and ordering pins.

The EAGLE draft-extend CUDA-graph runner publishes
``model_runner.shared_read_done_event`` -- the event the scheduler WAR barrier
(``Scheduler._apply_war_barrier``) waits on before its next shared-buffer
write (e.g. ``assign_req_to_token_pool_func`` in ``mem_cache/allocation.py``).
TRTLLM-MHA records its draft-extend metadata rebuild INTO the captured graph
(``init_forward_metadata_in_graph`` -> ``_apply_cuda_graph_metadata`` ->
``update_trtllm_mha_graph_metadata``), so every replay re-reads
``req_to_token`` on device. Publishing the read-done event before the replay
launch therefore lets the scheduler race its next-batch ``req_to_token``
writes against the replay reads: silently wrong page tables, no error. The
runner keys the publication position on
``AttentionBackend.draft_extend_rereads_shared_state_in_graph()``.

Pins, on a real graph capture of the backend recorded metadata rebuild:
  1. the premise: replay re-reads ``req_to_token`` (no capture-time snapshot);
  2. the capability decoupling: rereads-shared-state is True while
     ``draft_extend_metadata_captured_in_graph()`` stays False (that one lets
     runners skip the eager out-graph rebuild and gates multi-layer
     single-graph draft modes -- TRTLLM-MHA satisfies neither);
  3. the ordering: with a deterministically widened race window, the pre-fix
     early publication reproduces the corruption, and the post-replay
     publication the capability selects keeps the snapshot intact.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
    read_done_publishes_after_replay,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

BS = 4
POOL_SIZE = 8
PAGE_SIZE = 32
MAX_NUM_PAGES = 16
MAX_CTX = MAX_NUM_PAGES * PAGE_SIZE
NUM_TOKENS_PER_REQ = 4


def _page_aligned_table(page_base: int, device) -> torch.Tensor:
    """req_to_token whose row r maps page p to page id page_base + r*MAX + p."""
    rows = []
    for r in range(POOL_SIZE):
        pages = page_base + r * MAX_NUM_PAGES + torch.arange(MAX_NUM_PAGES)
        rows.append(torch.repeat_interleave(pages * PAGE_SIZE, PAGE_SIZE))
    return torch.stack(rows).to(dtype=torch.int32, device=device)


def _expected_page_table(page_base: int, req_pool_indices, device) -> torch.Tensor:
    cols = torch.arange(MAX_NUM_PAGES, device=device, dtype=torch.int32)
    base = page_base + req_pool_indices.to(torch.int32) * MAX_NUM_PAGES
    return base[:, None] + cols[None, :]


class _StubCapturedBackend(AttentionBackend):
    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return True


class TestTRTLLMMHADraftExtendWARReadDone(CustomTestCase):
    def _make_backend(self, device):
        # Skeleton instance: only the state the draft-extend graph metadata
        # path touches, mirroring init_cuda_graph_state allocations.
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend.device = device
        backend.page_size = PAGE_SIZE
        backend.max_num_pages = MAX_NUM_PAGES
        backend._swa_kv_pool = None
        backend._swa_full_to_swa_mapping = None
        backend.use_sliding_window_kv_pool = False
        backend.expand_encoder_only_verify = False
        backend.speculative_step_id = 0
        backend.forward_metadata = None
        backend.req_to_token = _page_aligned_table(0, device)
        backend.draft_extend_metadata = {
            "cache_seqlens": torch.zeros(BS, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.zeros(BS + 1, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.zeros(BS + 1, dtype=torch.int32, device=device),
            "page_table": torch.zeros(
                BS, MAX_NUM_PAGES, dtype=torch.int32, device=device
            ),
            "swa_page_table": None,
        }
        return backend

    def _capture(self, device):
        """Capture the backend recorded draft-extend metadata rebuild, exactly
        as EAGLEDraftExtendCudaGraphRunner captures init_forward_metadata_in_graph."""
        backend = self._make_backend(device)
        req_pool_indices = torch.tensor([5, 2, 7, 0], dtype=torch.int64, device=device)
        fb = SimpleNamespace(
            batch_size=BS,
            req_pool_indices=req_pool_indices,
            # All pages live so the kernel rewrites every column.
            seq_lens=torch.full((BS,), MAX_CTX, dtype=torch.int64, device=device),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=SimpleNamespace(num_tokens_per_req=NUM_TOKENS_PER_REQ),
            out_cache_loc=None,
        )
        backend._build_cuda_graph_metadata(
            BS,
            BS * NUM_TOKENS_PER_REQ,
            ForwardMode.DRAFT_EXTEND_V2,
            fb.spec_info,
            device,
        )
        # Warm up the triton JIT outside capture.
        backend.init_forward_metadata_in_graph(fb)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.init_forward_metadata_in_graph(fb)
        torch.cuda.synchronize()
        return backend, graph, req_pool_indices

    def _page_table(self, backend) -> torch.Tensor:
        return backend.draft_extend_metadata[BS].page_table

    def test_replay_rereads_req_to_token(self):
        """The captured rebuild gathers from req_to_token at replay time."""
        device = torch.device("cuda")
        backend, graph, req_pool_indices = self._capture(device)
        backend.req_to_token.copy_(_page_aligned_table(1000, device))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            self._page_table(backend),
            _expected_page_table(1000, req_pool_indices, device),
            msg="replay did not re-read req_to_token; re-audit "
            "draft_extend_rereads_shared_state_in_graph before changing it",
        )

    def test_capability_decoupled_from_captured_in_graph(self):
        backend = object.__new__(TRTLLMHAAttnBackend)
        # WAR deferral: the captured graph re-reads shared state at replay.
        self.assertTrue(backend.draft_extend_rereads_shared_state_in_graph())
        # Unchanged: replay still needs the eager out-graph rebind, and
        # multi-layer single-graph draft modes must stay gated off.
        self.assertFalse(backend.draft_extend_metadata_captured_in_graph())

    def test_base_default_mirrors_captured_in_graph(self):
        base = object.__new__(AttentionBackend)
        self.assertFalse(base.draft_extend_rereads_shared_state_in_graph())
        stub = object.__new__(_StubCapturedBackend)
        self.assertTrue(stub.draft_extend_rereads_shared_state_in_graph())

    def _run_publication_protocol(self, publish_after_replay: bool) -> bool:
        """Emulate the runner/scheduler WAR protocol around one replay.

        The writer stream is the scheduler: it waits on the published
        read-done event, then overwrites req_to_token -- exactly what
        Scheduler._apply_war_barrier permits. The pre-fix early publication
        PERMITS the write to land before the replay's reads; that permitted
        interleaving is constructed deterministically (writer_done fence)
        instead of racing free-running streams. Returns True when the replay
        observed the pre-write snapshot (correct), False when it observed the
        writer mutation (silent corruption).
        """
        device = torch.device("cuda")
        backend, graph, req_pool_indices = self._capture(device)
        mutated = _page_aligned_table(1000, device)
        forward_stream = torch.cuda.Stream()
        writer_stream = torch.cuda.Stream()
        read_done = torch.cuda.Event()
        writer_done = torch.cuda.Event()
        torch.cuda.synchronize()
        if publish_after_replay:
            # Fixed protocol: the read-done record lands after the replay on
            # the same stream, so the waiting writer cannot precede the reads.
            with torch.cuda.stream(forward_stream):
                graph.replay()
                read_done.record()
            with torch.cuda.stream(writer_stream):
                writer_stream.wait_event(read_done)
                backend.req_to_token.copy_(mutated)
        else:
            # Pre-fix protocol: the event is already recorded when the writer
            # waits on it, so the write may land before the replay's reads.
            with torch.cuda.stream(forward_stream):
                read_done.record()
            with torch.cuda.stream(writer_stream):
                writer_stream.wait_event(read_done)
                backend.req_to_token.copy_(mutated)
                writer_done.record()
            with torch.cuda.stream(forward_stream):
                forward_stream.wait_event(writer_done)
                graph.replay()
        torch.cuda.synchronize()
        snapshot = _expected_page_table(0, req_pool_indices, device)
        return bool(torch.equal(self._page_table(backend), snapshot))

    def test_early_publication_reproduces_the_race(self):
        """Pre-fix ordering: the scheduler write beats the replay reads."""
        self.assertFalse(self._run_publication_protocol(publish_after_replay=False))

    def test_capability_selected_ordering_is_safe(self):
        """The ordering the runner actually derives for this backend
        (read_done_publishes_after_replay, the production predicate) keeps the
        replay reading the snapshot; regressing either the capability or the
        runner predicate makes this test observe the corruption again."""
        backend = object.__new__(TRTLLMHAAttnBackend)
        publish_after_replay = read_done_publishes_after_replay(backend)
        self.assertTrue(
            self._run_publication_protocol(publish_after_replay=publish_after_replay)
        )

    def _spy_replay_sequence(self, backend) -> list:
        """Drive the production replay/publication sequence with spies and
        return the observed call order."""
        calls = []
        recorded = []

        class _SpyEvent:
            def record(self):
                calls.append("record")
                recorded.append(self)

        runner = object.__new__(EAGLEDraftExtendCudaGraphRunner)
        runner.device_module = SimpleNamespace(Event=_SpyEvent)
        runner.model_runner = SimpleNamespace(
            device_timer=None, shared_read_done_event=None
        )
        runner.draft_extend_attn_backend = backend
        runner._replay_graph = lambda shape_key, forward_batch: calls.append("replay")
        runner._replay_and_publish_read_done(None, None)
        # The published event must be the one this phase recorded.
        self.assertEqual(len(recorded), 1)
        self.assertIs(runner.model_runner.shared_read_done_event, recorded[0])
        return calls

    def test_production_sequence_defers_record_for_trtllm_mha(self):
        """The runner's own replay/publication method must record the WAR
        read-done event after the replay launch for TRTLLM-MHA (its captured
        graph re-reads req_to_token during replay)."""
        backend = object.__new__(TRTLLMHAAttnBackend)
        self.assertEqual(self._spy_replay_sequence(backend), ["replay", "record"])

    def test_production_sequence_keeps_early_record_for_eager_backends(self):
        """Backends without in-graph shared-state re-reads keep the legacy
        early publication (scheduler overlap)."""
        backend = object.__new__(AttentionBackend)
        self.assertEqual(self._spy_replay_sequence(backend), ["record", "replay"])


if __name__ == "__main__":
    unittest.main()
