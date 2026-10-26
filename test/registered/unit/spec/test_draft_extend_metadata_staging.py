"""How a multi-layer EAGLE draft chain gets its draft-extend metadata: a staging
backend fills it once per cycle, through its explicit staging entry, and no
step's replay rebuilds it; any other backend keeps the per-step rebuild. A chain
staged before verify is fenced at the target's read end, which only a pool that
translates nothing after verify can take."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
    InklingShortConvAttnBackend,
    InklingShortConvHybridAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_memory_pool import (
    UnifiedHybridReqToTokenPool,
    UnifiedMambaSlotAllocator,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
    MultiLayerEagleWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, RAW_BS, WIDTH = 4, 3, 3


class _Backend:
    """A draft-extend backend recording which entry its metadata came through."""

    def __init__(self, *, staging, in_graph=False):
        self.supports_draft_extend_metadata_staging = staging
        self._in_graph = in_graph
        self.staged = []
        self.rebuilt = []

    def draft_extend_metadata_captured_in_graph(self):
        return self._in_graph

    def stage_draft_extend_metadata(self, batch):
        self.staged.append(batch)

    def init_forward_metadata_out_graph(self, batch, in_capture=False):
        self.rebuilt.append(batch)


def _composite(backends):
    composite = MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
        MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
    )
    composite.draft_extend_attn_backend_list = backends
    composite.captured_req_width = WIDTH
    composite.buffers = SimpleNamespace(
        req_pool_indices=torch.tensor([3, 1, 2, 7]),
        seq_lens=torch.tensor([9, 8, 7, 1]),
        extend_seq_lens=torch.full((BS,), WIDTH),
        out_cache_loc=torch.arange(BS * WIDTH),
        mamba_track_indices=None,
    )
    composite._out_cache_loc_virtual = torch.arange(100, 100 + RAW_BS * WIDTH)
    composite._kv_loc_plan = object()
    composite._kv_loc_cols = None
    return composite


class TestStageMetadata(CustomTestCase):
    def test_staging_backends_are_staged_through_their_entry(self):
        staged = [_Backend(staging=True) for _ in range(3)]
        in_graph = _Backend(staging=True, in_graph=True)
        eager = _Backend(staging=False)
        composite = _composite([*staged, in_graph, eager])

        composite._stage_metadata(BS, RAW_BS)

        for backend in staged:
            self.assertEqual(len(backend.staged), 1)
            self.assertEqual(backend.rebuilt, [])
            batch = backend.staged[0]
            self.assertEqual(batch.batch_size, BS)
            self.assertEqual(batch.forward_mode, ForwardMode.DRAFT_EXTEND_V2)
            # The cycle's plan and virtual window reach the backend as staged.
            self.assertIs(batch.kv_loc_plan, composite._kv_loc_plan)
            self.assertIs(batch.out_cache_loc_virtual, composite._out_cache_loc_virtual)
            self.assertEqual(batch.extend_seq_lens.tolist(), [WIDTH] * BS)
            self.assertEqual(batch.out_cache_loc.numel(), BS * WIDTH)
        # One batch serves every step backend.
        self.assertTrue(all(b.staged[0] is staged[0].staged[0] for b in staged))
        # Padded lanes read the sink row.
        self.assertEqual(composite.buffers.req_pool_indices.tolist(), [3, 1, 2, 0])
        for backend in (in_graph, eager):
            self.assertEqual((backend.staged, backend.rebuilt), ([], []))

    def test_a_step_replay_rebuilds_only_without_staging(self):
        for staging in (True, False):
            with self.subTest(staging=staging):
                backend = _Backend(staging=staging)
                runner = MultiLayerEagleDraftExtendCudaGraphRunner.__new__(
                    MultiLayerEagleDraftExtendCudaGraphRunner
                )
                runner.step = 0
                runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
                runner.captured_req_width = WIDTH
                runner.metadata_captured_in_graph = False
                runner.attn_backend = backend
                runner.eagle_worker = SimpleNamespace(
                    draft_extend_attn_backend_list=[backend]
                )
                runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
                runner.model_runner = SimpleNamespace(device_timer=None)
                runner.buffers = _composite([]).buffers
                runner.buffers.input_ids = torch.zeros(BS * WIDTH, dtype=torch.int64)
                runner._replay_graph = lambda shape_key, fb_view: "replayed"

                out = runner.replay(
                    BS,
                    None,
                    SimpleNamespace(),
                    None,
                    out_cache_loc_virtual=None,
                    kv_loc_plan=None,
                    kv_loc_cols=None,
                )

                self.assertEqual(out, "replayed")
                self.assertEqual(len(backend.rebuilt), 0 if staging else 1)
                self.assertEqual(backend.staged, [])


class TestInklingStaging(CustomTestCase):
    def _wrapper(self, *, full_staging=True, recordable=True):
        wrapper = InklingShortConvHybridAttnBackend.__new__(
            InklingShortConvHybridAttnBackend
        )
        calls = MagicMock()
        wrapper.full_attn_backend = calls.full
        wrapper.full_attn_backend.supports_draft_extend_metadata_staging = full_staging
        wrapper.short_conv_backend = calls.short_conv
        wrapper.short_conv_backend._slot_gather_recordable = recordable
        return wrapper, calls

    def test_staging_follows_its_attention_and_slot_gather(self):
        for full_staging in (True, False):
            for recordable in (True, False):
                wrapper, _ = self._wrapper(
                    full_staging=full_staging, recordable=recordable
                )
                self.assertEqual(
                    wrapper.supports_draft_extend_metadata_staging,
                    full_staging and recordable,
                )

    def test_staging_fills_attention_tables_then_slot_indices(self):
        wrapper, calls = self._wrapper()
        batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        wrapper.stage_draft_extend_metadata(batch)
        self.assertEqual(
            calls.mock_calls,
            [
                call.full.stage_draft_extend_metadata(batch),
                call.short_conv._prepare_slot_indices(batch),
            ],
        )

    def test_capture_binds_the_graph_buffers_before_staging(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        for in_capture in (True, False):
            with self.subTest(in_capture=in_capture):
                wrapper, calls = self._wrapper()
                wrapper.init_forward_metadata_out_graph(batch, in_capture=in_capture)
                staging = [
                    call.full.stage_draft_extend_metadata(batch),
                    call.short_conv._prepare_slot_indices(batch),
                ]
                binding = [
                    call.full.init_forward_metadata_out_graph(batch, in_capture=True)
                ]
                self.assertEqual(
                    calls.mock_calls, (binding if in_capture else []) + staging
                )


class TestEarlyStagingFence(CustomTestCase):
    """Staging a chain before verify moves the scheduler's fence to the target's
    read end. Inkling's draft graph translates its checkpoint slots through the
    unified table during replay, after that fence, so a unified chain must keep
    the draft's fence; a static pool's replay reads its slots as staged."""

    def _inkling(self, v2p):
        if v2p is None:
            pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        else:
            pool = UnifiedHybridReqToTokenPool.__new__(UnifiedHybridReqToTokenPool)
            pool.mamba_allocator = UnifiedMambaSlotAllocator(
                SimpleNamespace(virtual_to_physical=v2p),
                max_size=v2p.numel() - 1,
                device="cpu",
            )
        short_conv = InklingShortConvAttnBackend.__new__(InklingShortConvAttnBackend)
        short_conv.req_to_token_pool = pool
        short_conv._slot_gather_recordable = True
        short_conv._graph_track_indices = torch.empty(BS, dtype=torch.int64)
        short_conv._cache_indices = None
        wrapper = InklingShortConvHybridAttnBackend.__new__(
            InklingShortConvHybridAttnBackend
        )
        wrapper.short_conv_backend = short_conv
        wrapper.full_attn_backend = _Backend(staging=True)
        return wrapper

    def _stage_and_fence(self, backend, *, translating):
        """Whether the chain staged before verify, and the runner whose read end
        the scheduler then waits for."""
        staged_calls = []
        target_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                in_graph_metadata_prep_done=object()
            )
        )
        draft_worker = SimpleNamespace(
            cuda_graph_runner_for_draft_extend=SimpleNamespace(
                require_mlp_tp_gather=False,
                max_bs=BS,
                disable_padding=False,
                stage_shared_reads=lambda **kwargs: staged_calls.append(kwargs),
            ),
            draft_extend_attn_backend_list=[backend],
            target_worker=SimpleNamespace(model_runner=target_runner),
            draft_runner_list=[
                SimpleNamespace(
                    kv_index_translator=SimpleNamespace(
                        is_translating=translating, plan=lambda **kwargs: object()
                    )
                )
            ],
            topk=1,
            speculative_num_draft_tokens=WIDTH,
            _compute_boundary_kv_locs_positions=lambda batch: (
                torch.arange(RAW_BS * WIDTH),
                None,
            ),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([9, 8, 7]),
            req_pool_indices=torch.tensor([3, 1, 2]),
        )
        staged = MultiLayerEagleDraftWorker._draft_extend_plan_for_decode(
            draft_worker, batch
        )
        self.assertEqual(len(staged_calls), int(staged))
        draft_runner = object()
        worker = SimpleNamespace(
            draft_worker=SimpleNamespace(last_draft_extend_staged=staged),
            target_worker=draft_worker.target_worker,
            _draft_worker=SimpleNamespace(draft_runner=draft_runner),
        )
        fence = MultiLayerEagleWorkerV2.last_shared_read_runner.fget(worker)
        return staged, fence is target_runner, fence is draft_runner

    def _replay_track_slots(self, backend, virtual_slots):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2, mamba_track_indices=virtual_slots
        )
        backend.init_forward_metadata_in_graph(batch)
        return backend.short_conv_backend.sconv_metadata.track_cache_indices

    def test_a_unified_chain_keeps_the_draft_fence(self):
        v2p = torch.tensor([0, 1, 3, 2, 4, 5, 6, 7])
        backend = self._inkling(v2p)
        self.assertEqual(
            self._stage_and_fence(backend, translating=True), (False, False, True)
        )
        # What the draft's fence covers: a free after the target's read end would
        # unmap the slot this replay is about to read.
        checkpoint = torch.tensor([2])
        self.assertEqual(self._replay_track_slots(backend, checkpoint).tolist(), [3])
        v2p[2] = -1
        self.assertEqual(self._replay_track_slots(backend, checkpoint).tolist(), [-1])

    def test_a_static_chain_stages_before_verify(self):
        backend = self._inkling(None)
        self.assertEqual(
            self._stage_and_fence(backend, translating=False), (True, True, False)
        )
        checkpoint = torch.tensor([2])
        self.assertIs(self._replay_track_slots(backend, checkpoint), checkpoint)


class TestBaseStaging(CustomTestCase):
    def test_a_backend_without_staging_refuses_to_stage(self):
        backend = AttentionBackend.__new__(AttentionBackend)
        self.assertFalse(backend.supports_draft_extend_metadata_staging)
        with self.assertRaises(NotImplementedError):
            backend.stage_draft_extend_metadata(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
