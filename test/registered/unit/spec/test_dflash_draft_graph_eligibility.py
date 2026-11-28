"""The scheduler's graph admission must reach the DFLASH draft runner, and the
aux graphs are all captured before serving."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import (
    _prefix_valid_mla_write_loc,
    _set_mha_kv_buffer_prefix_valid,
    unwrap_write_loc,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashDraftGraphEligibility(unittest.TestCase):
    def test_scheduler_admission_reaches_the_draft_forward_batch(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 2
        worker.draft_model = SimpleNamespace()
        embedding = torch.nn.Embedding(16, 4)
        target_model = SimpleNamespace(
            get_input_embeddings=lambda: embedding, lm_head=torch.nn.Linear(4, 16)
        )
        iteration_ids = SimpleNamespace(bind=Mock())
        target_translator = SimpleNamespace(plan=Mock(return_value=iteration_ids))
        draft_translator = SimpleNamespace()
        worker.model_runner = SimpleNamespace(
            runner_ids=target_translator,
            model=target_model,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16).view(1, 16)
            ),
        )
        worker._target_worker = SimpleNamespace(model_runner=worker.model_runner)
        # Stop at the actual runner boundary, before target verification.
        worker.draft_model_runner = SimpleNamespace(
            runner_ids=draft_translator,
            forward=Mock(side_effect=StopIteration),
        )
        worker.draft_owns_attention = False
        worker._draft_block_ids_buf = None
        worker._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=2, device="cpu"
        )
        worker._block_pos_offsets = torch.arange(2)
        worker._use_triton_prepare_block = False
        worker._mask_token_id = 2
        worker._full_embed_gpu = None
        worker._noise_embed_scale = 1.0
        worker.use_compact_draft_cache = False
        worker.draft_kv_ring = None
        worker.selector = None
        worker.lilicorr = None
        worker._info_dumper = Mock()

        draft_input = DFlashDraftInputV2.create_idle_input(device="cpu")
        draft_input.bonus_tokens = torch.tensor([1])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            sampling_info=None,
            spec_info=draft_input,
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            req_pool_indices=torch.tensor([0]),
        )
        # A bound runtime that returns None hands the round to the stock one.
        declining = Mock()
        declining.decode_round.return_value = None
        for allowed, runtime in ((False, None), (True, None), (True, declining)):
            batch.can_run_decode_cuda_graph = allowed
            worker._runtime = runtime
            with (
                self.subTest(allowed=allowed, runtime=runtime),
                patch.object(
                    dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
                ),
                patch.object(torch.Tensor, "record_stream"),
                patch.object(torch, "get_device_module", return_value=Mock()),
                patch.object(
                    dflash_worker_v2,
                    "assign_extend_cache_locs_func",
                    return_value=torch.tensor([3, 4]),
                ),
                patch.object(
                    dflash_worker_v2, "enable_num_token_non_padded", return_value=False
                ),
                self.assertRaises(StopIteration),
            ):
                worker.forward_batch_generation(batch)
            forward_batch = worker.draft_model_runner.forward.call_args.args[0]
            iteration_ids.bind.assert_called_with(forward_batch, draft_translator)
            self.assertIs(forward_batch.can_run_decode_cuda_graph, allowed)
            self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)
            torch.testing.assert_close(forward_batch.input_ids, torch.tensor([1, 2]))
        declining.decode_round.assert_called_once_with(
            batch, on_publish=None, grammar_barrier=None
        )
        # Otherwise it runs the whole round: nothing of the stock round launches.
        taking = Mock()
        taking.decode_round.return_value = "the runtime's round"
        worker._runtime = taking
        worker.draft_model_runner.forward.reset_mock()
        self.assertEqual(
            worker.forward_batch_generation(batch, grammar_barrier=print),
            "the runtime's round",
        )
        taking.decode_round.assert_called_once_with(
            batch, on_publish=None, grammar_barrier=print
        )
        worker.draft_model_runner.forward.assert_not_called()


class _FakeGraph:
    """Records the writes issued inside a capture and reissues them on replay,
    reading the captured input buffers at replay time like a CUDA graph."""

    capturing = None

    def __init__(self):
        self.ops = []

    def replay(self):
        for op in self.ops:
            op()


class TestDFlashPrewarmAfterKvSizing(unittest.TestCase):
    BLOCK = 2
    WIDTH = 6  # num_context_features (2) x draft hidden_size (3)
    CAPTURE_BS = [1, 2, 4]
    SLOTS = 32

    def _worker(self, *, windowed=False):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = self.BLOCK
        worker.model_runner = SimpleNamespace(
            decode_cuda_graph_capture_bs=self.CAPTURE_BS,
            dtype=torch.float32,
            max_running_requests=8,
            max_decode_logits_rows=lambda: 64,
        )
        worker._target_worker = SimpleNamespace(
            model_runner=worker.model_runner,
            model_config=SimpleNamespace(vocab_size=8),
        )
        worker.draft_model = SimpleNamespace(
            fc_tp_columns=None,
            num_context_features=2,
            config=SimpleNamespace(hidden_size=3),
        )

        def space(kind):
            self.assertIs(kind, IdSpaceKind.SLIDING_WINDOW)
            return object() if windowed else None

        worker.draft_model_runner = SimpleNamespace(
            runner_ids=SimpleNamespace(space=space)
        )
        worker.draft_kv_ring = None
        worker._use_aux_graph = True
        worker._aux_graphs = {}
        worker._aux_overlap = False
        worker._batched_small_copy = False
        worker._use_triton_accept_bonus = False
        worker._accept_bonus_buffer_slot = 1
        worker._selector_sample = ("stale", "selector sample")
        worker._selector_sampling_accept = Mock(side_effect=AssertionError)
        worker._simulated_acc_len_rng = None
        worker._tp_sync = SimpleNamespace(sync=lambda site, values: values)

        # The draft writers: the real prefix-valid MHA commit (its torch path)
        # and the real MLA masking, onto two fake pools.
        self.mha = torch.full((self.SLOTS, self.WIDTH), -1.0)
        self.mla = torch.full((self.SLOTS, self.WIDTH), -1.0)
        self.eager_writes = []

        def store(buffer):
            def set_kv_buffer(layer, write_loc, cache_k, *args, **kwargs):
                buffer[unwrap_write_loc(write_loc)[0]] = cache_k

            return set_kv_buffer

        mha_pool = SimpleNamespace(k_buffer=[self.mha], set_kv_buffer=store(self.mha))

        def write(*, target_hidden, cache_loc_2d, commit_lens, **kwargs):
            def run():
                _set_mha_kv_buffer_prefix_valid(
                    mha_pool,
                    SimpleNamespace(layer_id=0),
                    cache_loc_2d,
                    commit_lens,
                    target_hidden,
                    target_hidden,
                )
                store(self.mla)(
                    None,
                    _prefix_valid_mla_write_loc(
                        cache_loc_2d, commit_lens, target_hidden
                    ),
                    target_hidden,
                )

            if _FakeGraph.capturing is not None:
                _FakeGraph.capturing.ops.append(run)
            else:
                self.eager_writes.append(kwargs.get("swa_cache_loc"))
                run()

        worker._append_target_hidden_to_draft_kv_by_loc = write
        return worker

    @contextmanager
    def _fake_cuda(self):
        self.pools = []
        stream = SimpleNamespace(wait_stream=lambda other: None)

        @contextmanager
        def graph(cuda_graph, *, pool, stream):
            self.pools.append(pool)
            _FakeGraph.capturing = cuda_graph
            try:
                yield
            finally:
                _FakeGraph.capturing = None

        with (
            patch.object(torch.cuda, "CUDAGraph", _FakeGraph),
            patch.object(torch.cuda, "graph", graph),
            patch.object(torch.cuda, "graph_pool_handle", object),
            patch.object(torch.cuda, "Stream", lambda device: stream),
            patch.object(torch.cuda, "current_stream", lambda device: stream),
            patch.object(torch.cuda, "stream", lambda s: nullcontext()),
        ):
            yield

    def _serve(self, worker, bs, *, first_slot):
        """One round's append: request i commits i + 1 tokens (capped at the
        block) into its own slots, with a hidden row value naming the slot."""
        loc = torch.arange(first_slot, first_slot + bs * self.BLOCK)
        commit = torch.clamp(torch.arange(1, bs + 1), max=self.BLOCK).to(torch.int32)
        hidden = loc[:, None].float().expand(-1, self.WIDTH).contiguous()
        worker._append_target_hidden_decode(
            target_hidden=hidden,
            cache_loc=loc,
            positions=loc,
            cache_loc_2d=loc.view(bs, self.BLOCK),
            commit_lens=commit,
        )
        written = loc.view(bs, self.BLOCK)[
            torch.arange(self.BLOCK)[None, :] < commit[:, None]
        ]
        return written

    def test_prewarm_captures_every_graph_batch_size_without_writing_kv(self):
        worker = self._worker()
        with self._fake_cuda():
            worker.prewarm_after_kv_sizing()

            # Every batch size up to the largest decode graph, not only the
            # captured ones (3 runs on the bs-4 target graph).
            self.assertEqual(
                sorted(key[1][0] for key in worker._aux_graphs), [1, 2, 3, 4]
            )
            # One pool for every capture.
            self.assertEqual(len(self.pools), 4)
            self.assertEqual(len({id(pool) for pool in self.pools}), 1)
            # Zero commit lengths: no MHA write, MLA rows only to slot 0.
            self.assertTrue(torch.all(self.mha == -1))
            self.assertTrue(torch.all(self.mla[1:] == -1))
            self.assertFalse(worker._aux_pending)

            # Serving: no capture, and a smaller batch after a larger one on
            # the shared input buffers writes exactly its own committed rows.
            graphs = dict(worker._aux_graphs)
            self.eager_writes.clear()  # the pre-capture warm-ups
            expected = torch.full((self.SLOTS,), -1.0)
            for bs, first_slot in ((4, 1), (3, 10), (1, 20), (2, 24)):
                written = self._serve(worker, bs, first_slot=first_slot)
                expected[written] = written.float()
            self.assertEqual(worker._aux_graphs, graphs)
            self.assertEqual(self.eager_writes, [])
            torch.testing.assert_close(self.mha[:, 0], expected)

            # Past the largest decode graph the append runs eagerly.
            self._serve(worker, 5, first_slot=20)
            self.assertEqual(worker._aux_graphs, graphs)
            self.assertEqual(self.eager_writes, [None])

    def test_windowed_draft_rehearses_its_eager_append(self):
        worker = self._worker(windowed=True)
        with self._fake_cuda():
            worker.prewarm_after_kv_sizing()
        self.assertEqual(worker._aux_graphs, {})
        # Two appends per batch size, each with the window ids.
        self.assertEqual(len(self.eager_writes), 2 * 4)
        self.assertTrue(all(loc is not None for loc in self.eager_writes))
        self.assertTrue(torch.all(self.mha == -1))
        self.assertTrue(torch.all(self.mla[1:] == -1))

    def test_prewarm_restores_rng_and_round_state(self):
        worker = self._worker()
        # Three requests at most: an odd number of greedy accepts flips the
        # double-buffered accept outputs off their slot.
        worker.model_runner.max_running_requests = 3
        worker._use_triton_accept_bonus = True
        worker._accept_bonus_buffer_cap = 0
        worker._simulated_acc_len_rng = torch.Generator().manual_seed(7)
        sim_state = worker._simulated_acc_len_rng.get_state()
        accepts = []

        def sampled_accept(*, candidates, sampling_info, **kwargs):
            accepts.append(("sampled", candidates.shape[0]))
            torch.rand(candidates.shape, device=candidates.device)
            bs = candidates.shape[0]
            return torch.zeros(bs, dtype=torch.int32), torch.zeros(
                bs, dtype=torch.int64
            )

        def simulated(*, accept_len, new_seq_lens, generator, **kwargs):
            torch.rand(accept_len.shape, generator=generator)
            return new_seq_lens

        def greedy_accept(*, candidates, commit_lens_out, new_seq_lens_out, **kwargs):
            accepts.append(("greedy", candidates.shape[0]))
            commit_lens_out.fill_(1)
            new_seq_lens_out.fill_(1)

        cpu_state = torch.get_rng_state()
        with (
            self._fake_cuda(),
            patch.object(
                dflash_worker_v2, "is_dflash_sampling_verify_available", lambda: True
            ),
            patch.object(
                dflash_worker_v2,
                "compute_dflash_sampling_correct_drafts_and_bonus",
                sampled_accept,
            ),
            patch.object(
                dflash_worker_v2,
                "_compute_dflash_accept_bonus_triton_unchecked",
                greedy_accept,
            ),
            patch.object(dflash_worker_v2, "SIMULATE_ACC_LEN", 2),
            patch.object(
                dflash_worker_v2, "apply_dflash_simulated_acceptance", simulated
            ),
        ):
            worker.prewarm_after_kv_sizing()

        self.assertEqual(
            accepts,
            [(kind, bs) for bs in (3, 2, 1) for kind in ("greedy", "sampled")],
        )
        self.assertEqual(sorted(key[1][0] for key in worker._aux_graphs), [1, 2, 3])
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        self.assertTrue(
            torch.equal(worker._simulated_acc_len_rng.get_state(), sim_state)
        )
        self.assertEqual(worker._accept_bonus_buffer_slot, 1)
        self.assertEqual(worker._selector_sample, ("stale", "selector sample"))
        worker._selector_sampling_accept.assert_not_called()

    def test_no_decode_graphs_no_prewarm(self):
        worker = self._worker()
        worker.model_runner.decode_cuda_graph_capture_bs = []
        worker._accept_block = Mock(side_effect=AssertionError)
        with self._fake_cuda():
            worker.prewarm_after_kv_sizing()
        self.assertEqual(worker._aux_graphs, {})


if __name__ == "__main__":
    unittest.main()
