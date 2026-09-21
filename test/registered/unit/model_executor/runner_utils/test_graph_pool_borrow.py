"""CUDA graph-pool borrowing allocator and lifetime regression tests."""

import contextlib
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils import pool
from sglang.srt.speculative import dflash_utils, dflash_worker_v2, eagle_utils
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")


class TestGraphPoolBorrow(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.state = pool.GraphPoolBorrowState()
        # ExitStack rather than enterContext, which is 3.11+.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(pool.get_resources().override(graph_pool_borrow=self.state))

    def tearDown(self):
        if torch.cuda.is_available():
            pool._teardown_borrow_pool()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def test_mixed_segment_runs_exclude_live_blocks(self):
        """A mixed segment's free runs are borrowable, but a returned run must
        never overlap the live block — overlap would silently corrupt
        graph-owned data instead of raising an OOM."""
        snapshot = [
            {
                "allocated_size": 4096,
                "total_size": 3 * 4096,
                "blocks": [
                    {"state": "inactive", "address": 0x1000, "size": 4096},
                    {"state": "active_allocated", "address": 0x2000, "size": 4096},
                    {"state": "inactive", "address": 0x3000, "size": 4096},
                ],
            }
        ]
        with patch.object(pool.torch.cuda, "memory_snapshot", return_value=snapshot):
            runs = pool.find_free_graph_pool_runs((0, 1))
        self.assertEqual(sorted(runs), [(0x1000, 4096), (0x3000, 4096)])

    def test_free_run_snapshot_lifetime_follows_borrow_state(self):
        runs = [{"blocks": [{"state": "inactive", "address": 0x1000, "size": 8192}]}]
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "is_cuda", return_value=True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=(1, 2)),
            patch.object(
                pool.torch.cuda, "memory_snapshot", side_effect=[runs, [], runs]
            ) as snapshot,
        ):
            self.assertEqual(pool.graph_pool_borrow_largest_run(), 8192)
            self.assertEqual(pool.graph_pool_borrow_largest_run(), 8192)
            snapshot.assert_called_once_with((1, 2), include_traces=False)
            pool._teardown_borrow_pool()
            self.assertEqual(pool.graph_pool_borrow_largest_run(), 0)
            self.assertEqual(snapshot.call_count, 2)
            with pool.get_resources().override(graph_pool_borrow=None):
                self.assertEqual(pool.graph_pool_borrow_largest_run(), 8192)
            self.assertEqual(pool.graph_pool_borrow_largest_run(), 0)
            self.assertEqual(snapshot.call_count, 3)

    def test_borrow_capacity_includes_rounding_and_small_pool_reserve(self):
        for payload, largest_run, expected in (
            (-1, 64 << 20, False),
            (0, 64 << 20, False),
            (1, 0, False),
            (1, 32 << 20, False),
            (1, 34 << 20, True),
            (2 << 20, 34 << 20, True),
            ((2 << 20) + 1, 34 << 20, False),
            ((2 << 20) + 1, 36 << 20, True),
        ):
            with (
                self.subTest(payload=payload, largest_run=largest_run),
                patch.object(
                    pool, "graph_pool_borrow_largest_run", return_value=largest_run
                ),
            ):
                self.assertEqual(pool.graph_pool_borrow_can_fit(payload), expected)

    def test_graph_replay_fails_during_active_pool_borrow(self):
        graph = Mock()
        backend = object.__new__(FullCudaGraphBackend)
        backend._graphs = {"shape": graph}
        backend._outputs = {"shape": None}
        snapshot = [
            {
                "allocated_size": 0,
                "blocks": [{"state": "inactive", "address": 4096, "size": 4096}],
            }
        ]

        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=(1, 2)),
            patch.object(self.state, "stub", MagicMock(cursor_bytes=0, freed_bytes=0)),
            patch.object(self.state, "mem_pool", None),
            patch.object(pool.torch.cuda, "MemPool"),
            patch.object(pool.torch.cuda, "use_mem_pool"),
            patch.object(pool.torch.cuda, "current_stream"),
            patch.object(pool.torch.cuda, "memory_snapshot", return_value=snapshot),
            pool.borrow_graph_pool(user="test"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "graph pool already has live user"
            ):
                backend.replay("shape", None)

        graph.replay.assert_not_called()

    def test_high_cursor_keeps_reusable_cached_segments(self):
        stub = MagicMock(cursor_bytes=600, freed_bytes=0)
        mem_pool = MagicMock()
        stream = object()

        with (
            patch.object(pool, "graph_pool_borrow_enabled", return_value=True),
            patch.object(self.state, "stub", stub),
            patch.object(self.state, "mem_pool", mem_pool),
            patch.object(self.state, "stream", stream),
            patch.object(self.state, "extents_total", 1000),
            patch.object(pool, "_teardown_borrow_pool") as teardown,
            patch.object(pool.torch, "empty"),
            patch.object(pool.torch.cuda, "use_mem_pool"),
            patch.object(pool.torch.cuda, "current_stream", return_value=stream),
        ):
            with pool.borrow_graph_pool(user="test"):
                pass

        teardown.assert_not_called()

    def test_external_graph_storage_can_disable_borrowing(self):
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "is_cuda", return_value=True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=(1, 2)),
        ):
            self.assertTrue(pool.graph_pool_borrow_enabled())
            pool.disable_graph_pool_borrow("graph storage is externally managed")
            self.assertFalse(pool.graph_pool_borrow_enabled())

    def test_setting_static_runs_retires_previous_borrow_pool(self):
        runs = [(0x1000, 4096), (0x2000, 8192)]

        def reset_static_runs():
            self.state.static_runs = None

        with patch.object(
            pool, "_teardown_borrow_pool", side_effect=reset_static_runs
        ) as teardown:
            pool.set_graph_pool_borrow_runs(runs)

        teardown.assert_called_once_with()
        self.assertEqual(self.state.static_runs, [(0x2000, 8192), (0x1000, 4096)])

    def test_eagle_non_greedy_probabilities_do_not_borrow_graph_pool(self):
        def fake_sampling(**kwargs):
            kwargs["predicts"].fill_(3)
            kwargs["accept_index"].fill_(0)
            kwargs["accept_token_num"].fill_(1)

        verify_input = SimpleNamespace(
            draft_token_num=2,
            draft_token=torch.tensor([1, 2], dtype=torch.int32),
            max_tree_depth=2,
            tree_topk=1,
            retrieve_index=torch.zeros((1, 2), dtype=torch.int32),
            retrieve_next_token=torch.zeros((1, 2), dtype=torch.int32),
            retrieve_next_sibling=torch.zeros((1, 2), dtype=torch.int32),
            draft_probs=None,
        )
        sampling_info = SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=False,
            temperatures=torch.ones((1, 1)),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            sampling_seed=None,
        )
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.tensor([4], dtype=torch.int32),
            sampling_info=sampling_info,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        logits_output = SimpleNamespace(next_token_logits=torch.randn((2, 8)))
        spec_config = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        tp_group = SimpleNamespace(world_size=1)

        with (
            patch.object(pool, "borrow_graph_pool") as borrow_graph_pool,
            patch.object(eagle_utils, "get_spec", return_value=spec_config),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            # `parallel_state`, not the package re-export: a stub on the
            # re-export is never consulted.
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
            patch(
                "sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only",
                side_effect=fake_sampling,
            ),
        ):
            predict, accept_lens, accept_index = eagle_utils.eagle_sample(
                verify_input, batch, logits_output
            )

        borrow_graph_pool.assert_not_called()
        self.assertTrue(torch.equal(predict, torch.full_like(predict, 3)))
        self.assertTrue(torch.equal(accept_lens, torch.full_like(accept_lens, 2)))
        self.assertTrue(torch.equal(accept_index, torch.zeros_like(accept_index)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_borrowed_allocations_land_on_free_graph_pool_runs(self):
        """Borrowed blocks span free runs, recycle, and add no reservation."""
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            # Two capture-only transients become disjoint free graph-pool runs.
            transient_a = torch.empty(48 << 20, dtype=torch.uint8, device="cuda")
            transient_b = torch.empty(24 << 20, dtype=torch.uint8, device="cuda")
            y = x + 1
            del transient_a, transient_b
        torch.cuda.synchronize()

        device_id = torch.cuda.current_device()
        borrow_stream = torch.cuda.Stream()
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
            torch.cuda.stream(borrow_stream),
        ):
            lhs = torch.ones((32, 16), dtype=torch.bfloat16, device="cuda")
            copied_product = torch.empty((32, 32), dtype=torch.float32, device="cuda")
            pool.prewarm_graph_pool_borrow()
            reserved_before = torch.cuda.memory_reserved(device_id)
            runs = pool.find_free_graph_pool_runs(handle)
            self.assertGreaterEqual(len(runs), 2)
            largest_run_bytes = runs[0][1]

            def on_a_run(tensor):
                start = tensor.data_ptr()
                end = start + tensor.nbytes
                return any(
                    address <= start and end <= address + nbytes
                    for address, nbytes in runs
                )

            with pool.borrow_graph_pool(user="test"):
                # Together these exceed the largest run, forcing first-fit to
                # use more than one captured extent.
                a = torch.empty(40 << 20, dtype=torch.uint8, device="cuda")
                b = torch.empty(20 << 20, dtype=torch.uint8, device="cuda")
                self.assertGreater(a.nbytes + b.nbytes, largest_run_bytes)
                self.assertTrue(on_a_run(a) and on_a_run(b))
                self.assertTrue(
                    a.data_ptr() + a.nbytes <= b.data_ptr()
                    or b.data_ptr() + b.nbytes <= a.data_ptr()
                )

                recycled_address = a.data_ptr()
                del a
                c = torch.empty(40 << 20, dtype=torch.uint8, device="cuda")
                self.assertEqual(c.data_ptr(), recycled_address)
                del b, c

            self.assertEqual(self.state.stream, torch.cuda.current_stream())
            with (
                torch.cuda.stream(stream),
                self.assertRaisesRegex(
                    RuntimeError, "stream that created the borrow pool"
                ),
            ):
                with pool.borrow_graph_pool(user="wrong stream"):
                    self.fail("cross-stream borrowing must fail before allocating")
            self.assertIsNone(self.state.active_user)
            with pool.borrow_graph_pool(user="same stream"):
                reused = torch.empty(40 << 20, dtype=torch.uint8, device="cuda")
                self.assertEqual(reused.data_ptr(), recycled_address)
                del reused

                # The first GEMM on this stream must not cache its workspace
                # in borrowed storage, which replay would overwrite.
                product = torch.mm(lhs, lhs.T, out_dtype=torch.float32)
                self.assertTrue(on_a_run(product))
                copied_product.copy_(product)
                del product

            # Captures retire the persistent borrow pool. Its storage aliases
            # existing graph-pool runs, so the reserved footprint is unchanged.
            pool._teardown_borrow_pool()
            self.assertIsNone(self.state.stream)
            with torch.cuda.stream(stream), pool.borrow_graph_pool(user="new pool"):
                self.assertEqual(self.state.stream, stream)
                reused = torch.empty(40 << 20, dtype=torch.uint8, device="cuda")
                self.assertTrue(on_a_run(reused))
                del reused
            pool._teardown_borrow_pool()
            with pool.graph_pool_replay_scope():
                graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(y, torch.ones_like(y)))
            self.assertTrue(
                torch.equal(copied_product, torch.full_like(copied_product, 16))
            )

        self.assertEqual(torch.cuda.memory_reserved(device_id), reserved_before)
        del graph, y

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_input_logprobs_survive_replay_with_growing_chunks(self):
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        seed = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(200 << 20, dtype=torch.uint8, device="cuda")
            transient.fill_(7)
            keep = seed + 1
            del transient
        torch.cuda.synchronize()

        processor = InputLogprobProcessor(vocab_size=4096)
        processor.enable_fast_input_logprobs = False
        processor.enable_logprobs_chunk = True
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
        ):
            # Chunk sizes grow without a teardown in between, so each borrow
            # must still fit after the previous iterations' carves.
            for rows, chunk_size in (
                (6, 4),
                (1500, 1500),
                (2000, 2000),
                (2500, 2500),
                (2600, 2600),
            ):
                with self.subTest(rows=rows, chunk_size=chunk_size):
                    logits = torch.randn(rows, 4096, device="cuda")
                    token_ids = torch.randint(0, 4096, (rows,), device="cuda")
                    split = rows // 2
                    sample_indices = [split - 1, rows - 1]
                    metadata = SimpleNamespace(
                        sample_indices_cpu=sample_indices,
                        input_logprob_indices_cpu=list(range(rows)),
                        extend_return_top_logprob=True,
                        extend_token_ids_logprob=True,
                        top_logprobs_nums=[2, 3],
                        extend_logprob_pruned_lens_cpu=[split, rows - split],
                        extend_input_logprob_token_ids_gpu=token_ids,
                        token_ids_logprobs=[[0, 7], [4]],
                    )
                    processor.logprobs_chunk_size = chunk_size
                    get_logits = Mock(side_effect=lambda states, *_args, **_kw: states)
                    result, sampled = processor.forward(
                        pruned_states=logits,
                        sample_indices=torch.tensor(sample_indices, device="cuda"),
                        input_logprob_indices=torch.arange(rows, device="cuda"),
                        token_to_seq_idx=[0] * split + [1] * (rows - split),
                        lm_head=None,
                        get_logits_fn=get_logits,
                        logits_metadata=metadata,
                    )
                    self.assertIsNotNone(result.input_copy_done)
                    self.assertTrue(result.token_logprobs.is_pinned())
                    self.assertEqual(
                        [call.args[0].shape[0] for call in get_logits.call_args_list],
                        [min(chunk_size, rows - i) for i in range(0, rows, chunk_size)],
                    )
                    output = LogitsProcessorOutput(next_token_logits=sampled)
                    result.write_input_to(output)
                    with pool.graph_pool_replay_scope():
                        graph.replay()
                    SchedulerBatchResultProcessor.move_logprobs_to_cpu(
                        None,
                        batch=SimpleNamespace(return_logprob=True),
                        logits_output=output,
                    )
                    expected = torch.log_softmax(logits, dim=-1)
                    self.assertEqual(
                        output.input_token_logprobs,
                        tuple(
                            expected[
                                torch.arange(rows, device="cuda"), token_ids
                            ].tolist()
                        ),
                    )
                    self.assertTrue(torch.equal(sampled, logits[sample_indices]))
                    for i, (lo, hi) in enumerate(((0, split), (split, rows))):
                        values, indices = expected[lo:hi].topk(
                            metadata.top_logprobs_nums[i]
                        )
                        self.assertEqual(
                            output.input_top_logprobs_val[i], values.tolist()
                        )
                        self.assertEqual(
                            output.input_top_logprobs_idx[i], indices.tolist()
                        )
                        self.assertEqual(
                            output.input_token_ids_logprobs_val[i],
                            expected[lo:hi, metadata.token_ids_logprobs[i]].tolist(),
                        )
                    self.assertIsNone(output.input_logprobs_copy_done)
            pool._teardown_borrow_pool()
        del graph, keep

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_borrow_preserves_planned_chunks(self):
        """Graph capacity controls borrowing without changing LM-head shapes."""
        rows, vocab = 8192, 4096
        states = torch.randint(-2, 3, (rows, 64), device="cuda").to(torch.bfloat16)
        weight = torch.randint(-2, 3, (vocab, 64), device="cuda").to(torch.bfloat16)
        logits = torch.mm(states, weight.T).float()
        expected = torch.log_softmax(logits, dim=-1)
        token_ids = torch.randint(0, vocab, (rows,), device="cuda")
        metadata = SimpleNamespace(
            sample_indices_cpu=[rows - 1],
            input_logprob_indices_cpu=list(range(rows)),
            extend_return_top_logprob=False,
            extend_token_ids_logprob=False,
            top_logprobs_nums=None,
            extend_logprob_pruned_lens_cpu=[rows],
            extend_input_logprob_token_ids_gpu=token_ids,
            token_ids_logprobs=None,
        )
        processor = InputLogprobProcessor(vocab_size=vocab)
        processor.enable_fast_input_logprobs = False
        processor.enable_logprobs_chunk = True
        processor.logprobs_chunk_size = rows

        def run_with_pool_run_of(nbytes, disabled, lm_head=None):
            handle = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            seed = torch.zeros(8, device="cuda")
            stream = torch.cuda.Stream()
            with (
                torch.cuda.stream(stream),
                torch.cuda.graph(graph, pool=handle, stream=stream),
            ):
                transient = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                keep = seed + 1
                del transient
            torch.cuda.synchronize()
            allocations_borrowed = []
            chunk_rows = []
            runs = pool.find_free_graph_pool_runs(handle)

            def get_logits(chunk, *_args, **_kwargs):
                chunk_rows.append(chunk.shape[0])
                projected = torch.mm(chunk, weight.T)
                converted = projected.float()
                allocations_borrowed.extend(
                    any(
                        lo <= tensor.data_ptr()
                        and tensor.data_ptr() + tensor.nbytes <= lo + size
                        for lo, size in runs
                    )
                    for tensor in (projected, converted)
                )
                return converted

            with (
                envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
                patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
                patch.object(
                    self.state, "disabled_reason", "test fallback" if disabled else None
                ),
                patch.object(
                    torch.cuda,
                    "mem_get_info",
                    side_effect=AssertionError(
                        "Chunk sizing must not query heap headroom"
                    ),
                ),
            ):
                # Precarve outside the measurement so the peak covers the forward.
                with pool.borrow_graph_pool(user="warmup"):
                    pass
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_stats()["allocated_bytes.all.current"]
                result, sampled = processor.forward(
                    pruned_states=states,
                    sample_indices=torch.tensor([rows - 1], device="cuda"),
                    input_logprob_indices=torch.arange(rows, device="cuda"),
                    token_to_seq_idx=[0] * rows,
                    lm_head=lm_head,
                    get_logits_fn=get_logits,
                    logits_metadata=metadata,
                )
                torch.cuda.synchronize()
                peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base
                with pool.graph_pool_replay_scope():
                    graph.replay()
                pool._teardown_borrow_pool()
            del graph, keep
            return result, sampled, peak, allocations_borrowed, chunk_rows

        expected_token_logprobs = expected[
            torch.arange(rows, device="cuda"), token_ids
        ].cpu()

        for chunk_size, nbytes, disabled, borrowing in (
            (rows, 512 << 20, False, True),
            (rows, 96 << 20, False, False),
            (rows, 16 << 20, False, False),
            (rows, 512 << 20, True, False),
            (32, 66 << 20, False, True),
            (32, 66 << 20, True, False),
        ):
            with self.subTest(chunk_size=chunk_size, nbytes=nbytes, disabled=disabled):
                processor.logprobs_chunk_size = chunk_size
                result, sampled, peak, borrowed, chunk_rows = run_with_pool_run_of(
                    nbytes, disabled
                )
                self.assertEqual(
                    chunk_rows,
                    [min(chunk_size, rows - i) for i in range(0, rows, chunk_size)],
                )
                self.assertTrue(all(value == borrowing for value in borrowed))
                if borrowing:
                    self.assertLessEqual(
                        peak, 2 * max(chunk_rows) * vocab * 4 + (2 << 20)
                    )
                    self.assertIsNotNone(result.input_copy_done)
                    result.input_copy_done.synchronize()
                    self.assertTrue(result.token_logprobs.is_pinned())
                else:
                    self.assertIsNone(result.input_copy_done)
                    self.assertTrue(result.token_logprobs.is_cuda)
                self.assertTrue(
                    torch.equal(result.token_logprobs.cpu(), expected_token_logprobs)
                )
                self.assertTrue(torch.equal(sampled, logits[-1:]))

        # LoRA has already prepared adapter metadata for each configured pass.
        processor.logprobs_chunk_size = rows // 2
        lm_head = Mock(spec=["set_lm_head_pass", "reset_lm_head_pass"])
        _, _, _, _, chunk_rows = run_with_pool_run_of(96 << 20, False, lm_head)
        self.assertEqual(chunk_rows, [rows // 2, rows // 2])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_borrow_recovers_from_arena_fragmentation(self):
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        seed = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(200 << 20, dtype=torch.uint8, device="cuda")
            keep = seed + 1
            del transient
        torch.cuda.synchronize()

        address, run_bytes = pool.find_free_graph_pool_runs(handle)[0]
        self.assertEqual(run_bytes, 200 << 20)
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
        ):
            # Unseeded 24/32/40 MiB segments strand 192 MiB before the 42 MiB request.
            for rows in (1500, 2000, 2500, 2600):
                with self.subTest(rows=rows), pool.borrow_graph_pool(user="test"):
                    first = torch.empty(
                        (rows, 4096), dtype=torch.float32, device="cuda"
                    )
                    second = torch.empty(
                        (rows, 4096), dtype=torch.float32, device="cuda"
                    )
                    self.assertTrue(
                        all(
                            address <= tensor.data_ptr()
                            and tensor.data_ptr() + tensor.nbytes <= address + run_bytes
                            for tensor in (first, second)
                        )
                    )
                    del first, second
            pool._teardown_borrow_pool()
        del graph, keep

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_replay_raises_when_borrowed_tensor_is_still_referenced(self):
        """Reject borrowed tensors that replay would silently overwrite."""
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        seed = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(48 << 20, dtype=torch.uint8, device="cuda")
            keep = seed + 1
            del transient
        torch.cuda.synchronize()

        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
        ):
            with pool.borrow_graph_pool(user="leaky"):
                leaked = torch.empty(1 << 20, device="cuda")
            with self.assertRaisesRegex(
                RuntimeError,
                f"graph replay: {leaked.nbytes} bytes",
            ):
                with pool.graph_pool_replay_scope():
                    pass

            # A fresh borrow re-arms the replay check after releasing the leak.
            del leaked
            with pool.borrow_graph_pool(user="clean"):
                released = torch.empty(1 << 20, device="cuda")
            del released
            with pool.graph_pool_replay_scope():
                pass
            pool._teardown_borrow_pool()
        del graph, keep

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_static_borrow_runs_serve_without_a_pool_snapshot(self):
        """Fixed extents serve borrows without consulting the shared pool."""
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            y = x + 1
            del transient
        torch.cuda.synchronize()

        runs = pool.find_free_graph_pool_runs(handle)
        self.assertTrue(runs)
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=None),
        ):
            pool.set_graph_pool_borrow_runs(runs)
            self.assertTrue(pool.graph_pool_borrow_enabled())
            with pool.borrow_graph_pool(user="test"):
                borrowed = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
                self.assertTrue(
                    any(
                        address <= borrowed.data_ptr() < address + nbytes
                        for address, nbytes in runs
                    )
                )
                del borrowed
            pool._teardown_borrow_pool()

        del graph, y

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_oversized_borrow_raises_oom_then_regular_allocation_succeeds(self):
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            y = x + 1
            del transient
        torch.cuda.synchronize()

        runs = pool.find_free_graph_pool_runs(handle)
        address, run_bytes = next(run for run in runs if run[1] >= 16 << 20)
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=None),
        ):
            pool.set_graph_pool_borrow_runs([(address, 8 << 20)])
            with self.assertRaises(torch.OutOfMemoryError):
                with pool.borrow_graph_pool(user="undersized-test"):
                    torch.empty(16 << 20, dtype=torch.uint8, device="cuda")

            pool.disable_graph_pool_borrow("undersized test pool")
            regular = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
            self.assertEqual(regular.nbytes, 16 << 20)
            del regular

        self.assertGreaterEqual(run_bytes, 16 << 20)
        del graph, y

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cross_stream_borrow_frees_resolve_before_pointer_reuse(self):
        """Deferred record_stream frees must not collide on the next borrow."""
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(graph, pool=handle, stream=stream),
        ):
            transient = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
            y = x + 1
            del transient
        torch.cuda.synchronize()

        side = torch.cuda.Stream()
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
        ):
            for _ in range(3):
                with pool.borrow_graph_pool(user="test"):
                    borrowed = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
                    # Stream-keyed segments allow side-stream copies, not allocations.
                    sink = torch.empty_like(borrowed)
                    with torch.cuda.stream(side):
                        sink.copy_(borrowed)
                    borrowed.record_stream(side)
                    del borrowed, sink
            # Regression: this used to fail with "Trying to free a pointer not
            # allocated here" after a deferred free was re-issued too early.
            torch.cuda.empty_cache()
            pool._teardown_borrow_pool()

        del graph, y

    def test_dflash_verify_output_buffers_predate_the_borrow_scope(self):
        """The chain verify buffers outlive the step, so creating them inside
        the borrow scope would let the next replay overwrite the accept
        length instead of raising."""
        events = []

        @contextmanager
        def recording_borrow(user):
            events.append(f"borrow:{user}")
            yield
            events.append("release")

        real_buffers = dflash_utils._get_or_create_chain_verify_buffers

        def recording_buffers(**kwargs):
            events.append("buffers")
            return real_buffers(**kwargs)

        def fake_sampling(**kwargs):
            kwargs["predicts"].fill_(3)
            kwargs["accept_index"].fill_(0)
            kwargs["accept_token_num"].fill_(1)

        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1)),
            top_ks=torch.ones(1, dtype=torch.int32),
            top_ps=torch.ones(1),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        )
        with (
            patch.object(dflash_utils, "borrow_graph_pool", recording_borrow),
            patch.object(
                dflash_utils,
                "_get_or_create_chain_verify_buffers",
                recording_buffers,
            ),
            patch.object(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True),
            patch.object(
                dflash_utils,
                "tree_speculative_sampling_target_only",
                fake_sampling,
            ),
        ):
            correct_len, bonus = (
                dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=torch.zeros((1, 2), dtype=torch.int64),
                    next_token_logits=torch.randn((2, 8)),
                    sampling_info=sampling_info,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                )
            )

        self.assertEqual(
            events, ["buffers", "borrow:DFLASH verify probabilities", "release"]
        )
        self.assertTrue(torch.equal(correct_len, torch.ones_like(correct_len)))
        self.assertTrue(torch.equal(bonus, torch.full_like(bonus, 3)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dflash_prewarm_falls_back_when_the_rehearsal_exhausts_the_pool(self):
        """A rehearsal too large for the pool must retire borrowing and
        re-measure, rather than crash startup or leave KV sizing without the
        headroom it now has to reserve."""
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 4
        worker.device = "cuda"
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                max_running_requests=2,
                max_decode_logits_rows=lambda: 8,
                sampling_prewarm_result=None,
            ),
            model_config=SimpleNamespace(vocab_size=32),
        )
        worker.model_runner = worker._target_worker.model_runner

        calls = []

        def rehearse(**kwargs):
            calls.append(pool.graph_pool_borrow_enabled())
            if len(calls) == 2:
                raise torch.OutOfMemoryError("rehearsal too large")
            return torch.zeros(2), torch.zeros(2)

        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=(1, 2)),
            patch.object(
                dflash_worker_v2,
                "compute_dflash_sampling_correct_drafts_and_bonus",
                rehearse,
            ),
        ):
            self.assertTrue(pool.graph_pool_borrow_enabled())
            result = worker.prewarm_sampling()
            self.assertFalse(pool.graph_pool_borrow_enabled())

        # Warm pass outside the pool, borrowed pass that OOMs, retry after the
        # fallback retires borrowing.
        self.assertEqual(calls, [False, True, False])
        # 2 rows x 4 draft tokens x 32 vocab x 4 bytes.
        self.assertEqual(result.sampling_input_bytes, 2 * 4 * 32 * 4)
        self.assertGreaterEqual(
            result.sampling_headroom_bytes, result.sampling_input_bytes
        )
        self.assertIs(worker.model_runner.sampling_prewarm_result, result)


if __name__ == "__main__":
    unittest.main()
