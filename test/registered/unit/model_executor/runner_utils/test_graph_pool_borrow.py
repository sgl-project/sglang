"""CUDA graph-pool borrowing allocator and lifetime regression tests."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.environ import envs
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
        self._reset_borrow_state()

    def tearDown(self):
        try:
            if torch.cuda.is_available():
                pool._teardown_borrow_pool()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        finally:
            self._reset_borrow_state()

    @staticmethod
    def _reset_borrow_state():
        pool._active_graph_pool_user = None
        pool._borrow_stub = None
        pool._borrow_mem_pool = None
        pool._borrow_disabled_reason = None
        pool._borrow_static_runs = None
        pool._borrow_extents_total = 0
        pool._largest_logged_graph_pool_borrow = 0

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
            patch.object(
                pool, "_borrow_stub", MagicMock(cursor_bytes=0, freed_bytes=0)
            ),
            patch.object(pool, "_borrow_mem_pool", None),
            patch.object(pool.torch.cuda, "MemPool"),
            patch.object(pool.torch.cuda, "use_mem_pool"),
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

        with (
            patch.object(pool, "graph_pool_borrow_enabled", return_value=True),
            patch.object(pool, "_borrow_stub", stub),
            patch.object(pool, "_borrow_mem_pool", mem_pool),
            patch.object(pool, "_borrow_extents_total", 1000),
            patch.object(pool, "_teardown_borrow_pool") as teardown,
            patch.object(pool.torch, "empty"),
            patch.object(pool.torch.cuda, "use_mem_pool"),
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
            pool._borrow_static_runs = None

        with patch.object(
            pool, "_teardown_borrow_pool", side_effect=reset_static_runs
        ) as teardown:
            pool.set_graph_pool_borrow_runs(runs)

        teardown.assert_called_once_with()
        self.assertEqual(pool._borrow_static_runs, [(0x2000, 8192), (0x1000, 4096)])

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
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            patch(
                "sgl_kernel.tree_speculative_sampling_target_only",
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
        reserved_before = torch.cuda.memory_reserved(device_id)
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=handle),
            patch.object(pool, "_borrow_mem_pool", None),
        ):
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

            # Captures retire the persistent borrow pool. Its storage aliases
            # existing graph-pool runs, so the reserved footprint is unchanged.
            pool._teardown_borrow_pool()

        self.assertEqual(torch.cuda.memory_reserved(device_id), reserved_before)
        del graph, y

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
            patch.object(pool, "_borrow_static_runs", None),
            patch.object(pool, "_borrow_mem_pool", None),
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
            patch.object(pool, "_borrow_mem_pool", None),
        ):
            for _ in range(3):
                with pool.borrow_graph_pool(user="test"):
                    borrowed = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
                    with torch.cuda.stream(side):
                        widened = borrowed.to(torch.int32)
                    borrowed.record_stream(side)
                    del borrowed, widened
            # Regression: this used to fail with "Trying to free a pointer not
            # allocated here" after a deferred free was re-issued too early.
            torch.cuda.empty_cache()

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
