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
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


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

    def test_external_graph_storage_can_disable_borrowing(self):
        with (
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
            patch.object(pool, "is_cuda", return_value=True),
            patch.object(pool, "get_global_graph_memory_pool", return_value=(1, 2)),
        ):
            self.assertTrue(pool.graph_pool_borrow_enabled())
            pool.disable_graph_pool_borrow("graph storage is externally managed")
            self.assertFalse(pool.graph_pool_borrow_enabled())

    def test_eagle_non_greedy_probabilities_use_borrow_scope(self):
        state = {"active": False, "users": []}

        @contextmanager
        def tracking_borrow(*, user):
            self.assertFalse(state["active"])
            state["active"] = True
            state["users"].append(user)
            try:
                yield
            finally:
                state["active"] = False

        import torch.nn.functional as F

        real_softmax = F.softmax

        def checked_softmax(*args, **kwargs):
            self.assertTrue(state["active"])
            return real_softmax(*args, **kwargs)

        def fake_sampling(**kwargs):
            self.assertTrue(state["active"])
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
            patch.object(eagle_utils, "borrow_graph_pool", tracking_borrow),
            patch.object(eagle_utils, "get_spec", return_value=spec_config),
            patch("torch.nn.functional.softmax", side_effect=checked_softmax),
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

        self.assertEqual(state["users"], ["EAGLE probability borrow"])
        self.assertFalse(state["active"])
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
        with torch.cuda.stream(stream), torch.cuda.graph(
            graph, pool=handle, stream=stream
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
        with torch.cuda.stream(stream), torch.cuda.graph(
            graph, pool=handle, stream=stream
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
    def test_cross_stream_borrow_frees_resolve_before_pointer_reuse(self):
        """Deferred record_stream frees must not collide on the next borrow."""
        handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(
            graph, pool=handle, stream=stream
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


if __name__ == "__main__":
    unittest.main()
