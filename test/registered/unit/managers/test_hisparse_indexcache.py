import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import (
    build_indexcache_prefetch_layers_after,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_UNIT_PATH = Path(__file__).with_name("test_hisparse_unit.py")
_spec = importlib.util.spec_from_file_location("test_hisparse_unit_fixture", _UNIT_PATH)
_unit = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_unit)

DEVICE_BUFFER_SIZE = _unit.DEVICE_BUFFER_SIZE
LAYER_NUM = _unit.LAYER_NUM
MAX_NUM_REQS = _unit.MAX_NUM_REQS
SIZE = _unit.SIZE
TOP_K = _unit.TOP_K
_make_req = _unit._make_req


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


class TestHiSparseIndexCachePrefetchPlan(unittest.TestCase):
    def test_disabled_without_indexcache(self):
        cfg = SimpleNamespace(num_hidden_layers=4)

        self.assertEqual(
            build_indexcache_prefetch_layers_after(cfg, start_layer=0, layer_num=4),
            {},
        )

    def test_pattern_groups_consecutive_shared_layers(self):
        cfg = SimpleNamespace(
            num_hidden_layers=6,
            index_topk_pattern=["F", "S", "S", "F", "S", "F"],
        )

        self.assertEqual(
            build_indexcache_prefetch_layers_after(cfg, start_layer=0, layer_num=6),
            {0: (1, 2), 3: (4,)},
        )

    def test_freq_matches_existing_skip_topk_semantics(self):
        cfg = SimpleNamespace(num_hidden_layers=6, index_topk_freq=2)

        self.assertEqual(
            build_indexcache_prefetch_layers_after(cfg, start_layer=0, layer_num=6),
            {1: (2,), 3: (4,)},
        )

    def test_offset_matches_existing_skip_topk_semantics(self):
        cfg = SimpleNamespace(
            num_hidden_layers=8,
            index_topk_freq=3,
            index_skip_topk_offset=2,
        )

        self.assertEqual(
            build_indexcache_prefetch_layers_after(cfg, start_layer=0, layer_num=8),
            {1: (2, 3), 4: (5, 6)},
        )

    def test_pipeline_local_range_keeps_global_layer_ids(self):
        cfg = SimpleNamespace(
            num_hidden_layers=8,
            index_topk_pattern=["F", "S", "F", "S", "S", "F", "S", "F"],
        )

        self.assertEqual(
            build_indexcache_prefetch_layers_after(cfg, start_layer=2, layer_num=4),
            {2: (3, 4)},
        )


class TestHiSparseIndexCacheUnit(unittest.TestCase):
    setUpClass = classmethod(_unit.TestHiSparseUnit.setUpClass.__func__)
    tearDownClass = classmethod(_unit.TestHiSparseUnit.tearDownClass.__func__)
    setUp = _unit.TestHiSparseUnit.setUp

    _alloc_req_slot = _unit.TestHiSparseUnit._alloc_req_slot
    _free_req_slot = _unit.TestHiSparseUnit._free_req_slot
    _alloc_kv = _unit.TestHiSparseUnit._alloc_kv
    _kv_pattern = staticmethod(_unit.TestHiSparseUnit._kv_pattern)
    _populate_host_pool = _unit.TestHiSparseUnit._populate_host_pool
    _build_topk_tokens = _unit.TestHiSparseUnit._build_topk_tokens
    _make_batch_tensors = _unit.TestHiSparseUnit._make_batch_tensors
    _assert_kv_correct = _unit.TestHiSparseUnit._assert_kv_correct
    _cleanup_req = _unit.TestHiSparseUnit._cleanup_req
    _get_initial_sizes = _unit.TestHiSparseUnit._get_initial_sizes
    _assert_sizes_restored = _unit.TestHiSparseUnit._assert_sizes_restored
    _reset_layer_hot_cache_tags = _unit.TestHiSparseUnit._reset_layer_hot_cache_tags
    _reset_layer_hot_cache_tags_for_reqs = (
        _unit.TestHiSparseUnit._reset_layer_hot_cache_tags_for_reqs
    )
    _measure_cuda_graph_ms = staticmethod(_unit.TestHiSparseUnit._measure_cuda_graph_ms)

    def test_indexcache_prefetch_plan_pattern(self):
        """Explicit IndexCache pattern marks layer 1 as sharing layer 0 top-k."""
        cfg = SimpleNamespace(
            num_hidden_layers=LAYER_NUM, index_topk_pattern=["F", "S"]
        )
        self.coordinator.configure_indexcache_prefetch(cfg)

        self.assertTrue(self.coordinator._indexcache_prefetch_enabled)
        self.assertEqual(
            self.coordinator._indexcache_prefetch_layers_after,
            {0: (1,)},
        )

    def test_indexcache_prefetch_shared_layer_swap_in(self):
        """Full layer can preload KV for its following Shared layer."""
        initial = self._get_initial_sizes()
        cfg = SimpleNamespace(
            num_hidden_layers=LAYER_NUM, index_topk_pattern=["F", "S"]
        )
        self.coordinator.configure_indexcache_prefetch(cfg)

        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("indexcache-prefetch", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        self.coordinator.num_real_reqs[0] = rpi.shape[0]

        self.coordinator.prefetch_indexcache_shared_layers(rpi, sls, batch, layer_id=0)
        locs = self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)
        torch.cuda.synchronize()

        self.assertFalse(self.coordinator._indexcache_prefetch_pending[1])
        self._assert_kv_correct(
            locs[0], tokens, layer_id=1, count=TOP_K, msg="IndexCache prefetch: "
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "indexcache_prefetch")

    def test_cuda_graph_swap_in_selected_pages_long_seq(self):
        """HiSparse swap-in is captured and replayed by CUDA graph."""
        if not is_cuda():
            self.skipTest("CUDA graph test only supports CUDA.")

        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("cuda-graph-swap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        self.coordinator.num_real_reqs[0] = rpi.shape[0]

        self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)
        torch.cuda.synchronize()
        self._reset_layer_hot_cache_tags(req, layer_id=1)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            locs = self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=1, count=TOP_K, msg="CUDA graph swap: "
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "cuda_graph_swap")

    def test_cuda_graph_indexcache_prefetch_shared_layer_swap_in(self):
        """IndexCache prefetch side-stream work is part of the CUDA graph DAG."""
        if not is_cuda():
            self.skipTest("CUDA graph test only supports CUDA.")

        initial = self._get_initial_sizes()
        cfg = SimpleNamespace(
            num_hidden_layers=LAYER_NUM, index_topk_pattern=["F", "S"]
        )
        self.coordinator.configure_indexcache_prefetch(cfg)

        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("cuda-graph-indexcache-prefetch", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        self.coordinator.num_real_reqs[0] = rpi.shape[0]

        def graph_body():
            self.coordinator.prefetch_indexcache_shared_layers(
                rpi, sls, batch, layer_id=0
            )
            return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)

        graph_body()
        torch.cuda.synchronize()
        self._reset_layer_hot_cache_tags(req, layer_id=1)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            locs = graph_body()
        graph.replay()
        torch.cuda.synchronize()

        self.assertFalse(self.coordinator._indexcache_prefetch_pending[1])
        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0],
            tokens,
            layer_id=1,
            count=TOP_K,
            msg="CUDA graph IndexCache prefetch: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "cuda_graph_indexcache_prefetch")

    def test_cuda_graph_indexcache_prefetch_overlap_perf(self):
        """Opt-in CUDA graph microbenchmark for three HiSparse overlap modes."""
        if not is_cuda():
            self.skipTest("CUDA graph test only supports CUDA.")
        if not _env_flag("SGLANG_RUN_HISPARSE_CUDA_GRAPH_OVERLAP_PERF"):
            self.skipTest(
                "Set SGLANG_RUN_HISPARSE_CUDA_GRAPH_OVERLAP_PERF=1 to run "
                "the CUDA graph overlap microbenchmark."
            )
        if not hasattr(torch.cuda, "_sleep"):
            self.skipTest("torch.cuda._sleep is required for synthetic compute.")

        initial = self._get_initial_sizes()
        cfg = SimpleNamespace(
            num_hidden_layers=LAYER_NUM, index_topk_pattern=["F", "S"]
        )
        self.coordinator.configure_indexcache_prefetch(cfg)

        num_reqs = int(os.getenv("SGLANG_HISPARSE_OVERLAP_PERF_NUM_REQS", "3"))
        num_reqs = max(1, min(num_reqs, MAX_NUM_REQS))
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        if fill_len * num_reqs > SIZE:
            self.skipTest(
                f"Fixture SIZE={SIZE} is too small for num_reqs={num_reqs}, "
                f"fill_len={fill_len}."
            )

        reqs = [
            _make_req(f"overlap-perf-{i}", list(range(fill_len)))
            for i in range(num_reqs)
        ]
        kv_locs = []
        for req in reqs:
            self._alloc_req_slot(req)
            kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
            kv_locs.append(kv_loc)
            self._populate_host_pool(req, fill_len)
            self.coordinator.admit_request_direct(req)

        row = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        batch = row.unsqueeze(0).repeat(num_reqs, 1).contiguous()
        rpi, sls = self._make_batch_tensors(reqs, [fill_len] * num_reqs)
        self.coordinator.num_real_reqs[0] = rpi.shape[0]

        sleep_cycles = int(
            os.getenv("SGLANG_HISPARSE_OVERLAP_PERF_SLEEP_CYCLES", "200000")
        )
        iters = int(os.getenv("SGLANG_HISPARSE_OVERLAP_PERF_ITERS", "50"))
        min_speedup = float(
            os.getenv("SGLANG_HISPARSE_OVERLAP_PERF_MIN_SPEEDUP", "1.10")
        )

        def synthetic_current_layer_compute():
            torch.cuda._sleep(sleep_cycles)

        def no_hisparse_body():
            synthetic_current_layer_compute()

        def no_overlap_body():
            self._reset_layer_hot_cache_tags_for_reqs(reqs, layer_id=1)
            synthetic_current_layer_compute()
            return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)

        def overlap_body():
            self._reset_layer_hot_cache_tags_for_reqs(reqs, layer_id=1)
            self.coordinator.prefetch_indexcache_shared_layers(
                rpi, sls, batch, layer_id=0
            )
            synthetic_current_layer_compute()
            return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id=1)

        try:
            no_hisparse_body()
            no_overlap_body()
            overlap_body()
            torch.cuda.synchronize()

            no_hisparse_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(no_hisparse_graph):
                no_hisparse_body()

            no_overlap_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(no_overlap_graph):
                no_overlap_locs = no_overlap_body()

            overlap_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(overlap_graph):
                overlap_locs = overlap_body()

            no_hisparse_ms = self._measure_cuda_graph_ms(no_hisparse_graph, iters=iters)
            no_overlap_ms = self._measure_cuda_graph_ms(no_overlap_graph, iters=iters)
            overlap_ms = self._measure_cuda_graph_ms(overlap_graph, iters=iters)
            torch.cuda.synchronize()

            self.assertTrue(torch.all(no_overlap_locs[:, :TOP_K] >= 0))
            self.assertTrue(torch.all(overlap_locs[:, :TOP_K] >= 0))

            speedup = no_overlap_ms / overlap_ms if overlap_ms > 0 else float("inf")
            no_overlap_overhead = (
                no_overlap_ms / no_hisparse_ms if no_hisparse_ms > 0 else float("inf")
            )
            overlap_overhead = (
                overlap_ms / no_hisparse_ms if no_hisparse_ms > 0 else float("inf")
            )
            print(
                "cuda_graph_indexcache_overlap_perf: "
                f"no_hisparse={no_hisparse_ms:.4f} ms, "
                f"hisparse_no_overlap={no_overlap_ms:.4f} ms, "
                f"hisparse_overlap={overlap_ms:.4f} ms, "
                f"overlap_speedup={speedup:.3f}x, "
                f"no_overlap_overhead={no_overlap_overhead:.3f}x, "
                f"overlap_overhead={overlap_overhead:.3f}x, "
                f"sleep_cycles={sleep_cycles}, "
                f"num_reqs={num_reqs}, iters={iters}"
            )
            self.assertGreater(
                speedup,
                min_speedup,
                "CUDA graph IndexCache overlap microbenchmark did not show enough "
                f"speedup: no_overlap={no_overlap_ms:.4f} ms, "
                f"overlap={overlap_ms:.4f} ms, speedup={speedup:.3f}x, "
                f"min_speedup={min_speedup:.3f}x, sleep_cycles={sleep_cycles}, "
                f"num_reqs={num_reqs}, iters={iters}",
            )
        finally:
            for req, kv_loc in zip(reqs, kv_locs):
                self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "cuda_graph_indexcache_overlap_perf")


if __name__ == "__main__":
    unittest.main()
