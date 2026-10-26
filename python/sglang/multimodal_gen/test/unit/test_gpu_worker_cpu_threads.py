# SPDX-License-Identifier: Apache-2.0
"""Contracts for per-worker host CPU setup: intra-op threads and NUMA binding."""

import gc
import os
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _bind_worker_to_gpu_numa_node,
    _worker_cpu_intra_op_threads,
)
from sglang.multimodal_gen.runtime.managers.request_gc import (
    RequestGarbageCollector,
    request_gc,
)
from sglang.test.test_utils import CustomTestCase


class TestRequestGarbageCollection(CustomTestCase):
    def setUp(self):
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        self.addCleanup(gc.set_threshold, *gc.get_threshold())
        gc.enable()

    def test_deferred_cycles_and_reference_counting(self):
        class Object:
            pass

        collector = RequestGarbageCollector()
        gc.collect()
        gc.set_threshold(10, 1, 1)
        with collector.defer():
            before = gc.get_stats()
            refs = []
            for _ in range(1000):
                obj = Object()
                obj.cycle = obj
                refs.append(weakref.ref(obj))
            del obj
            self.assertEqual(gc.get_stats(), before)
            self.assertTrue(all(ref() is not None for ref in refs))
            obj = Object()
            acyclic = weakref.ref(obj)
            del obj
            self.assertIsNone(acyclic())
        self.assertTrue(gc.isenabled())
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_nested_failure_preserves_original_gc_settings(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                gc.enable() if enabled else gc.disable()
                thresholds = gc.get_threshold()
                with request_gc.defer():
                    with self.assertRaisesRegex(RuntimeError, "request failed"):
                        with request_gc.defer():
                            raise RuntimeError("request failed")
                    self.assertFalse(gc.isenabled())
                self.assertEqual(gc.isenabled(), enabled)
                self.assertEqual(gc.get_threshold(), thresholds)

    def test_warmup_forward_collects_only_after_final_or_failure(self):
        """Warmup cycles survive forward but not its final or failed boundary."""

        class Temporary:
            pass

        # Keep automatic GC from satisfying the explicit boundary assertions.
        gc.set_threshold(2**31 - 1, 2**31 - 1, 2**31 - 1)
        worker = gpu_worker.GPUWorker.__new__(gpu_worker.GPUWorker)
        worker.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                supports_auto_residency=False,
                supports_sequential_multi_output_inference=lambda: False,
            )
        )
        worker.is_output_rank = False
        worker._release_warmup_pool_before_serving = False
        worker._realtime_sessions = SimpleNamespace(attach=lambda req: None)
        with (
            patch.object(gpu_worker.current_platform, "is_cpu", return_value=True),
            patch.object(gpu_worker.IPC_A2A, "check_timeout"),
            patch.object(gpu_worker.IPC_A2A_MULTI, "check_timeout"),
        ):
            for internal in (False, True):
                for last, fail in ((False, False), (True, False), (False, True)):
                    with self.subTest(internal=internal, last=last, fail=fail):
                        refs = []
                        req = gpu_worker.Req(
                            sampling_params=SamplingParams(),
                            request_id="warmup-gc",
                            is_warmup=not internal,
                            suppress_logs=True,
                            extra={
                                "server_based_warmup": True,
                                "server_internal_prewarm": internal,
                                "warmup_last": last,
                            },
                        )

                        def forward(batch, server_args):
                            self.assertFalse(gc.isenabled())
                            before = gc.get_stats()
                            obj = Temporary()
                            obj.cycle = obj
                            refs.append(weakref.ref(obj))
                            del obj
                            self.assertIsNotNone(refs[0]())
                            self.assertEqual(gc.get_stats(), before)
                            if fail:
                                raise RuntimeError("recoverable warmup failure")
                            return batch

                        worker.pipeline = SimpleNamespace(forward=forward)
                        try:
                            with patch.object(
                                gc, "collect", wraps=gc.collect
                            ) as collect:
                                output = worker.execute_forward([req], return_req=True)
                                self.assertEqual(collect.call_count, int(last or fail))
                            self.assertEqual(len(refs), 1)
                            self.assertEqual(refs[0]() is None, last or fail)
                            self.assertTrue(gc.isenabled())
                            if fail:
                                self.assertIn(
                                    "recoverable warmup failure", output.error
                                )
                            else:
                                self.assertIs(output, req)
                        finally:
                            request_gc.collect_after_warmup()

            for fail in (False, True):
                with self.subTest(sequential=True, fail=fail):
                    refs = []
                    req = gpu_worker.Req(
                        sampling_params=SamplingParams(), is_warmup=True
                    )

                    class Outputs:
                        def __next__(self):
                            if gc.isenabled():
                                raise AssertionError("GC enabled during forward")
                            if fail:
                                raise RuntimeError("sequential warmup failure")
                            return gpu_worker.OutputBatch()

                        def close(self):
                            # Only collection after close can reclaim this cycle.
                            obj = Temporary()
                            obj.cycle = obj
                            refs.append(weakref.ref(obj))

                    worker.pipeline = SimpleNamespace(
                        forward_batch_sequentially=lambda *args: Outputs()
                    )
                    request_gc.mark_warmup()
                    try:
                        with patch.object(gc, "collect", wraps=gc.collect) as collect:
                            if fail:
                                with self.assertRaisesRegex(
                                    RuntimeError, "sequential warmup failure"
                                ):
                                    list(worker.execute_forward_sequentially([req]))
                            else:
                                outputs = list(
                                    worker.execute_forward_sequentially([req])
                                )
                                self.assertEqual(len(outputs), 1)
                            self.assertEqual(collect.call_count, int(fail))
                        self.assertEqual(len(refs), 1)
                        self.assertEqual(refs[0]() is None, fail)
                        self.assertTrue(gc.isenabled())
                    finally:
                        request_gc.collect_after_warmup()

    def test_overlap_restores_gc_only_after_last_thread(self):
        for first_to_finish in (0, 1):
            with self.subTest(first_to_finish=first_to_finish):
                entered = [threading.Event(), threading.Event()]
                release = [threading.Event(), threading.Event()]

                def run(index):
                    with request_gc.defer():
                        entered[index].set()
                        self.assertTrue(release[index].wait(10))
                        self.assertFalse(gc.isenabled())

                with ThreadPoolExecutor(2) as executor:
                    futures = [executor.submit(run, i) for i in range(2)]
                    try:
                        self.assertTrue(all(event.wait(10) for event in entered))
                        release[first_to_finish].set()
                        futures[first_to_finish].result(timeout=10)
                        self.assertFalse(gc.isenabled())
                    finally:
                        for event in release:
                            event.set()
                        for future in futures:
                            future.result(timeout=10)
                self.assertTrue(gc.isenabled())

    @unittest.skipUnless(os.name == "posix", "requires fork")
    def test_fork_does_not_inherit_disabled_gc(self):
        with request_gc.defer():
            pid = os.fork()
            if pid == 0:
                restored = gc.isenabled()
                with request_gc.defer():
                    restored &= not gc.isenabled()
                os._exit(int(not (restored and gc.isenabled())))
            self.assertEqual(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), 0)
            self.assertFalse(gc.isenabled())


class TestWorkerStartup(CustomTestCase):
    def test_reclaims_loading_cycles_before_first_request(self):
        class LoadingObject:
            pass

        garbage_refs = []

        def load_model(worker):
            garbage = LoadingObject()
            garbage.cycle = garbage
            garbage_refs.append(weakref.ref(garbage))
            worker.pipeline = LoadingObject()
            # survive a young collection like objects retained during imports
            gc.collect(1)

        # prevent an automatic collection from making the unfixed code pass
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        thresholds = gc.get_threshold()
        gc.disable()
        group = SimpleNamespace(cpu_group=None)
        with (
            patch.object(gpu_worker, "init_metrics", return_value=None),
            patch.object(gpu_worker.current_platform, "is_cpu", return_value=True),
            patch.object(gpu_worker, "get_sp_group", return_value=group),
            patch.object(gpu_worker, "get_tp_group", return_value=group),
            patch.object(gpu_worker, "get_cfg_group", return_value=group),
            patch.object(gpu_worker.GPUWorker, "init_device_and_model", load_model),
            patch.object(gpu_worker.GPUWorker, "_update_lora_metrics"),
        ):
            worker = gpu_worker.GPUWorker(
                local_rank=0,
                rank=0,
                master_port=30005,
                server_args=SimpleNamespace(
                    num_gpus=1, dp_size=1, async_output_save=False
                ),
            )

        self.assertIsNone(garbage_refs[0]())
        self.assertIsInstance(worker.pipeline, LoadingObject)
        self.assertEqual(gc.get_threshold(), thresholds)
        self.assertFalse(gc.isenabled())


class TestWorkerCpuIntraOpThreads(unittest.TestCase):
    def test_divides_host_cores_across_colocated_workers(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.cpu_count", return_value=128),
        ):
            import os

            os.environ.pop("OMP_NUM_THREADS", None)
            self.assertEqual(_worker_cpu_intra_op_threads(8), 16)
            self.assertEqual(_worker_cpu_intra_op_threads(4), 16)  # capped
            self.assertEqual(_worker_cpu_intra_op_threads(128), 1)
            self.assertEqual(_worker_cpu_intra_op_threads(256), 1)  # floor

    def test_single_gpu_keeps_cap(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.cpu_count", return_value=8),
        ):
            import os

            os.environ.pop("OMP_NUM_THREADS", None)
            self.assertEqual(_worker_cpu_intra_op_threads(1), 8)

    def test_explicit_omp_setting_wins(self):
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "32"}):
            self.assertIsNone(_worker_cpu_intra_op_threads(8))


class TestWorkerNumaBinding(unittest.TestCase):
    def test_explicit_node_is_indexed_by_local_gpu(self):
        with patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind:
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=[0, 0, 1, 1]), local_rank=2
            )
        self.assertEqual(node, 1)
        bind.assert_called_once_with(1)

    def test_auto_detection_queries_the_worker_gpu(self):
        with (
            patch.dict("os.environ", {"SGLANG_AUTO_NUMA_BIND": "1"}),
            patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True),
            patch(
                "sglang.srt.utils.numa_utils._query_numa_node_for_gpu",
                return_value=[1],
            ) as query,
            patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind,
        ):
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=None), local_rank=3
            )
        query.assert_called_once_with(3)
        self.assertEqual(node, 1)
        bind.assert_called_once_with(1)

    def test_cpu_affinity_still_binds_without_mempolicy(self):
        with (
            patch.dict("os.environ", {"SGLANG_AUTO_NUMA_BIND": "1"}),
            patch("sglang.srt.utils.numa_utils._is_cuda", True),
            patch("os.path.isdir", return_value=True),
            patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=False),
            patch(
                "sglang.srt.utils.numa_utils.shutil.which",
                return_value="/usr/bin/numactl",
            ),
            patch(
                "sglang.srt.utils.numa_utils._query_numa_node_for_gpu",
                return_value=[1],
            ),
            patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind,
        ):
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=None), local_rank=1
            )
        self.assertEqual(node, 1)
        bind.assert_called_once_with(1)

    def test_disabled_auto_bind_leaves_worker_unbound(self):
        with (
            patch.dict("os.environ", {"SGLANG_AUTO_NUMA_BIND": "0"}),
            patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind,
        ):
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=None), local_rank=0
            )
        self.assertIsNone(node)
        bind.assert_not_called()


if __name__ == "__main__":
    unittest.main()
