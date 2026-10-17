import contextlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine
from sglang.srt.managers import data_parallel_controller
from sglang.srt.weight_cache.protocol import compute_local_gpu_id

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGpuLaunchPlacement(unittest.TestCase):
    # nnodes, PP, TP, node rank, base GPU, GPU stride, expected local IDs
    cases = (
        (1, 2, 2, 0, 0, 2, [0, 2, 4, 6]),
        (1, 4, 1, 0, 1, 3, [1, 4, 7, 10]),
        (2, 4, 2, 1, 0, 2, [0, 2, 4, 6]),
        (4, 2, 4, 3, 0, 2, [0, 2]),
        (1, 2, 2, 0, 0, 1, [0, 1, 2, 3]),
        (1, 1, 4, 0, 3, 2, [3, 5, 7, 9]),
    )

    @contextlib.contextmanager
    def launch_boundaries(self, module, case):
        nnodes, pp, tp, node_rank, base, step, _ = case
        parallel = SimpleNamespace(
            nnodes=nnodes,
            pp_size=pp,
            tp_size=tp,
            node_rank=node_rank,
            num_dp_ranks=1,
            attn_dp_enabled=False,
            moe_dp_size=1,
            ep_size=1,
            ep_join_rank_offset=0,
        )
        device = SimpleNamespace(base_gpu_id=base, gpu_id_step=step)
        execution = SimpleNamespace(
            moe=SimpleNamespace(ep_join_mode="none", is_ep_scale_joiner=False),
            features=SimpleNamespace(enable_memory_saver=False),
        )
        process = Mock(side_effect=lambda **kwargs: Mock(pid=1234))
        reader = Mock()
        reader.recv.return_value = {"max_total_num_tokens": 1, "max_req_input_len": 1}
        adapter = SimpleNamespace(configure_subprocess=contextlib.nullcontext)
        with (
            patch.object(module, "get_parallel", return_value=parallel),
            patch.object(module, "get_device", return_value=device),
            patch.object(module, "get_exec", return_value=execution),
            patch.object(module.mp, "Process", process),
            patch.object(module.mp, "Pipe", return_value=(reader, object())),
            patch.object(
                module.TorchMemorySaverAdapter, "create", return_value=adapter
            ),
            patch.object(
                module, "maybe_reindex_device_id", side_effect=contextlib.nullcontext
            ),
            patch.object(
                module.numa_utils,
                "configure_subprocess",
                side_effect=lambda *args: contextlib.nullcontext(),
            ),
        ):
            yield process

    def test_engine_applies_stride_across_pipeline_stages(self):
        for case in self.cases:
            with (
                self.subTest(case=case),
                self.launch_boundaries(engine, case) as process,
            ):
                engine.Engine._launch_scheduler_processes(object(), object(), Mock())
                assigned = [call.kwargs["args"][2] for call in process.call_args_list]
                self.assertEqual(assigned, case[-1])
                self.assertEqual(len(assigned), len(set(assigned)))

    def test_dp_group_preserves_replica_offset_with_pipeline_stride(self):
        for case in self.cases:
            for replica_offset in (0, 12):
                with (
                    self.subTest(case=case, replica_offset=replica_offset),
                    self.launch_boundaries(data_parallel_controller, case) as process,
                    patch.object(
                        data_parallel_controller,
                        "aggregate_scheduler_startup_times",
                        return_value={},
                    ),
                ):
                    controller = (
                        data_parallel_controller.DataParallelController.__new__(
                            data_parallel_controller.DataParallelController
                        )
                    )
                    controller.env_lock = threading.Lock()
                    controller.scheduler_procs = []
                    controller.run_scheduler_process_func = Mock()
                    controller.launch_tensor_parallel_group(
                        object(), object(), replica_offset, 1
                    )
                    assigned = [
                        call.kwargs["args"][2] for call in process.call_args_list
                    ]
                    self.assertEqual(
                        assigned, [replica_offset + gpu_id for gpu_id in case[-1]]
                    )
                    self.assertEqual(len(assigned), len(set(assigned)))

    def test_weight_cache_daemons_use_the_same_pipeline_stride(self):
        for case in self.cases:
            with self.subTest(case=case), self.launch_boundaries(engine, case):
                _, _, _, node_rank, base, step, expected = case
                pp_ranks, tp_ranks, pp_local, tp_local = engine._calculate_rank_ranges(
                    node_rank
                )
                assigned = [
                    compute_local_gpu_id(
                        pp_rank,
                        tp_rank,
                        pp_local,
                        tp_local,
                        base_gpu_id=base,
                        gpu_id_step=step,
                    )
                    for pp_rank in pp_ranks
                    for tp_rank in tp_ranks
                ]
                self.assertEqual(assigned, expected)


if __name__ == "__main__":
    unittest.main()
