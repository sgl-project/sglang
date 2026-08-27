import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.benchmark import one_batch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


class TestTorchRunnerOwnership(unittest.TestCase):
    def test_one_batch_always_constructs_standard_model_runner(self):
        server_args = SimpleNamespace(
            tp_size=1,
            ep_size=1,
            enable_dp_attention=False,
            dp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            dcp_size=1,
            mem_fraction_static=0.7,
            tokenizer_path="test-tokenizer",
            tokenizer_mode="auto",
            trust_remote_code=False,
            is_startup_weight_load_overlap=False,
        )
        port_args = SimpleNamespace(nccl_port=12345)
        torch_runner = mock.MagicMock(max_total_num_tokens=1024)

        with (
            mock.patch.dict(os.environ, {"SGLANG_USE_MLX": "1"}),
            mock.patch.object(one_batch.ModelConfig, "from_server_args"),
            mock.patch.object(
                one_batch,
                "compute_dp_attention_world_info",
                return_value=(0, 1, 0, 1),
            ),
            mock.patch.object(one_batch, "ParallelState"),
            mock.patch.object(
                one_batch, "ModelRunner", return_value=torch_runner
            ) as model_runner_cls,
            mock.patch.object(one_batch, "get_tokenizer", return_value=object()),
            mock.patch.object(one_batch, "suppress_other_loggers"),
        ):
            runner, _ = one_batch.load_model(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
            )

        model_runner_cls.assert_called_once()
        self.assertIsInstance(runner, one_batch._TorchBenchRunner)
        self.assertIs(runner.torch_runner, torch_runner)

    def test_scheduler_always_constructs_standard_tp_worker(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = object()
        scheduler.ps = SimpleNamespace(gpu_id=0)
        scheduler.nccl_port = 12345
        worker = object()

        with (
            mock.patch.dict(os.environ, {"SGLANG_USE_MLX": "1"}),
            mock.patch(
                "sglang.srt.managers.tp_worker.TpModelWorker",
                return_value=worker,
            ) as worker_cls,
        ):
            scheduler.init_tp_model_worker()

        worker_cls.assert_called_once_with(
            server_args=scheduler.server_args,
            gpu_id=0,
            ps=scheduler.ps,
            nccl_port=12345,
        )
        self.assertIs(scheduler.tp_worker, worker)

    def test_mps_defaults_cannot_be_bypassed_by_legacy_environment(self):
        fake = SimpleNamespace(device="mps", _declare=mock.Mock())

        with mock.patch.dict(os.environ, {"SGLANG_USE_MLX": "1"}):
            ServerArgs._handle_mps_backends(fake)

        self.assertEqual(
            fake._declare.call_args_list,
            [
                mock.call(
                    "_handle_mps_backends",
                    disable_overlap_schedule=True,
                ),
                mock.call(
                    "_handle_mps_backends",
                    sampling_backend="pytorch",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
